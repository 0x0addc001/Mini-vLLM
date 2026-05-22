"""
全局块管理器 (Global Block Manager)

负责跨 GPU 的 KV cache 块状态同步、全局 LRU 冷块选择和 swap 目标决策
设计要点：
1. 全局管理节点 (master_rank) 可配置，默认 rank 0，支持灾后迁移
2. 全局页表：hash -> 该 hash 的块分布在哪些 GPU 上（一对多）
3. NVLink 拓扑感知的 swap 目标选择：NVLink 直连 > 同 Socket PIX > 跨 Socket NODE
4. 三级内存池枯竭应对：递归 swap -> 远端 LRU 覆盖 -> CPU fallback signal
"""

import logging
import time
import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class BlockLocation:
    """描述一个 KV 块在集群中的物理位置"""
    gpu_id: int              # 块所在的 GPU rank
    block_id: int            # 该 GPU 上的物理块索引
    hash: int                # 块内容的 hash 值（-1 表示 partial block / 未 hash）
    last_access_time: float  # 最近访问时间戳（用于 LRU 驱逐决策）


# ============================================================================
# 全局块管理器
# ============================================================================

class GlobalBlockManager:
    """
    集群内所有 GPU 的 KV cache 块全局管理器。

    职责：
    - 维护全局页表 (hash -> 所有 BlockLocation)
    - 维护每 GPU 空闲块数量
    - 提供拓扑感知的冷块驱逐选择
    - 记录块迁移后的位置变更
    - 支持全局管理节点可配置与故障迁移
    """

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def __init__(
        self,
        rank: int,
        world_size: int,
        num_blocks_per_gpu: int,
        master_rank: int = 0,
        nvlink_pairs: Optional[List[Tuple[int, int]]] = None,
        socket_groups: Optional[List[List[int]]] = None,
    ):
        """
        参数:
            rank: 当前进程的 GPU rank
            world_size: GPU 总数
            num_blocks_per_gpu: 每 GPU 的物理块总数
            master_rank: 全局管理节点的 rank（默认 0）
            nvlink_pairs: NVLink 直连 GPU 对列表，如 [(0,2), (1,3), (4,5), (6,7)]
            socket_groups: 同 CPU Socket 的 GPU 分组，如 [[0,1,2,3], [4,5,6,7]]
        """
        self.rank = rank
        self.world_size = world_size
        self.num_blocks_per_gpu = num_blocks_per_gpu
        self.master_rank = master_rank

        # ---------- 拓扑信息 ----------
        self.nvlink_pairs: Set[Tuple[int, int]] = set(nvlink_pairs or [])
        self.socket_groups: List[List[int]] = socket_groups or [list(range(world_size))]

        # 建立 GPU -> NVLink 伙伴的快速查找
        self.nvlink_partner: Dict[int, int] = {}
        for a, b in self.nvlink_pairs:
            self.nvlink_partner[a] = b
            self.nvlink_partner[b] = a

        # 建立 GPU -> 同 Socket 其他 GPU 的快速查找（排除自己，优先 PIX）
        self.same_socket_gpus: Dict[int, List[int]] = {}
        for group in self.socket_groups:
            for gpu in group:
                self.same_socket_gpus[gpu] = [g for g in group if g != gpu]

        # ---------- 全局状态（仅 master_rank 维护权威副本） ----------
        # 全局页表：hash -> 该 hash 所在的所有位置
        self.global_page_table: Dict[int, List[BlockLocation]] = {}
        # 每 GPU 的空闲块计数
        self.free_blocks_per_gpu: List[int] = [num_blocks_per_gpu] * world_size
        # 每 GPU 的已用块访问时间：gpu_id -> {block_id: last_access_time}
        self.block_access_time: List[Dict[int, float]] = [{} for _ in range(world_size)]
        # 每 GPU 的块 hash 记录：gpu_id -> {block_id: hash}
        self.block_hash: List[Dict[int, int]] = [{} for _ in range(world_size)]

        # ---------- 同步配置 ----------
        self.sync_interval: int = 10        # 多少轮调度后广播一次页表
        self._sync_counter: int = 0         # 当前轮次计数

        # ---------- 管理节点心跳 ----------
        self.master_heartbeat: float = time.time()
        self.heartbeat_timeout: float = 100.0  # 心跳超时（秒），超时后触发管理节点迁移

        # if self.is_master:
        #     self.broadcast_page_table()  # 初始化时就同步一次，让所有 rank 拿到一致的状态

    # ------------------------------------------------------------------
    # 管理节点故障迁移
    # ------------------------------------------------------------------

    def check_master_health(self) -> bool:
        """
        检查当前 master_rank 是否健康
        如果心跳超时，选择新的管理节点
        返回当前节点是否成为新 master
        """
        if self.rank == self.master_rank:
            # 我是 master，刷新自己的心跳
            self.master_heartbeat = time.time()
            return True

        # 非 master 节点检查心跳，所有 rank 都参与广播
        try:
            heartbeat = self._broadcast_master_heartbeat()
            if time.time() - heartbeat > self.heartbeat_timeout:
                # master 故障，发起选举
                new_master = self._elect_new_master()
                old_master = self.master_rank
                self.master_rank = new_master
                if self.rank == new_master:
                    logger.warning("master failover: rank %s -> rank %s (me)", old_master, new_master)
                    return True
                else:
                    logger.warning("master failover: rank %s -> rank %s", old_master, new_master)
                    return False
        except Exception:
            pass
        return False

    def _broadcast_master_heartbeat(self) -> float:
        """广播 master 的心跳时间戳到所有 rank"""
        heartbeat_tensor = torch.tensor([self.master_heartbeat], dtype=torch.float64)
        dist.broadcast(heartbeat_tensor, src=self.master_rank)
        return heartbeat_tensor.item()

    def _elect_new_master(self) -> int:
        """
        选择新的管理节点
        策略：选择最小的存活 rank。简化实现，仅 fallback 到 (old_master + 1) % world_size
        """
        # 简化：直接轮转到下一个 rank
        return (self.master_rank + 1) % self.world_size

    def set_master_rank(self, new_master: int):
        """手动指定新的管理节点（用于灾后迁移）"""
        self.master_rank = new_master

    @property
    def is_master(self) -> bool:
        return self.rank == self.master_rank

    # ------------------------------------------------------------------
    # 全局页表同步
    # ------------------------------------------------------------------

    def gather_local_state(self):
        """
        收集所有 rank 的空闲块数到 self.free_blocks_per_gpu。
        由 master_rank 在 broadcast_page_table 之前调用。
        """
        if self.world_size == 1:
            return

        local_free = torch.tensor(
            [self.free_blocks_per_gpu[self.rank]],
            dtype=torch.int64,
            device=f'cuda:{self.rank}'
        )
        all_free = torch.zeros(self.world_size, dtype=torch.int64, device=f'cuda:{self.rank}')
        dist.all_gather_into_tensor(all_free, local_free)

        if self.is_master:
            for i in range(self.world_size):
                self.free_blocks_per_gpu[i] = all_free[i].item()

    def broadcast_page_table(self):
        """
        管理节点将全局页表广播到所有 GPU,，其他节点接收并更新本地缓存
        广播前先收集所有 rank 的最新空闲块状态
        """
        self.gather_local_state()
        data = (
            self.global_page_table,
            self.free_blocks_per_gpu,
            self.block_access_time,
            self.block_hash,
            self.master_rank,
        )
        # 使用 broadcast_object_list 广播 Python 对象
        obj_list = [data]
        dist.broadcast_object_list(obj_list, src=self.master_rank)
        if not self.is_master:
            (
                self.global_page_table,
                self.free_blocks_per_gpu,
                self.block_access_time,
                self.block_hash,
                self.master_rank,
            ) = obj_list[0]

    def sync_from_master(self):
        """非管理节点从管理节点拉取最新全局页表"""
        self.broadcast_page_table()

    def maybe_sync(self):
        """周期性检查是否需要同步页表"""
        self._sync_counter += 1
        if self._sync_counter % self.sync_interval == 0:
            # self.check_master_health() # 暂时禁用故障检测，只做页表广播
            if self.is_master:
                self.broadcast_page_table()

    def update_gpu_state(
        self,
        gpu_id: int,
        free_blocks: int,
        block_hashes: Dict[int, int],
    ):
        """
        Master-only state ingestion boundary.

        Each worker owns the real local BlockManager/KV cache for its GPU and
        reports a compact snapshot through the engine message queue. Rank 0 is
        the authoritative GlobalBlockManager master for routing decisions.
        """
        if not self.is_master:
            return

        now = time.time()
        self.free_blocks_per_gpu[gpu_id] = free_blocks

        old_hashes = self.block_hash[gpu_id]
        for block_id, old_hash in list(old_hashes.items()):
            if old_hash in self.global_page_table:
                self.global_page_table[old_hash] = [
                    loc for loc in self.global_page_table[old_hash]
                    if not (loc.gpu_id == gpu_id and loc.block_id == block_id)
                ]
                if not self.global_page_table[old_hash]:
                    del self.global_page_table[old_hash]

        self.block_hash[gpu_id] = dict(block_hashes)
        self.block_access_time[gpu_id] = {
            block_id: now for block_id in block_hashes
        }

        for block_id, block_hash in block_hashes.items():
            if block_hash == -1:
                continue
            self.global_page_table.setdefault(block_hash, []).append(
                BlockLocation(gpu_id, block_id, block_hash, now)
            )

    def reserve_blocks(self, gpu_id: int, num_blocks: int):
        """
        Master-only optimistic reservation for requests routed to a worker.

        The worker later reports the exact local BlockManager state through
        update_gpu_state(), which overwrites this estimate.
        """
        if not self.is_master:
            return
        self.free_blocks_per_gpu[gpu_id] = max(
            0,
            self.free_blocks_per_gpu[gpu_id] - num_blocks,
        )

    # ------------------------------------------------------------------
    # 拓扑辅助函数
    # ------------------------------------------------------------------

    def _get_nvlink_partner(self, gpu_id: int) -> Optional[int]:
        """返回 NVLink 直连对端 GPU ID，无则返回 None"""
        return self.nvlink_partner.get(gpu_id)

    def _get_same_socket_gpus(self, gpu_id: int) -> List[int]:
        """返回同 Socket 内其他 GPU 列表（已在 __init__ 中排好序）"""
        return self.same_socket_gpus.get(gpu_id, [])

    def _get_other_socket_gpus(self, gpu_id: int) -> List[int]:
        """返回跨 Socket 的 GPU 列表"""
        result = []
        for group in self.socket_groups:
            if gpu_id not in group:
                result.extend(group)
        return result

    def _get_target_gpu_order(self, gpu_id: int) -> List[int]:
        """
        返回 swap 目标 GPU 的优先级顺序：
        1. NVLink 直连伙伴（如果有的话）
        2. 同 Socket 其他 GPU（按空闲块数降序）
        3. 跨 Socket GPU（按空闲块数降序）
        """
        ordered = []
        seen = {gpu_id}

        # Tier 1: NVLink 直连伙伴
        partner = self._get_nvlink_partner(gpu_id)
        if partner is not None and partner not in seen:
            ordered.append(partner)
            seen.add(partner)

        # Tier 2: 同 Socket 其他 GPU（按空闲块数降序）
        same_socket = [g for g in self._get_same_socket_gpus(gpu_id) if g not in seen]
        same_socket.sort(key=lambda g: self.free_blocks_per_gpu[g], reverse=True)
        ordered.extend(same_socket)
        seen.update(same_socket)

        # Tier 3: 跨 Socket GPU（按空闲块数降序）
        other_socket = [g for g in self._get_other_socket_gpus(gpu_id) if g not in seen]
        other_socket.sort(key=lambda g: self.free_blocks_per_gpu[g], reverse=True)
        ordered.extend(other_socket)

        return ordered

    # ------------------------------------------------------------------
    # 前缀查找
    # ------------------------------------------------------------------

    def lookup_prefix(self, prefix_hash: int) -> List[BlockLocation]:
        """
        查询拥有指定前缀的块分布在哪些 GPU 上。
        返回按「NVLink 亲和性 + 命中数量」降序排列的 BlockLocation 列表。
        
        排序权重：
        - NVLink 伙伴 GPU 上的块优先（权重 × 2）
        - 同 Socket GPU 次之（权重 × 1.5）
        - 跨 Socket GPU 最后（权重 × 1.0）
        """
        if prefix_hash not in self.global_page_table:
            return []

        locations = self.global_page_table[prefix_hash]

        def sort_key(loc: BlockLocation) -> float:
            score = 1.0
            if self._get_nvlink_partner(loc.gpu_id) == self.rank:
                score = 2.0
            elif self.rank in self._get_same_socket_gpus(loc.gpu_id):
                score = 1.5
            return -score  # 负号使得高权重排在前面

        return sorted(locations, key=sort_key)

    # ------------------------------------------------------------------
    # 空闲块检查
    # ------------------------------------------------------------------

    def can_allocate_global(self, gpu_id: int, num_blocks: int) -> bool:
        """
        检查指定 GPU 是否有足够空闲块。
        如果不够，返回 False，由上层调用 select_eviction_candidates + swap。
        """
        return self.free_blocks_per_gpu[gpu_id] >= num_blocks

    def get_free_blocks_count(self, gpu_id: int) -> int:
        """获取指定 GPU 当前空闲块数量"""
        return self.free_blocks_per_gpu[gpu_id]

    def get_global_free_blocks_count(self) -> int:
        """获取集群总空闲块数量"""
        return sum(self.free_blocks_per_gpu)

    # ------------------------------------------------------------------
    # 全局分配
    # ------------------------------------------------------------------

    def allocate_global(
        self,
        gpu_id: int,
        num_blocks: int,
        block_hashes: List[int],
    ) -> Optional[List[int]]:
        """
        在指定 GPU 上分配 num_blocks 个块，注册到全局页表。
        
        参数:
            gpu_id: 目标 GPU rank
            num_blocks: 需要分配的块数量
            block_hashes: 每个块的 hash 值（长度 == num_blocks）
        
        返回:
            分配的本地 block_id 列表；如果全局都无法满足则返回 None
        """
        if self.free_blocks_per_gpu[gpu_id] >= num_blocks:
            # 本地空闲充足：直接预分配 block_id
            start_id = self.num_blocks_per_gpu - self.free_blocks_per_gpu[gpu_id]
            new_blocks = list(range(start_id, start_id + num_blocks))
            self._commit_alloc(gpu_id, new_blocks, block_hashes)
            return new_blocks
        else:
            # 本地不足：返回 None，由上层协调 swap
            return None

    def _commit_alloc(self, gpu_id: int, block_ids: List[int], hashes: List[int]):
        """提交分配：更新空闲计数、访问时间、全局页表"""
        now = time.time()
        for bid, h in zip(block_ids, hashes):
            self.free_blocks_per_gpu[gpu_id] -= 1
            self.block_access_time[gpu_id][bid] = now
            self.block_hash[gpu_id][bid] = h
            if h not in self.global_page_table:
                self.global_page_table[h] = []
            self.global_page_table[h].append(BlockLocation(gpu_id, bid, h, now))

    # ------------------------------------------------------------------
    # 释放
    # ------------------------------------------------------------------

    def free_global(self, gpu_id: int, block_ids: List[int]):
        """释放指定 GPU 上的块，从全局页表中移除"""
        for bid in block_ids:
            # 增加空闲计数
            self.free_blocks_per_gpu[gpu_id] += 1
            # 从访问时间中移除
            self.block_access_time[gpu_id].pop(bid, None)
            # 从全局页表中移除
            h = self.block_hash[gpu_id].pop(bid, None)
            if h is not None and h in self.global_page_table:
                self.global_page_table[h] = [
                    loc for loc in self.global_page_table[h]
                    if not (loc.gpu_id == gpu_id and loc.block_id == bid)
                ]
                if not self.global_page_table[h]:
                    del self.global_page_table[h]

    # ------------------------------------------------------------------
    # 冷块驱逐选择
    # ------------------------------------------------------------------

    def select_eviction_candidates(
        self,
        gpu_id: int,
        num_blocks: int,
    ) -> List[Tuple[int, int]]:
        """
        从指定 GPU 选择 num_blocks 个 LRU 冷块作为 swap 候选
        
        swap 目标选择策略（三级递进）：
        1. 找一个有至少 num_blocks 空闲块的目标 GPU（按拓扑优先级）
        2. 若所有 GPU 都需要更多空闲块，尝试递归驱逐：从目标 GPU 上也选冷块
        3. 若全局无空闲块：选择远端 LRU 最冷的块直接覆盖（丢弃远端数据）
        
        返回:
            [(local_block_id_to_evict, target_gpu_id), ...]
        """
        # 1. 选出本地最冷的 num_blocks 个块
        if not self.block_access_time[gpu_id]:
            return []

        sorted_blocks = sorted(
            self.block_access_time[gpu_id].items(),
            key=lambda kv: kv[1]  # 按 last_access_time 升序（最老的优先驱逐）
        )
        cold_blocks = [bid for bid, _ in sorted_blocks[:num_blocks]]

        candidates = []
        remaining = list(cold_blocks)

        # 2. 为目标 GPU 排序（拓扑优先级）
        target_order = self._get_target_gpu_order(gpu_id)

        # 3. 对每个冷块找目标
        for block_id in cold_blocks:
            placed = False
            for target in target_order:
                if self.free_blocks_per_gpu[target] > 0:
                    candidates.append((block_id, target))
                    # 临时扣除目标 GPU 的空闲块（用于后续块的计算）
                    self.free_blocks_per_gpu[target] -= 1
                    placed = True
                    break
            if not placed:
                # 所有目标 GPU 都没空闲块：触发递归驱逐或覆盖
                # 选一个远端 LRU 最冷的块来覆盖
                target = target_order[0]  # 拓扑最近的目标
                victim_block = self._select_remote_victim(target)
                if victim_block is not None:
                    # 递归驱逐：先把 target 上的 victim 驱逐到更远的 GPU
                    self.free_blocks_per_gpu[target] += 1  # 释放远端块
                    self.block_access_time[target].pop(victim_block, None)
                    h = self.block_hash[target].pop(victim_block, None)
                    if h is not None and h in self.global_page_table:
                        self.global_page_table[h] = [
                            loc for loc in self.global_page_table[h]
                            if loc.block_id != victim_block
                        ]
                    # 现在 target 有空闲块了，可以接收 swap
                    self.free_blocks_per_gpu[target] -= 1
                else:
                    # 远端也无块可驱逐：直接覆盖，丢弃远端旧数据
                    # target 的块计数不变（覆盖写入）
                    pass
                candidates.append((block_id, target))

        return candidates

    def _select_remote_victim(self, gpu_id: int) -> Optional[int]:
        """在指定 GPU 上选出一个 LRU 最冷的块作为二级驱逐候选"""
        if not self.block_access_time[gpu_id]:
            return None
        return min(self.block_access_time[gpu_id], key=self.block_access_time[gpu_id].get)

    # ------------------------------------------------------------------
    # 块迁移记录
    # ------------------------------------------------------------------

    def record_block_transfer(
        self,
        block_id: int,
        src_gpu: int,
        dst_gpu: int,
        new_block_id: Optional[int] = None,
    ):
        """
        块迁移后更新全局页表。

        参数:
            block_id: 原始 block_id（源 GPU 上）
            src_gpu: 源 GPU rank
            dst_gpu: 目标 GPU rank
            new_block_id: 目标 GPU 上的新 block_id（默认与源相同）
        """
        if new_block_id is None:
            new_block_id = block_id

        now = time.time()

        # 从源 GPU 移除
        self.free_blocks_per_gpu[src_gpu] += 1
        self.block_access_time[src_gpu].pop(block_id, None)
        h = self.block_hash[src_gpu].pop(block_id, None)

        if h is not None and h in self.global_page_table:
            self.global_page_table[h] = [
                loc for loc in self.global_page_table[h]
                if not (loc.gpu_id == src_gpu and loc.block_id == block_id)
            ]
            if not self.global_page_table[h]:
                del self.global_page_table[h]

        # 在目标 GPU 上注册
        self.free_blocks_per_gpu[dst_gpu] -= 1
        self.block_access_time[dst_gpu][new_block_id] = now
        if h is not None:
            self.block_hash[dst_gpu][new_block_id] = h
            if h not in self.global_page_table:
                self.global_page_table[h] = []
            self.global_page_table[h].append(BlockLocation(dst_gpu, new_block_id, h, now))

    # ------------------------------------------------------------------
    # 便利方法
    # ------------------------------------------------------------------

    def get_block_hash(self, gpu_id: int, block_id: int) -> Optional[int]:
        """获取指定块的内容 hash"""
        return self.block_hash[gpu_id].get(block_id)

    def get_block_location(self, hash_val: int) -> List[BlockLocation]:
        """通过 hash 查找块的所有位置"""
        return self.global_page_table.get(hash_val, [])

    def touch_block(self, gpu_id: int, block_id: int):
        """更新块的访问时间（用于 LRU）"""
        self.block_access_time[gpu_id][block_id] = time.time()
