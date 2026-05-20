"""
全局调度器 (Global Scheduler)

负责跨 GPU 的两类决策：
1. 请求路由：新序列应该在哪个 GPU 上执行（前缀复用 + 负载均衡）
2. 显存重平衡：本地空闲块不足时，编排跨 GPU 的 swap 操作

设计要点：
1. 依赖 GlobalBlockManager 获取全局页表和空闲块分布
2. 依赖本地 BlockManager 做前缀 hash 计算
3. swap 编排需要和目标 GPU 上的 GlobalBlockManager 协同
"""

import torch
import torch.distributed as dist
from typing import List, Tuple, Optional
from myvllm.engine.sequence import Sequence


class GlobalScheduler:
    """
    全局调度器

    职责：
    - route_sequence:   决定新序列的归属 GPU
    - rebalance:        编排 swap，为本 GPU 腾出空闲块
    - preempt_for_swap: 当 rebalance 失败时，选择序列回退
    """

    def __init__(self, gbm, block_manager):
        """
        参数:
            gbm: GlobalBlockManager 实例（维护全局页表）
            block_manager: 本地 BlockManager 实例（提供 compute_hash 等接口）
        """
        self.gbm = gbm
        self.block_manager = block_manager
        
        # test
        # self.cnt=0

    # ------------------------------------------------------------------
    # 请求路由
    # ------------------------------------------------------------------

    def route_sequence(self, seq: Sequence) -> int:
        """
        决定 seq 应该在哪个 GPU 上执行

        策略（按优先级）:
        1. 计算 seq 的前缀 hash
        2. 查询 gbm.lookup_prefix 获取前缀命中的 GPU 列表
        3. 选择前缀命中数 × 拓扑权重最高的 GPU
        4. 若无命中，选择当前空闲块最多的 GPU
        5. 如果命中 GPU 空闲块不足，选择最适合做 swap 的目标 GPU

        返回:
            target_gpu_id: 推荐的执行 GPU rank
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # test
        # self.cnt+=1
        # return self.cnt%2
        print(f"[GlobalScheduler] Routing seq {seq.seq_id} (tokens={seq.num_tokens}, blocks={seq.num_blocks})")

        # 1. 计算前缀 hash
        prefix_hash = self._compute_prefix_hash(seq)

        if prefix_hash is None:
            # 没有完整的块前缀，选择空闲最多的 GPU
            target = self._select_most_free_gpu(rank, world_size)
            print(f"[GlobalScheduler] seq {seq.seq_id}: no full blocks -> GPU {target} (most free)")
            return target

        # 2. 查询全局前缀命中
        hits = self.gbm.lookup_prefix(prefix_hash)

        if not hits:
            # 没有命中任何 GPU，选择空闲最多的 GPU
            target = self._select_most_free_gpu(rank, world_size)
            print(f"[GlobalScheduler] seq {seq.seq_id}: prefix hash={prefix_hash}, no hits -> GPU {target} (most free)")
            return target

        # 3. 按 GPU 聚合命中块数
        gpu_hit_count: dict[int, int] = {}
        for loc in hits:
            gpu_hit_count[loc.gpu_id] = gpu_hit_count.get(loc.gpu_id, 0) + 1

        # 4. 加权打分
        # score = 命中块数 × 拓扑权重
        # 拓扑权重：同 GPU=3.0, NVLink 伙伴=2.0, 同 Socket=1.5, 跨 Socket=1.0
        best_gpu = rank  # 默认本地
        best_score = -1.0
        failed_gpus = []  # 记录空闲不足的命中 GPU

        for gpu_id, hit_count in gpu_hit_count.items():
            topo_weight = self._get_topo_weight(rank, gpu_id)
            score = hit_count * topo_weight

            # 检查空闲块是否足够（需要 seq.num_blocks 个块）
            needed = seq.num_blocks
            if self.gbm.get_free_blocks_count(gpu_id) >= needed:
                if score > best_score:
                    best_score = score
                    best_gpu = gpu_id
            else:
                # 空闲不足，暂存作为备选
                failed_gpus.append((gpu_id, score, hit_count))

        if best_score >= 0:
            print(f"[GlobalScheduler] seq {seq.seq_id}: prefix hash={prefix_hash}, "
              f"hits={gpu_hit_count}, best=GPU {best_gpu} (score={best_score:.1f}, "
              f"free={self.gbm.get_free_blocks_count(best_gpu)}/{seq.num_blocks})")
            return best_gpu

        # 5. 命中 GPU 空闲都不够 -> 选择权重最高的，后续 rebalance 会腾空间
        if failed_gpus:
            failed_gpus.sort(key=lambda x: x[1], reverse=True)
            target = failed_gpus[0][0]
            print(f"[GlobalScheduler] seq {seq.seq_id}: prefix hash={prefix_hash}, "
                    f"all hit GPUs full, fallback=GPU {target} "
                    f"(failed={[(g, s) for g, s, _ in failed_gpus]})")
            return target

        # 6. 兜底：本地或空闲最多的 GPU
        target = self._select_most_free_gpu(rank, world_size)
        print(f"[GlobalScheduler] seq {seq.seq_id}: fallback -> GPU {target} (most free)")
        return target

    def _compute_prefix_hash(self, seq: Sequence) -> Optional[int]:
        """
        计算序列的前缀 hash
        使用 BlockManager 的 compute_hash 方法，只 hash 完整的块（不含 partial 尾块）。
        
        返回:
            hash 值，如果序列没有完整块则返回 None
        """
        full_blocks = int(seq.num_tokens // seq.block_size)
        if full_blocks == 0:
            return None

        # 只取完整块的部分做 hash
        hash_val = -1
        for i in range(full_blocks):
            block_tokens = seq.token_ids[i * seq.block_size : (i + 1) * seq.block_size]
            hash_val = self.block_manager.compute_hash(block_tokens, hash_val)
        return hash_val

    def _get_topo_weight(self, my_rank: int, target_gpu: int) -> float:
        """
        计算拓扑权重
        - 同 GPU: 3.0
        - NVLink 直连: 2.0
        - 同 Socket PCIe: 1.5
        - 跨 Socket: 1.0
        """
        if target_gpu == my_rank:
            return 3.0
        partner = self.gbm._get_nvlink_partner(my_rank)
        if partner is not None and partner == target_gpu:
            return 2.0
        same_socket = self.gbm._get_same_socket_gpus(my_rank)
        if target_gpu in same_socket:
            return 1.5
        return 1.0

    def _select_most_free_gpu(self, my_rank: int, world_size: int) -> int:
        """选择全局空闲块最多的 GPU，同等条件优先本地"""
        best_gpu = my_rank
        best_free = self.gbm.get_free_blocks_count(my_rank)
        for gpu_id in range(world_size):
            free = self.gbm.get_free_blocks_count(gpu_id)
            if free > best_free or (free == best_free and gpu_id == my_rank):
                best_free = free
                best_gpu = gpu_id
        return best_gpu

    # ------------------------------------------------------------------
    # 显存重平衡
    # ------------------------------------------------------------------

    def rebalance(self, gpu_id: int, needed_blocks: int) -> bool:
        """
        当 gpu_id 需要 needed_blocks 个空闲块但本地不足时调用

        流程:
        1. 调用 gbm.select_eviction_candidates 获取换出方案
        2. 逐对执行 swap_out
        3. 更新受影响的序列 block_table
        4. 通知目标 GPU 的 GlobalBlockManager 更新页表

        返回:
            是否成功腾出至少 needed_blocks 个空闲块
        """
        rank = dist.get_rank()

        # 1. 获取驱逐候选
        candidates = self.gbm.select_eviction_candidates(gpu_id, needed_blocks)

        if not candidates:
            return False

        # 2. 检查是否足够
        # 每个 candidate 释放 gpu_id 上的一个块，所以 candidates 长度应 >= needed_blocks
        if len(candidates) < needed_blocks:
            return False

        actual_candidates = candidates[:needed_blocks]

        # 3. 执行 swap_out
        # 按 target_gpu 分组，一次 NCCL 操作处理同一目标的批量块
        groups: dict[int, List[int]] = {}
        for local_block, target_gpu in actual_candidates:
            if target_gpu not in groups:
                groups[target_gpu] = []
            groups[target_gpu].append(local_block)

        for target_gpu, blocks in groups.items():
            if rank == gpu_id and target_gpu != rank:
                self._execute_swap_out(blocks, gpu_id, target_gpu)
            elif rank == target_gpu and gpu_id != rank:
                self._execute_swap_in_accept(blocks, gpu_id, target_gpu)

        # 4. 更新本地空闲块计数和页表
        for local_block, target_gpu in actual_candidates:
            # 释放本地块
            self.gbm.free_global(gpu_id, [local_block])
            # 目标 GPU 上减去一个空闲块（由 record_block_transfer 处理）
            # 这里由上层 GlobalBlockManager.record_block_transfer 统一更新

        return True

    def _execute_swap_out(
        self,
        blocks: List[int],
        local_gpu: int,
        target_gpu: int,
    ):
        """
        在源 GPU 上执行 swap_out
        直接调用 kv_transfer 的 send 逻辑
        """
        from myvllm.engine.kv_transfer import _send_block_list, _compute_tag
        import time

        device = f"cuda:{local_gpu}"
        # 这里需要 kv_cache 的引用，由外部 ModelRunner 提供
        # 暂时留空，由实际调用方注入
        raise NotImplementedError(
            "swap_out 需要 kv_cache 张量引用，请在 ModelRunner 中调用 "
            "kv_transfer.swap_out() 完成实际数据传输"
        )

    def _execute_swap_in_accept(
        self,
        blocks: List[int],
        source_gpu: int,
        local_gpu: int,
    ):
        """
        在目标 GPU 上接收 swap_out 的数据。
        """
        from myvllm.engine.kv_transfer import _recv_block_list
        raise NotImplementedError(
            "swap_in_accept 需要 kv_cache 张量引用，请在 ModelRunner 中调用 "
            "kv_transfer 完成实际数据传输"
        )

    # ------------------------------------------------------------------
    # 抢占回退
    # ------------------------------------------------------------------

    def preempt_for_rebalance(
        self,
        running_sequences: list,
        gpu_id: int,
        needed_blocks: int,
    ) -> bool:
        """
        当 swap 无法满足需求时，选择序列回退到 WAITING 状态
        
        策略：
        选择最短的 running 序列，释放其所有块，直到满足 needed_blocks。
        
        返回:
            是否成功腾出足够空间
        """
        freed = 0
        victims = []

        for seq in running_sequences:
            if freed >= needed_blocks:
                break
            victims.append(seq)
            freed += len(seq.block_table)

        if freed < needed_blocks:
            return False

        for seq in victims:
            self.gbm.free_global(gpu_id, seq.block_table)
            seq.block_table = []
            seq.status = 2  # WAITING
            seq.num_cached_tokens = 0

        return True