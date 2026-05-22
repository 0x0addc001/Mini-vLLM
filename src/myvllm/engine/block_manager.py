import xxhash
import numpy as np
from collections import deque
from typing import Tuple, Optional
import torch
import torch.distributed as dist

from myvllm.engine.sequence import Sequence


class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.hash = -1 
        self.ref_count = 0
        self.token_ids = []


    def update(self, h: int, token_ids: list[int]):
        self.hash = h 
        self.token_ids = token_ids

    def reset(self):
        self.hash = -1 
        self.ref_count = 0
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, gbm=None):
        """
        参数:
            num_blocks: 每 GPU 物理块总数
            block_size: 每块容纳的 token 数
            gbm: GlobalBlockManager 实例（启用全局池化时用）
        """
        # block_size: number of tokens per block
        self.block_size: int = block_size
        # list of all blocks
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # hash to block id: this is for prefix caching (local)
        self.hash_to_block_id: dict[int, int] = {}
        # free block ids
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # used block ids
        self.used_block_ids: set[int] = set()

        # 全局块管理器引用（None 则为单卡模式）
        self.gbm = gbm

    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        h = xxhash.xxh64()
        if prefix_hash_value != -1:
            h.update(prefix_hash_value.to_bytes(8, 'little'))
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        elif isinstance(token_ids, (list, tuple)):
            token_ids = [t.item() if isinstance(t, torch.Tensor) else t for t in token_ids]
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    # move this block to used list
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0, "Block is still in use"
        block = self.blocks[block_id]
        # 从本地 hash 映射中移除
        if block.hash != -1 and block.hash in self.hash_to_block_id:
            if self.hash_to_block_id[block.hash] == block_id:
                del self.hash_to_block_id[block.hash]
        block.token_ids = []
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # whether we can allocate a block for this sequence
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks


    def allocate(self, seq: Sequence) -> None:
        h = -1
        for i in range(seq.num_blocks):
            no_cache_found = False

            token_ids = seq.block(i)
            # only compute hash for full blocks, always -1 for partial blocks
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            
            # if cache miss or hash collision
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                no_cache_found = True

            if not no_cache_found:
                # update sequence information
                seq.num_cached_tokens += self.block_size  # which == len(token_ids)
                # update block information, considering the edge case that the block is not allocated yet but with hash code
                if block_id not in self.used_block_ids:
                    block = self._allocate_block(block_id)
                else:
                    # update block information
                    block = self.blocks[self.hash_to_block_id[h]]
                    block.ref_count += 1
            else:
                # cache miss
                block = self._allocate_block(self.free_block_ids[0])
                block.update(h=h, token_ids=token_ids)
                if h != -1:
                    self.hash_to_block_id[h] = block.block_id
            seq.block_table.append(block.block_id)

        # 注册到全局页表
        if self.gbm is not None and h != -1:
            rank = dist.get_rank() if dist.is_initialized() else 0
            self.gbm._commit_alloc(rank, [block.block_id], [h])

    def deallocate(self, seq: Sequence) -> None:
        # update block information
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # update sequence information
        seq.block_table = []
        seq.num_cached_tokens = 0

    # this is to check whether we can append tokens to this sequence
    # when that token would require allocating a new block.
    def can_append(self, seq: Sequence) -> bool:
        if seq.num_tokens % self.block_size == 0:
            return len(self.free_block_ids) > 0
        return True

    # this is the actual work to append tokens to this sequence
    # this is called when the new token has been added to the seq information
    # but no block in gpu has yet allocate for it
    def append(self, seq: Sequence) -> None:
        block_tables = seq.block_table
        last_block_for_seq_id = block_tables[-1]

        # if the last block is now full, compute hash
        if seq.num_tokens % self.block_size == 0:
            token_ids=seq.block(seq.num_blocks - 1)

            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()  # 转成 Python list

            h = self.compute_hash(
                token_ids=token_ids,
                prefix_hash_value=-1 if len(block_tables) == 1 else self.blocks[block_tables[-2]].hash
            )
            block = self.blocks[last_block_for_seq_id]
            block.update(h=h, token_ids=seq.block(seq.num_blocks - 1))
            self.hash_to_block_id[h] = block.block_id

            # 注册到全局页表
            if self.gbm is not None and h != -1:
                rank = dist.get_rank() if dist.is_initialized() else 0
                self.gbm._commit_alloc(rank, [block.block_id], [h])

        # if one new block is needed
        elif seq.num_tokens % self.block_size == 1:
            # Previous block should be finalized
            assert self.blocks[last_block_for_seq_id].hash != -1
            block = self._allocate_block(self.free_block_ids[0])
            block_tables.append(block.block_id)
        # else, do nothing
        else:
            assert last_block_for_seq_id in self.used_block_ids, "Last block should be allocated"
            assert self.blocks[last_block_for_seq_id].hash == -1, "Last block should be partial block with hash -1"

    # ------------------------------------------------------------------
    # 全局 KV cache 池相关方法
    # ------------------------------------------------------------------

    def try_allocate_remote(self, seq: Sequence) -> Tuple[bool, int]:
        """
        检查是否可以通过远程前缀复用来分配块

        流程:
        1. 计算序列前缀 hash
        2. 调用 gbm.lookup_prefix 查找远程命中
        3. 若命中，标记 seq 的远程前缀信息，后续由 ModelRunner 做 swap_in

        返回:
            (是否有远程前缀命中, 命中块所在的 GPU rank)
        """
        if self.gbm is None:
            return False, -1

        # 计算前缀 hash（只取完整块）
        full_blocks = int(seq.num_tokens // self.block_size)
        if full_blocks == 0:
            return False, -1

        prefix_hash = -1
        for i in range(full_blocks):
            token_ids = seq.block(i)
            prefix_hash = self.compute_hash(token_ids, prefix_hash)

        # 查询全局页表
        hits = self.gbm.lookup_prefix(prefix_hash)
        if not hits:
            return False, -1

        # 按 GPU 聚合命中块数
        gpu_hit_count: dict[int, list[int]] = {}
        for loc in hits:
            if loc.gpu_id not in gpu_hit_count:
                gpu_hit_count[loc.gpu_id] = []
            gpu_hit_count[loc.gpu_id].append(loc.block_id)

        if not gpu_hit_count:
            return False, -1

        # 选择命中块数最多的 GPU（优先本地的 NVLink 伙伴）
        rank = dist.get_rank()
        best_gpu = -1
        best_count = 0

        for gpu_id, blocks in gpu_hit_count.items():
            count = len(blocks)
            # 拓扑加权
            if gpu_id == rank:
                count = count * 3
            elif self.gbm._get_nvlink_partner(rank) == gpu_id:
                count = count * 2
            elif gpu_id in self.gbm._get_same_socket_gpus(rank):
                count = count * 1.5

            if count > best_count:
                best_count = count
                best_gpu = gpu_id

        if best_gpu <= 0:
            return False, -1

        # 标记序列的远程前缀信息
        seq.is_remote_prefix = True
        seq.remote_gpu_id = best_gpu
        # 记录需要 swap_in 的远端块（按 block table 顺序）
        seq.pending_swap_in = gpu_hit_count[best_gpu]

        return True, best_gpu

    def allocate_with_swap(self, seq: Sequence) -> bool:
        """
        当本地空闲不足时的分配入口

        流程:
        1. 检查本地空闲块是否够
        2. 不够则调用 gbm.select_eviction_candidates 获取换出候选
        3. 标记需要 swap_out 的块（实际传输由 ModelRunner 执行）
        4. 本地分配新块

        返回:
            是否成功分配（本地空闲现在够用）
        """
        if self.can_allocate(seq):
            self.allocate(seq)
            return True

        if self.gbm is None:
            return False

        needed = seq.num_blocks
        available = len(self.free_block_ids)
        shortage = needed - available

        # 获取驱逐候选
        rank = dist.get_rank()
        candidates = self.gbm.select_eviction_candidates(rank, shortage)

        if not candidates or len(candidates) < shortage:
            return False

        # 释放被选中的本地冷块（数据已由 swap_out 搬走）
        for local_block, target_gpu in candidates:
            # 从 used 移到 free（但不立即复用，等分配时再取）
            block = self.blocks[local_block]
            if block.ref_count == 0:
                self._deallocate_block(local_block)
            # 更新全局页表
            self.gbm.record_block_transfer(
                block_id=local_block,
                src_gpu=rank,
                dst_gpu=target_gpu,
            )

        # 现在应该有足够的空闲块了
        if not self.can_allocate(seq):
            return False

        self.allocate(seq)
        return True

    def append_with_swap(self, seq: Sequence) -> bool:
        """
        解码追加时空间不足的 swap 版本

        流程同 allocate_with_swap，但只分配 1 个块。

        返回:
            是否成功
        """
        if self.can_append(seq):
            self.append(seq)
            return True

        if self.gbm is None:
            return False

        # 只需要 1 个块
        rank = dist.get_rank()
        candidates = self.gbm.select_eviction_candidates(rank, 1)

        if not candidates:
            return False

        local_block, target_gpu = candidates[0]

        # 释放冷块
        block = self.blocks[local_block]
        if block.ref_count == 0:
            self._deallocate_block(local_block)

        # 更新全局页表
        self.gbm.record_block_transfer(
            block_id=local_block,
            src_gpu=rank,
            dst_gpu=target_gpu,
        )

        # 现在追加
        if not self.can_append(seq):
            return False

        self.append(seq)
        return True

    def get_local_block_hashes(self) -> dict[int, int]:
        """
        获取本地所有已用块的 (block_id -> hash) 映射
        用于向 GlobalBlockManager 上报本地状态
        """
        return {bid: self.blocks[bid].hash for bid in self.used_block_ids}

    def sync_with_global(self):
        """
        将本地块状态推送到 GlobalBlockManager
        在管理节点上调用
        """
        if self.gbm is None or not self.gbm.is_master:
            return

        rank = dist.get_rank()
        for bid in self.used_block_ids:
            block = self.blocks[bid]
            if block.hash != -1:
                self.gbm.block_hash[rank][bid] = block.hash
                self.gbm.block_access_time[rank][bid] = __import__('time').time()