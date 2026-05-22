"""
KV 块跨 GPU 传输模块

提供 swap_out / swap_in 两个核心原语，基于 NCCL send/recv 实现 GPU 间
KV cache 数据的直接搬运

设计要点：
1. 传输粒度：逐块、逐层传输 KV 张量切片
2. 通信协议：交换块索引 -> 逐层 send/recv -> 同步屏障
3. 支持全局内存池枯竭时的覆盖写入（overwrite）
4. 目标 GPU 空闲块分配由 GlobalBlockManager 协调
"""
import logging
import torch
import torch.distributed as dist
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# 工具函数
# ============================================================================

def _allocate_empty_block(kv_cache: torch.Tensor, block_id: int) -> torch.Tensor:
    """
    返回指定 block_id 在 KV cache 中的视图引用（不分配新内存）

    Mini-vLLM 中 KV cache 形状：
    (2, max_cached_blocks, block_size, num_kv_heads, head_dim)
    或每层独立：kv_cache[layer_idx] 形状同上
    具体形状取决于 model_runner 初始化时的分配方式

    参数:
        kv_cache: 单层 KV cache 张量（形状不含 num_layers）
        block_id: 目标块索引
    """
    return kv_cache[:, block_id, ...]


def _compute_tag(block_id: int, layer_idx: int, is_k: bool) -> int:
    """生成 NCCL 通信 tag，避免冲突"""
    # K=0, V=1 -> 编码进 tag
    type_bit = 0 if is_k else 1
    # tag = layer_idx * 2 + type_bit，上限 block_id * 10000 避免冲突
    return block_id * 10000 + layer_idx * 2 + type_bit


# ============================================================================
# 协议消息
# ============================================================================

def _send_block_list(blocks: List[int], dst: int):
    """发送块索引列表（用于协商）"""
    tensor = torch.tensor(blocks, dtype=torch.int64, device=f"cuda:{dist.get_rank()}")
    length = torch.tensor([len(blocks)], dtype=torch.int64, device=tensor.device)
    dist.send(length, dst=dst, tag=99999)
    if len(blocks) > 0:
        dist.send(tensor, dst=dst, tag=99998)


def _recv_block_list(src: int) -> List[int]:
    """接收块索引列表"""
    length = torch.tensor([0], dtype=torch.int64, device=f"cuda:{dist.get_rank()}")
    dist.recv(length, src=src, tag=99999)
    n = length.item()
    if n == 0:
        return []
    tensor = torch.zeros(n, dtype=torch.int64, device=f"cuda:{dist.get_rank()}")
    dist.recv(tensor, src=src, tag=99998)
    return tensor.tolist()


# ============================================================================
# 传输函数
# ============================================================================

def swap_out(
    local_gpu: int,
    blocks_to_evict: List[int],
    target_gpu: int,
    kv_cache: torch.Tensor,
    num_layers: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    target_free_blocks: Optional[List[int]] = None,
) -> List[int]:
    """
    将本地 GPU 上的冷 KV 块搬运到 target_gpu

    参数:
        local_gpu:   源 GPU rank（当前进程）
        blocks_to_evict: 需要换出的本地块索引列表
        target_gpu:  目标 GPU rank
        kv_cache:    单层 KV cache 张量，形状 (2, max_blocks, block_size, num_kv_heads, head_dim)
        num_layers:  模型层数
        block_size:  每块的 token 数
        num_kv_heads: KV head 数
        head_dim:    head 维度
        target_free_blocks: 目标 GPU 上空闲块列表（可选，不传则由对方自行分配）

    协议（每层）:
        源端                            目标端
        ─────                          ─────
        send block_ids + target_ids    recv block_ids + target_ids
        逐块 send K slice             逐块 recv K -> 写入 target_ids
        逐块 send V slice             逐块 recv V -> 写入 target_ids
        barrier                        barrier

    返回:
        target_gpu 上新分配的块索引列表
    """
    logger.info(f"swap_out: GPU{local_gpu} -> GPU{target_gpu} | blocks={blocks_to_evict}")

    rank = dist.get_rank()
    num_blocks = len(blocks_to_evict)
    device = f"cuda:{rank}"

    # 1. 协商目标块索引
    if rank == local_gpu and rank != target_gpu:
        # 我是发送方：告诉接收方“我要换出这些块”
        _send_block_list(blocks_to_evict, dst=target_gpu)
        if target_free_blocks is not None:
            _send_block_list(target_free_blocks, dst=target_gpu)
        # 接收目标 GPU 分配的空闲块 ID
        target_blocks = _recv_block_list(src=target_gpu)
    elif rank == target_gpu and rank != local_gpu:
        # 我是接收方：接收块列表，分配空闲块
        remote_blocks = _recv_block_list(src=local_gpu)
        # 接收发送方指定的空闲块（如果有，用于覆盖模式）
        specified_blocks = _recv_block_list(src=local_gpu)
        if specified_blocks:
            target_blocks = specified_blocks
        else:
            target_blocks = []  # 由外部预先分配好
        _send_block_list(target_blocks, dst=local_gpu)
    else:
        # 本地传输或单 GPU 场景
        target_blocks = blocks_to_evict  # 同 GPU 内 swap 直接复用

    # 2. 逐层、逐块传输 KV 数据
    for layer_idx in range(num_layers):
        layer_kv = kv_cache[layer_idx] if kv_cache.dim() == 5 else kv_cache
        for i, (src_block, dst_block) in enumerate(zip(blocks_to_evict, target_blocks)):
            if rank == local_gpu and rank != target_gpu:
                # 读取本地 K、V 切片
                k_slice = layer_kv[0, src_block, ...].contiguous()
                v_slice = layer_kv[1, src_block, ...].contiguous()
                dist.send(k_slice, dst=target_gpu, tag=_compute_tag(src_block, layer_idx, is_k=True))
                dist.send(v_slice, dst=target_gpu, tag=_compute_tag(src_block, layer_idx, is_k=False))
            elif rank == target_gpu and rank != local_gpu:
                # 接收并写入目标块
                k_buf = torch.zeros(block_size, num_kv_heads, head_dim,
                                    dtype=layer_kv.dtype, device=device)
                v_buf = torch.zeros(block_size, num_kv_heads, head_dim,
                                    dtype=layer_kv.dtype, device=device)
                dist.recv(k_buf, src=local_gpu, tag=_compute_tag(src_block, layer_idx, is_k=True))
                dist.recv(v_buf, src=local_gpu, tag=_compute_tag(src_block, layer_idx, is_k=False))
                layer_kv[0, dst_block, ...].copy_(k_buf)
                layer_kv[1, dst_block, ...].copy_(v_buf)

    # 3. 全层同步
    barrier_tensor = torch.zeros(1, device=device)
    dist.all_reduce(barrier_tensor)

    return target_blocks


def swap_in(
    remote_gpu: int,
    remote_blocks: List[int],
    local_gpu: int,
    kv_cache: torch.Tensor,
    num_layers: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    local_target_blocks: Optional[List[int]] = None,
) -> List[int]:
    """
    从 remote_gpu 拉取 KV 块到本地。

    参数同 swap_out，方向相反。

    协议:
        本地端                              远端
        ─────                              ─────
        send remote_blocks                 recv remote_blocks
        send local_target_blocks (可选)    recv local_target_blocks
        recv local_target_blocks           send local_target_blocks
        逐块 recv K、V                     逐块 send K、V
        barrier                            barrier

    返回:
        本地新分配的块索引列表
    """
    logger.info(f"swap_in: GPU{remote_gpu} -> GPU{local_gpu} | blocks={remote_blocks}")

    rank = dist.get_rank()
    device = f"cuda:{rank}"

    # 1. 协商块索引
    if rank == local_gpu and rank != remote_gpu:
        _send_block_list(remote_blocks, dst=remote_gpu)
        if local_target_blocks is not None:
            _send_block_list(local_target_blocks, dst=remote_gpu)
        local_blocks = _recv_block_list(src=remote_gpu)
    elif rank == remote_gpu and rank != local_gpu:
        remote_blocks_recv = _recv_block_list(src=local_gpu)
        specified = _recv_block_list(src=local_gpu)
        if specified:
            local_blocks = specified
        else:
            local_blocks = remote_blocks_recv
        _send_block_list(local_blocks, dst=local_gpu)
    else:
        local_blocks = remote_blocks

    # 2. 逐层、逐块传输 KV 数据
    for layer_idx in range(num_layers):
        layer_kv = kv_cache[layer_idx] if kv_cache.dim() == 5 else kv_cache
        for src_block, dst_block in zip(remote_blocks, local_blocks):
            if rank == local_gpu and rank != remote_gpu:
                k_buf = torch.zeros(block_size, num_kv_heads, head_dim,
                                    dtype=layer_kv.dtype, device=device)
                v_buf = torch.zeros(block_size, num_kv_heads, head_dim,
                                    dtype=layer_kv.dtype, device=device)
                dist.recv(k_buf, src=remote_gpu, tag=_compute_tag(src_block, layer_idx, is_k=True))
                dist.recv(v_buf, src=remote_gpu, tag=_compute_tag(src_block, layer_idx, is_k=False))
                layer_kv[0, dst_block, ...].copy_(k_buf)
                layer_kv[1, dst_block, ...].copy_(v_buf)
            elif rank == remote_gpu and rank != local_gpu:
                k_slice = layer_kv[0, src_block, ...].contiguous()
                v_slice = layer_kv[1, src_block, ...].contiguous()
                dist.send(k_slice, dst=local_gpu, tag=_compute_tag(src_block, layer_idx, is_k=True))
                dist.send(v_slice, dst=local_gpu, tag=_compute_tag(src_block, layer_idx, is_k=False))

    # 3. 全层同步
    barrier_tensor = torch.zeros(1, device=device)
    dist.all_reduce(barrier_tensor)

    return local_blocks


# ============================================================================
# 块分配协商
# ============================================================================

def request_free_blocks(target_gpu: int, num_blocks: int) -> List[int]:
    """
    向目标 GPU 请求分配 num_blocks 个空闲块。
    返回目标 GPU 上空闲块的物理索引列表。

    注意：这只负责传输请求，实际分配需要目标 GPU 上运行的
    GlobalBlockManager.commit_alloc 来执行。
    """
    rank = dist.get_rank()
    device = f"cuda:{rank}"

    request = torch.tensor([num_blocks], dtype=torch.int64, device=device)
    if rank != target_gpu:
        dist.send(request, dst=target_gpu, tag=88888)
        # 接收分配的块列表
        n_tensor = torch.tensor([0], dtype=torch.int64, device=device)
        dist.recv(n_tensor, src=target_gpu, tag=88887)
        n = n_tensor.item()
        if n == 0:
            return []
        blocks = torch.zeros(n, dtype=torch.int64, device=device)
        dist.recv(blocks, src=target_gpu, tag=88886)
        return blocks.tolist()
    else:
        # 目标 GPU 端逻辑由 GlobalBlockManager 触发，此处不应直接运行
        raise RuntimeError("request_free_blocks should not be called on target GPU directly")


def respond_free_blocks(requester_gpu: int, allocated_blocks: List[int]):
    """
    响应空闲块请求，把分配好的块列表发回给请求方。
    在目标 GPU 上由 GlobalBlockManager 调用。
    """
    rank = dist.get_rank()
    device = f"cuda:{rank}"

    if rank != requester_gpu:
        n = len(allocated_blocks)
        n_tensor = torch.tensor([n], dtype=torch.int64, device=device)
        dist.send(n_tensor, dst=requester_gpu, tag=88887)
        if n > 0:
            blocks_tensor = torch.tensor(allocated_blocks, dtype=torch.int64, device=device)
            dist.send(blocks_tensor, dst=requester_gpu, tag=88886)