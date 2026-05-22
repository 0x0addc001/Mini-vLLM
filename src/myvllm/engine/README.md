# LMPool: Distributed KV Cache Pooling for LLM Inference

LMPool 将集群内多张 GPU 的 HBM 抽象为一个逻辑统一的全局 KV Cache 池，在 vLLM 的 Paged Attention 基础上扩展出跨 GPU 的块级前缀感知路由和冷热感知驱逐。

> **注意**：本项目基于 [Mini-vLLM](https://github.com/Wenyueh/MinivLLM) 构建，目前处于原型验证阶段

---

## 目录

1. [核心设计](#1-核心设计)
2. [架构总览](#2-架构总览)
3. [路由决策：Router](#3-路由决策router)
4. [Swap 编排：Swapper](#4-swap-编排swapper)
5. [全局块管理：GlobalBlockManager](#5-全局块管理globalblockmanager)
6. [KV 传输：Transfer](#6-kv-传输transfer)
7. [调度器扩展：Scheduler](#7-调度器扩展scheduler)
8. [模型执行器：ModelRunner](#8-模型执行器modelrunner)
9. [序列管理：Sequence](#9-序列管理sequence)
10. [配置与运行](#10-配置与运行)
11. [当前状态与下一步](#11-当前状态与下一步)

---

## 1. 核心设计

### 1.1 问题

vLLM 原始的 Paged Attention 中，每张 GPU 独立管理自己的显存，存在三个局限：

| 局限 | 现象 | 后果 |
|------|------|------|
| **无法跨卡复用前缀** | 多个请求共享相同前缀，但各卡各自存一份 | 显存浪费，有效吞吐下降 |
| **OOM 时无弹性** | 本地 HBM 耗尽 → OOM 或触发 CPU swap | 延迟飙升或请求中断 |
| **冷热分布失衡** | 本地 HBM 逐渐充满冷块，热块被挤到 CPU | 延迟持续上升 |

### 1.2 方案

将多 GPU 的 HBM 抽象为统一的分布式显存池：
- **逻辑统一**：`GlobalBlockManager` 维护跨 GPU 的全局页表，记录每个 KV 块的物理位置
- **前缀复用**：块级 hash 链编码前缀，跨 GPU 查重，相同前缀只存一份
- **冷热感知**：LRU 驱逐 + 拓扑优先的 swap（NVLink > PIX > NODE）
- **控制面/数据面分离**：`GlobalScheduler` 做决策，`kv_transfer` 做 NCCL 搬运

---

## 2. 架构总览

```
┌──────────────────────────────────────────────────┐
│                  Control Plane                   │
│  ┌──────────────────┐  ┌───────────────────────┐  │
│  │ GlobalScheduler  │  │ GlobalBlockManager    │  │
│  │ - route_sequence │  │ - global_page_table   │  │
│  │ - rebalance      │  │ - free_blocks_per_gpu │  │
│  └──────┬───────────┘  │ - block_access_time   │  │
│         │               └───────────┬───────────┘  │
│         │   调度/查询                │ 页表/空闲块   │
└─────────┼───────────────────────────┼──────────────┘
          │                           │
┌─────────┼───────────────────────────┼──────────────┐
│         ▼         Data Plane        ▼              │
│  ┌──────────┐    NVLink/NCCL   ┌──────────┐        │
│  │  GPU 0   │ ◄──────────────► │  GPU 1   │        │
│  │  - Block │   swap_out/in    │  - Block │        │
│  │  Manager │                  │  Manager │        │
│  │  - Model │                  │  - Model │        │
│  │  Runner  │                  │  Runner  │        │
│  └──────────┘                  └──────────┘        │
└────────────────────────────────────────────────────┘
```

---

## 3. 路由决策：Router

### 3.1 文件位置

`src/myvllm/engine/global_scheduler.py` → `GlobalScheduler.route_sequence()`

### 3.2 决策流程

```
新序列到达
  │
  ├─ 1. 计算前缀 hash（链式，只算完整块）
  │      full_blocks = num_tokens // block_size
  │      hash_0 = xxhash(tokens[0:256])
  │      hash_1 = xxhash(hash_0 + tokens[256:512])
  │      ...
  │
  ├─ 2. 查全局页表
  │      hits = gbm.lookup_prefix(prefix_hash)
  │      返回 List[BlockLocation] — 每个 BlockLocation 有 (gpu_id, block_id, hash, last_access_time)
  │
  ├─ 3. 按 GPU 聚合命中块数
  │      gpu_hit_count[gpu_id] = 命中块数
  │
  ├─ 4. 加权打分
  │      score = hit_count × topo_weight
  │      拓扑权重：同 GPU 3.0 | NVLink 伙伴 2.0 | 同 Socket 1.5 | 跨 Socket 1.0
  │
  ├─ 5. 选择最高分 GPU（空闲块数需足够）
  │      若命中 GPU 空闲不足 → 选权重最高的，后续 rebalance 腾空间
  │
  └─ 6. 无命中 → 选空闲最多的 GPU（同等条件优先本地）
```

### 3.3 关键辅助方法

| 方法 | 作用 |
|------|------|
| `_compute_prefix_hash(seq)` | 链式 hash，只 hash 完整的 256-token 块 |
| `_get_topo_weight(my_rank, target_gpu)` | 根据 NVLink 拓扑给 GPU 对打分 |
| `_select_most_free_gpu(my_rank, world_size)` | 选空闲块最多的 GPU，打平时优先本地 |
| `_hit_summary(hits)` | 将 `List[BlockLocation]` 聚合为 `{gpu_id: [block_ids]}` |

### 3.4 当前局限

- 路由到远程 GPU 后，目标 GPU 上的 `allocate` 会重新分配块，没有复用已有前缀 KV 缓存
- 修复方向：在 `route_sequence` 返回后，把命中块的 `BlockLocation` 信息传入 `seq.pending_swap_in`，目标 GPU 的 `allocate` 复用这些块

---

## 4. Swap 编排：Swapper

### 4.1 文件位置

`src/myvllm/engine/global_scheduler.py` → `GlobalScheduler.rebalance()`

### 4.2 决策流程

```
本地空闲块不足
  │
  ├─ 1. 调用 gbm.select_eviction_candidates(gpu_id, needed_blocks)
  │      返回 [(local_block_id, target_gpu_id), ...]
  │
  ├─ 2. 按 target_gpu 分组
  │
  ├─ 3. 通知目标 GPU 准备接收
  │      通过 send_queue 发 {"type": "swap_in", ...}
  │
  ├─ 4. 执行 swap_out（NCCL send）
  │      源端：execute_swap_out → barrier → send
  │      目标端：execute_swap_in → barrier → recv
  │
  └─ 5. 释放本地块，更新全局页表
```

### 4.3 关键辅助方法

| 方法 | 作用 |
|------|------|
| `_execute_swap_out(blocks, local_gpu, target_gpu)` | 通知目标 GPU，调 ModelRunner.execute_swap_out |
| `_execute_swap_in_accept(blocks, source_gpu, local_gpu)` | 调 ModelRunner.execute_swap_in 接收数据 |
| `preempt_for_rebalance(running, gpu_id, needed)` | swap 失败时的回退：释放最短序列的所有块 |

### 4.4 当前局限

- swap_in 端到端还没跑通（待验证）
- 换出块在目标 GPU 上的空闲块分配逻辑未实现
- 冷块选择只看 LRU，没有引用计数（多序列共享的块不该被换出）

---

## 5. 全局块管理：GlobalBlockManager

### 5.1 文件位置

`src/myvllm/engine/global_block_manager.py`

### 5.2 核心数据结构

```python
global_page_table: Dict[int, List[BlockLocation]]
# prefix_hash → 该 hash 所在的物理位置列表
# 一个 hash 可能对应多个 GPU 上的多个副本

free_blocks_per_gpu: List[int]
# 每 GPU 的空闲块计数

block_access_time: List[Dict[int, float]]
# 每 GPU 上每块的最近访问时间（LRU 用）

block_hash: List[Dict[int, int]]
# 每 GPU 上 block_id → hash 的映射
```

### 5.3 前缀查找：lookup_prefix

```
lookup_prefix(prefix_hash)
  │
  └─ 查 global_page_table[prefix_hash]
       │
       └─ 按拓扑亲和性排序：
            NVLink 伙伴上的块权重 × 2.0
            同 Socket 上的块权重 × 1.5
            跨 Socket 上的块权重 × 1.0
       │
       └─ 返回排序后的 List[BlockLocation]
```

**哈希链的块级前缀匹配**：`block_i` 的 hash 依赖于 `block_{i-1}` 的 hash，所以 `block_i` 的 hash 实际上编码了从 `block_0` 到 `block_i` 的全部内容。两个序列如果有相同的前缀 `[block_0, ..., block_k]`，它们的 `block_k` hash 必然相同。查 `global_page_table[hash_of_block_k]` 就能找到所有拥有这个完整前缀的 GPU。

### 5.4 拓扑感知分配：select_eviction_candidates

```
select_eviction_candidates(gpu_id, num_blocks)
  │
  ├─ 1. 选出本地 LRU 最冷的 num_blocks 个块
  │
  ├─ 2. 目标 GPU 排序：
  │      NVLink 直连伙伴 > 同 Socket 其他 GPU > 跨 Socket GPU
  │
  └─ 3. 对每个冷块找目标：
         ├─ 目标有空闲 → 直接分配
         ├─ 所有目标都满 → 递归驱逐（目标上选冷块搬到更远的 GPU）
         └─ 递归也失败 → 覆盖远端 LRU 最冷块（丢弃远端数据）
```

### 5.5 页表同步

```python
broadcast_page_table():
    gather_local_state()  # 收集所有 rank 的空闲块数
    dist.broadcast_object_list(...)  # Rank 0 广播全局页表

maybe_sync():
    每 sync_interval 轮调度后触发一次 broadcast_page_table
```

### 5.6 当前局限

- `maybe_sync` 当前被注释掉，两个 rank 的页表不同步
- 修复：恢复 `maybe_sync` 调用，或在每次分配/释放块后主动推送到 Rank 0

---

## 6. KV 传输：Transfer

### 6.1 文件位置

`src/myvllm/engine/kv_transfer.py`

### 6.2 核心原语

swap_out：
```
GPU_A 冷块数据 ──NCCL send──► GPU_B 空闲块
逐层传输：for layer in range(num_layers):
            send(k_cache[layer][block_id]) + send(v_cache[layer][block_id])
```

swap_in：
```
GPU_B 远程块 ──NCCL recv──► GPU_A 新分配的空闲块
逐层传输：for layer in range(num_layers):
            recv(k_cache[layer][block_id]) + recv(v_cache[layer][block_id])
```

### 6.3 设计要点

- 传输粒度：逐块、逐层传输 KV 张量切片
- NCCL tag 编码：`block_id * 10000 + layer_idx * 2 + is_k`，避免冲突
- 兼容覆盖写入模式：目标端空闲块不足时直接覆盖 LRU 冷块

---

## 7. 调度器扩展：Scheduler

### 7.1 文件位置

`src/myvllm/engine/scheduler.py`

### 7.2 扩展点

在原始 Mini-vLLM `Scheduler` 的基础上新增了两个决策点：

**Prefill 阶段**：
```python
if self.global_scheduler is not None:
    target_gpu = self.global_scheduler.route_sequence(seq)
    seq.remote_gpu_id = target_gpu if target_gpu != self.rank else -1
    if target_gpu != self.rank:
        # 发到远程 GPU，skip 本地分配
        scheduled_sequences.append(seq)
        continue
```

**Decode 阶段**：
```python
if not self.block_manager.can_append(seq):
    rebalance_success = self.global_scheduler.rebalance(self.rank, 1)
    if rebalance_success:
        self.running.appendleft(seq)  # 下轮重试
        break
    # 否则走原有抢占逻辑
```

---

## 8. 模型执行器：ModelRunner

### 8.1 文件位置

`src/myvllm/engine/model_runner.py`

### 8.2 扩展点

**全局池化钩子**：
```python
def run(self, seqs, is_prefill):
    # 执行前拉取远程块
    if self.gbm is not None:
        for seq in seqs:
            if seq.pending_swap_in:
                self._swap_in_remote_blocks(seq)
    # ... 正常 prefill/decode ...
```

**Swap 执行**：
```python
def execute_swap_out(self, blocks, target_gpu):
    dist.barrier()
    kv_transfer.swap_out(...)

def execute_swap_in(self, remote_gpu, remote_blocks):
    dist.barrier()
    kv_transfer.swap_in(...)
```

---

## 9. 序列管理：Sequence

### 9.1 文件位置

`src/myvllm/engine/sequence.py`

### 9.2 全局池化新增字段

```python
is_remote_prefix: bool = False   # 是否使用远程 GPU 上的前缀
remote_gpu_id: int = -1          # 远程前缀所在 GPU
pending_swap_in: List[int] = []  # 等待拉取的远端块 block_id 列表
```

这些字段通过 `__getstate__/__setstate__` 序列化，确保 `mp.Queue` 跨进程传递时不会丢失。

---

## 10. 配置与运行

### 10.1 关键配置项

```python
config = {
    'world_size': 2,                          # 使用 2 张 GPU
    'enable_global_pool': True,               # 启用全局 KV Cache 池
    'gpu_memory_utilization': 0.3,            # 调小以触发 swap 测试
    'swap_threshold': 0.85,                   # 显存使用率阈值
    'nvlink_topo': {                          # NVLink 拓扑
        'pairs': [(0,2), (1,3), (4,5), (6,7)],
        'sockets': [[0,1,2,3], [4,5,6,7]],
    },
}
```

### 10.2 运行命令

```bash
# 双卡 NVLink 测试
CUDA_VISIBLE_DEVICES=0,2 uv run python main.py

# 单卡基线
CUDA_VISIBLE_DEVICES=0 uv run python main.py
```

---

## 11. 当前状态与下一步

| 功能             | 状态       | 说明                                        |
| ---------------- | ---------- | ------------------------------------------- |
| 对等式多卡推理   | ✅ 完成     | 两个 rank 独立调度、执行、采样              |
| 跨 GPU 序列路由  | ✅ 完成     | `route_sequence` 正常工作                   |
| 全局页表同步     | ❌ 未启用   | `maybe_sync` 被注释，两个 rank 页表独立     |
| swap_out         | ✅ 触发     | 日志已看到 swap_out 执行                    |
| swap_in          | 🔄 进行中   | 端到端待验证                                |
| 前缀复用         | ❌ 未生效   | 页表不同步 + 远程分配不复用已有块           |
| 拓扑感知驱逐     | ✅ 代码就绪 | `select_eviction_candidates` 实现了三级策略 |
| RadixTree 前缀树 | ❌ 未实现   | 当前哈希链足够，后续优化                    |

**下一步**：
1. 恢复页表同步 → 让 `lookup_prefix` 能查到跨 GPU 的前缀命中
2. 跑通 swap 端到端 → 验证 NCCL 传输延迟
3. 实现前缀复用 → 路由到远程 GPU 后直接引用已有块
4. 构造高并发长前缀场景 → 量化吞吐提升