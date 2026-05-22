from collections import deque
import torch.distributed as dist
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.block_manager import BlockManager
from myvllm.engine.global_scheduler import GlobalScheduler


class Scheduler:
    """
    本地调度器

    扩展功能（当启用全局池化时）：
    - prefill 阶段：通过 GlobalScheduler.route_sequence 决定序列归属 GPU
    - decode 阶段：本地空闲不足时，通过 GlobalScheduler.rebalance 触发跨 GPU swap
    - 序列可能被标记为远程前缀（pending_swap_in 非空），后续由 ModelRunner 拉取
    """

    def __init__(
        self,
        max_num_sequences: int,
        max_num_batched_tokens: int,
        max_cached_blocks: int,
        block_size: int,
        eos: int,
        global_scheduler: GlobalScheduler = None,
    ):
        # block manager
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        # sequence queue
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.eos = eos

        # --------------------------------------------- #
        # 全局调度器接口
        # --------------------------------------------- #
        self.global_scheduler = global_scheduler
        
        # 当前 rank（用于路由判断）
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0

    def add_sequence(self, sequence: Sequence):
        self.waiting.append(sequence)


    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        调度

        返回:
            (scheduled_sequences, is_prefill)
            - scheduled_sequences: 本轮需要执行的序列列表
            - is_prefill: True 表示 prefill 阶段，False 表示 decode 阶段
        """
        print(f"[Rank {dist.get_rank()}] schedule() entered, waiting={len(self.waiting)}, running={len(self.running)}")

        scheduled_sequences = []
        current_scheduled_tokens = 0

        # ================================================================
        # Prefill 阶段
        # ================================================================
        while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            seq = self.waiting[0]

            # -------------------------------------------------------- #
            # 全局路由决策
            # -------------------------------------------------------- #
            if self.global_scheduler is not None:
                target_gpu = self.global_scheduler.route_sequence(seq)
                seq.remote_gpu_id = target_gpu if target_gpu != self.rank else -1

                # 如果序列路由到其他 GPU，从本地 waiting 移除并发送过去
                if target_gpu != self.rank:
                    # self.waiting.popleft()
                    # # 标记为远程前缀（如果命中了远程块）
                    # if seq.remote_gpu_id != -1:
                    #     seq.is_remote_prefix = True
                    # # 发送到目标 GPU（通过外部队列，这里只标记）
                    # # 实际发送逻辑在 LLMEngine 中处理
                    # continue
                    seq = self.waiting.popleft()
                    seq.status = SequenceStatus.RUNNING
                    scheduled_sequences.append(seq)       # 保留在调度列表里
                    current_scheduled_tokens += len(seq)
                    # 不在这里分配 block，由目标 rank 自己分配
                    continue
            # -------------------------------------------------------- #

            # 检查是否可以分配
            can_alloc = self.block_manager.can_allocate(seq)

            if not can_alloc:
                # 本地空闲不足：尝试 swap
                if self.global_scheduler is not None:
                    swapped = self.block_manager.allocate_with_swap(seq)
                    if swapped:
                        seq = self.waiting.popleft()
                        seq.status = SequenceStatus.RUNNING
                        self.running.append(seq)
                        current_scheduled_tokens += len(seq)
                        scheduled_sequences.append(seq)
                        continue
                # swap 失败或未启用全局池化：停止 prefill
                break

            # 检查 token 预算
            if len(seq) + current_scheduled_tokens > self.max_num_batched_tokens:
                break

            # 正常分配
            seq = self.waiting.popleft()
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            current_scheduled_tokens += len(seq)
            scheduled_sequences.append(seq)

        if scheduled_sequences:
            # # 周期性同步全局页表
            # if self.global_scheduler is not None:
            #     self.global_scheduler.gbm.maybe_sync()
            return scheduled_sequences, True

        # ================================================================
        # Decode 阶段
        # ================================================================
        while self.running:
            seq = self.running.popleft()
            print(f"[Rank {dist.get_rank()}] decode: seq {seq.seq_id}, num_tokens={seq.num_tokens}")

            # 检查是否可以追加一个 token
            if not self.block_manager.can_append(seq):
                print(f"[Rank {dist.get_rank()}] decode: can_append=False for seq {seq.seq_id}")
                # ---------------------------------------------------- #
                # 全局 rebalance：尝试腾出 1 个块
                # ---------------------------------------------------- #
                rebalance_success = False
                if self.global_scheduler is not None:
                    rebalance_success = self.global_scheduler.rebalance(self.rank, 1)

                if rebalance_success:
                    # rebalance 成功，把序列放回队首，下轮重试
                    self.running.appendleft(seq)
                    break

                # ---------------------------------------------------- #
                # rebalance 失败：原有抢占逻辑
                # ---------------------------------------------------- #
                if self.running:
                    self.running.appendleft(seq)
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                break

            # 检查 token 预算
            if current_scheduled_tokens >= self.max_num_batched_tokens:
                self.running.appendleft(seq)
                break
            if len(scheduled_sequences) >= self.max_num_sequences:
                self.running.appendleft(seq)
                break

            print(f"[Rank {dist.get_rank()}] decode: appending seq {seq.seq_id}")
            # 追加一个 token
            self.block_manager.append(seq)
            scheduled_sequences.append(seq)
            current_scheduled_tokens += 1
            print(f"[Rank {dist.get_rank()}] decode: appended seq {seq.seq_id}")

        # 把已调度的序列重新放回 running 队列末尾（保持轮转顺序）
        if scheduled_sequences:
            self.running.extendleft(reversed(scheduled_sequences))

        # # 周期性同步全局页表
        # if self.global_scheduler is not None:
        #     self.global_scheduler.gbm.maybe_sync()

        print(f"[Rank {dist.get_rank()}] schedule() returning, scheduled={len(scheduled_sequences)}")
        return scheduled_sequences, False


    def preempt(self, seq: Sequence) -> None:
        """抢占序列：释放块，放回 waiting 队首"""
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        seq.num_cached_tokens = 0
        seq.block_table = []
        # 清除远程前缀标记（抢占后重新调度可能换 GPU）
        seq.is_remote_prefix = False
        seq.remote_gpu_id = -1
        seq.pending_swap_in = []
        self.waiting.appendleft(seq)


    # postprocess after generation to check whether sequences are finished
    # if finished, deallocate blocks
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        """
        生成后处理：追加 token，检查停止条件，释放已完成序列的块
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)

            # 检查停止条件
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens
            stop_due_to_max_length = (
                seq.max_model_length is not None
                and seq.num_tokens >= seq.max_model_length
            )

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

            if seq.status == SequenceStatus.FINISHED:
                self.block_manager.deallocate(seq)
            elif seq.status == SequenceStatus.WAITING:
                # 进到这里说明该序列顺利完成了推理，且没有触发 FINISHED 结束条件
                # 意味着它刚刚完成了 Prefill 阶段，现在需要强制进入 Decode 状态
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)  # 塞进 running 队列，供下一轮 schedule() 挑出来做 DECODE