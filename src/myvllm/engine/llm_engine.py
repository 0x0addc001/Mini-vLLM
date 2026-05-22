import atexit
import torch
import torch
import torch.distributed as dist
import time
import torch.multiprocessing as mp
from multiprocessing import Queue

from myvllm.engine.sequence import Sequence
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.global_block_manager import GlobalBlockManager
from myvllm.engine.global_scheduler import GlobalScheduler
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer


def _as_token_list(tokens):
    return [t.item() if isinstance(t, torch.Tensor) else t for t in tokens]


# def worker_process(config, rank, event):
def worker_process(config, rank, recv_queue: Queue, send_queue: Queue):
    """Worker process function that initializes ModelRunner and enters loop."""
    # FIRST print before any other code
    import sys
    import os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # Line buffering
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    # model_runner = ModelRunner(config, rank, event)
    # model_runner.loop()

    # 创建 GlobalBlockManager（共享）
    gbm = GlobalBlockManager(
        rank=rank,
        world_size=config['world_size'],
        num_blocks_per_gpu=config['max_cached_blocks'],
        nvlink_pairs=config.get('nvlink_topo', {}).get('pairs', []),
        socket_groups=config.get('nvlink_topo', {}).get('sockets', []),
    )
    # model_runner = ModelRunner(config, rank, event, gbm)
    model_runner = ModelRunner(config, rank, gbm)
    scheduler = Scheduler(
        max_num_sequences=config.get("max_num_sequences", 16),
        max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
        max_cached_blocks=config.get("max_cached_blocks", 1024),
        block_size=config.get("block_size", 256),
        eos=config.get("eos", 50256),
        global_scheduler=GlobalScheduler(gbm=gbm, block_manager=None),
    )
    # scheduler.global_scheduler.block_manager = scheduler.block_manager
    # # 进入独立推理循环
    # while True:
    #     # 从某个队列接收序列，或者由外部推送
    #     seqs = receive_sequences()  # 需要实现
    #     if seqs:
    #         for seq in seqs:
    #             scheduler.add_sequence(seq)
    #     scheduled, is_prefill = scheduler.schedule()
    #     if scheduled:
    #         outputs = model_runner.run(scheduled, is_prefill)
    #         scheduler.postprocess(scheduled, outputs)

    # 独立推理循环
    was_idle = False
    while True:
        # 1. 接收 Rank 0 发来的序列
        received_any = False
        try:
            while True:
                msg = recv_queue.get_nowait()
                received_any = True
                if msg.get("type") == "exit":
                    model_runner.exit()
                    return
                if msg.get("type") == "sequence":
                    seq = msg["seq"]
                    print(f"[Rank {rank}] Received seq: type={type(seq)}, seq_id={getattr(seq, 'seq_id', 'MISSING')}, tokens={seq.token_ids[:10]}...")
                    scheduler.add_sequence(seq)
                    # scheduler.add_sequence(msg["seq"])
                    was_idle = False
        except Exception as e:
            if received_any:
                print(f"[Rank {rank}] recv drained: {e}")
            pass

        # 2. 调度
        print(f"[Rank {rank}] Before schedule: waiting={len(scheduler.waiting)}, running={len(scheduler.running)}")
        scheduled, is_prefill = scheduler.schedule()
        print(f"[Rank {rank}] After schedule: scheduled={len(scheduled)}, is_prefill={is_prefill}")
        # scheduled, is_prefill = scheduler.schedule()
        if not scheduled:
            # 如果本地没有序列且没有外部输入，短暂空转
            if scheduler.is_finished() and recv_queue.empty():
                if not was_idle: # 只在首次空闲时发
                    print(f"[Rank {rank}] Sending idle, queue size={send_queue.qsize()}")
                    # 通知 Rank 0 当前空闲
                    # send_queue.put({"type": "idle", "rank": rank})
                    try:
                        send_queue.put_nowait({"type": "idle", "rank": rank})
                    except:
                        pass
                    was_idle = True
                    print(f"[Rank {rank}] Idle sent")
                time.sleep(0.01)
            continue
        
        was_idle = False  # 有活干了，重置
        
        # 3. 分离本地和远程序列
        local_seqs = [s for s in scheduled if s.remote_gpu_id in (-1, rank)]
        remote_seqs = [s for s in scheduled if s.remote_gpu_id not in (-1, rank)]

        # 4. 发远程序列到目标 rank
        for seq in remote_seqs:
            send_queue.put({"type": "sequence", "target": seq.remote_gpu_id, "seq": seq})

        # 5. 本地执行
        if local_seqs:
            outputs = model_runner.run(local_seqs, is_prefill)
            # scheduler.postprocess(local_seqs, outputs)
            print(f"[Rank {rank}] Before postprocess")
            scheduler.postprocess(local_seqs, outputs)
            print(f"[Rank {rank}] After postprocess")

        # 6. 收集完成的序列，回传 Rank 0
        # finished = [(s.seq_id, s.completion_token_ids) for s in scheduled if s.is_finished]
        finished = [(s.seq_id, _as_token_list(s.completion_token_ids))
            for s in scheduled if s.is_finished]
        if finished:
            print(f"[Rank {rank}] Sending finished: {finished}")
            send_queue.put({"type": "finished", "data": finished})


class LLMEngine:
    """
    LLM 推理引擎（编排层）

    职责：
    - 不亲自做路由决策或显存决策
    - 按正确顺序调用 GlobalScheduler -> Scheduler -> ModelRunner
    - 管理多 GPU worker 进程的生命周期
    """

    def __init__(self, config: dict):
        self.config = config
        self.world_size = config.get("world_size", 1)
        ctx = mp.get_context("spawn")

        self.processes = []
        self.recv_queues = {}   # rank -> Queue，用于接收其他 rank 的消息
        self.send_queues = {}   # rank -> Queue，用于向其他 rank 发送消息

        self.remote_finished = set()
        self.remote_inflight_seq_ids = set()

        # 1. 先启动所有 worker 进程（在 dist 初始化之前）
        if self.world_size > 1:
            for i in range(1, self.world_size):
                recv_q = ctx.Queue()
                send_q = ctx.Queue()
                self.recv_queues[i] = recv_q
                self.send_queues[i] = send_q
                process = ctx.Process(
                    target=worker_process,
                    # args=(config, i, recv_q, send_q)
                    args=(config, i, send_q, recv_q)  # worker 从 send_q 接收，向 recv_q 发送
                )
                self.processes.append(process)
                process.start()
        # 2. 创建 ModelRunner（内部调 dist.init_process_group，所有 rank 同步）
        self.model_runner = ModelRunner(config, rank=0, gbm=None)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.get("model_name_or_path", "gpt2")
        )
        # Rank 0 的 GlobalBlockManager
        # gbm = GlobalBlockManager(
        #     rank=0,
        #     world_size=self.world_size,
        #     num_blocks_per_gpu=config['max_cached_blocks'],
        #     nvlink_pairs=config.get('nvlink_topo', {}).get('pairs', []),
        #     socket_groups=config.get('nvlink_topo', {}).get('sockets', []),
        # ) if config.get('enable_global_pool', False) else None
        #  # 现在 dist 已经初始化了，可以创建 GBM

        # 3. dist 已初始化，创建 GBM
        gbm = None
        if config.get('enable_global_pool', False):
            gbm = GlobalBlockManager(
                rank=0,
                world_size=self.world_size,
                num_blocks_per_gpu=config['max_cached_blocks'],
                nvlink_pairs=config.get('nvlink_topo', {}).get('pairs', []),
                socket_groups=config.get('nvlink_topo', {}).get('sockets', []),
            )
            # 把 GBM 回填到 ModelRunner
            self.model_runner.gbm = gbm

        # 4. 创建 Scheduler
        self.scheduler = Scheduler(
            max_num_sequences=config.get("max_num_sequences", 16),
            max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
            max_cached_blocks=config.get("max_cached_blocks", 1024),
            block_size=config.get("block_size", 256),
            eos=config.get("eos", 50256),
            global_scheduler=GlobalScheduler(gbm=gbm, block_manager=None) if gbm else None,
        )
        if gbm:
            self.scheduler.global_scheduler.block_manager = self.scheduler.block_manager
        atexit.register(self.exit)

        # self.events = []
        # for i in range(1, world_size):
        #     event = ctx.Event()
        #     process = ctx.Process(target=worker_process, args=(config, i, event))
        #     self.events.append(event)
        #     self.processes.append(process)
        #     process.start()
        # # start the engine only on the master thread with rank = 0
        # self.model_runner = ModelRunner(config, rank=0, event=self.events)
        # self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name_or_path", "gpt2"))
        # # ------------------------------------------------------------ #
        # # 初始化全局调度器（仅在启用全局池化时）
        # # ------------------------------------------------------------ #
        # self.enable_global_pool = config.get('enable_global_pool', False)
        # if self.enable_global_pool:
        #     # GlobalBlockManager 已在 ModelRunner 中初始化，直接复用
        #     gbm = self.model_runner.gbm
        #     # GlobalScheduler 需要 BlockManager 的 compute_hash 接口
        #     # 但 BlockManager 在 Scheduler 中——先创建临时引用，后面设置
        #     self.global_scheduler = GlobalScheduler(
        #         gbm=gbm,
        #         block_manager=None,  # 暂时为空，下面补上
        #     )
        # else:
        #     self.global_scheduler = None
        # # ------------------------------------------------------------ #
        # scheduler needs to init after model_runner: when world_size > 1,
        # ModelRunner.__init__ calls dist.init_process_group() which is a
        # collective barrier — rank-0 blocks until all worker ranks have joined.
        # The scheduler should only be created after that rendezvous completes.
        # When world_size == 1 there is no barrier and no real dependency.
        # self.scheduler = Scheduler(
        #     max_num_sequences=config.get("max_num_sequences", 16),
        #     max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
        #     max_cached_blocks=config.get("max_cached_blocks", 1024),
        #     block_size=config.get("block_size", 256),
        #     eos=config.get("eos", 50256),
        #     global_scheduler=self.global_scheduler,  # 传入全局调度器
        # )
        # # ------------------------------------------------------------ #
        # # 回填 GlobalScheduler 的 block_manager 引用
        # # ------------------------------------------------------------ #
        # if self.global_scheduler is not None:
        #     self.global_scheduler.block_manager = self.scheduler.block_manager
        # # ------------------------------------------------------------ #
        # atexit.register(self.exit)


    def exit(self):
        # self.model_runner.call("exit")
        for rank, q in self.send_queues.items():
            q.put({"type": "exit"})
        self.model_runner.exit()
        del self.model_runner
        for process in self.processes:
            process.join()

    # call scheduler to schedule the next batch
    # return scheduled sequences and whether it is for prefilling
    # call model_runner.run() to run the model
    # call postprocessor to process the outputs and update sequences and update block manager
    def _drain_worker_messages(self) -> list[tuple[int, list[int]]]:
        finished = []
        for rank, q in self.recv_queues.items():
            try:
                while True:
                    msg = q.get_nowait()
                    msg_type = msg.get("type")
                    if msg_type == "finished":
                        for seq_id, tokens in msg["data"]:
                            finished.append((seq_id, tokens))
                            self.remote_inflight_seq_ids.discard(seq_id)
                    elif msg_type == "idle":
                        self.remote_finished.add(rank)
            except:
                pass
        return finished

    def step(self) -> tuple[list[int], bool]:
        """
        推理

        流程：
        1. Scheduler.schedule() -> 选出一批序列（内部已调用 GlobalScheduler 做路由和 rebalance）
        2. ModelRunner.run() -> 执行模型 forward（内部已拉取远程块）
        3. postprocess -> 追加 token、检查停止条件

        返回:
            (outputs, num_processed_tokens, is_prefill)
        """
        # # 每一步开始前同步页表
        # if self.scheduler.global_scheduler is not None:
        #     self.scheduler.global_scheduler.gbm.maybe_sync()
        scheduled_sequences, is_prefill = self.scheduler.schedule()
        finished = self._drain_worker_messages()
        if not scheduled_sequences:
            if not finished:
                time.sleep(0.01)
            return finished, 0, is_prefill
        
        # run the model
        # outputs = self.model_runner.call("run", scheduled_sequences, is_prefill)
        # Move outputs to CPU and convert them to a list
        # if outputs is not None:
        #     outputs = outputs.cpu().tolist()
        # ------------------------------------------------------------ #
        # 按目标 GPU 拆分序列
        # ------------------------------------------------------------ #
        # local_seqs = []
        # remote_seqs = []
        # for seq in scheduled_sequences:
        #     if seq.remote_gpu_id == -1 or seq.remote_gpu_id == 0:
        #         local_seqs.append(seq)
        #     else:
        #         remote_seqs.append(seq)
        # 分离本地和远程序列
        local_seqs = [s for s in scheduled_sequences if s.remote_gpu_id in (-1, 0)]
        remote_seqs = [s for s in scheduled_sequences if s.remote_gpu_id not in (-1, 0)]
        
         # 发远程序列
        for seq in remote_seqs:
            target = seq.remote_gpu_id
            if target in self.send_queues:
                self.send_queues[target].put({"type": "sequence", "seq": seq})
                self.remote_inflight_seq_ids.add(seq.seq_id)
                self.remote_finished.discard(target)

        # # 本地执行
        # if local_seqs:
        #     outputs = self.model_runner.run(local_seqs, is_prefill)
        #     self.scheduler.postprocess(local_seqs, outputs)

        # 本地执行
        if local_seqs:
            print(f"[Rank 0] Running local prefill...")
            outputs = self.model_runner.run(local_seqs, is_prefill)
            print(f"[Rank 0] Local prefill done, outputs={outputs}")
            self.scheduler.postprocess(local_seqs, outputs)
            # 本地完成的序列
            finished.extend([(s.seq_id, _as_token_list(s.completion_token_ids)) for s in local_seqs if s.is_finished])
        # 收集其他 rank 完成的序列
        print(f"[Rank 0] Collecting finished from ranks {list(self.recv_queues.keys())}")
        finished.extend(self._drain_worker_messages())

        num_tokens = sum(len(s) for s in scheduled_sequences) if is_prefill else len(scheduled_sequences)
        return finished, num_tokens, is_prefill
        
        # # Rank 0 执行本地序列
        # local_outputs = None
        # if local_seqs:
        #     local_outputs = self.model_runner.call("run", local_seqs, is_prefill)
        # # 把远程序列发给对应的 Rank
        # remote_outputs = {}
        # if remote_seqs and self.world_size > 1:
        #     # 按目标 GPU 分组
        #     groups: dict[int, list] = {}
        #     for seq in remote_seqs:
        #         groups.setdefault(seq.remote_gpu_id, []).append(seq)
        #     for target_gpu, seqs_group in groups.items():
        #         self.model_runner.call("run", seqs_group, is_prefill)  # Rank 1 通过 loop() 收到并执行
        #         # 注意：Rank 1 的 run() 返回 None（不采样），这里需要额外机制取回 logits
        #         # 暂时 Rank 1 只写 KV cache，采样仍在 Rank 0 做
        #         remote_outputs[target_gpu] = None
        # # 合并 outputs
        # outputs = local_outputs  # Rank 1 的输出暂不取回，简化处理
        # if outputs is not None:
        #     outputs = outputs.cpu().tolist()
        
        # # postprocess the outputs
        # self.scheduler.postprocess(scheduled_sequences, outputs)

        # outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]
        # num_processed_tokens = sum(len(seq) for seq in scheduled_sequences) if is_prefill else len(scheduled_sequences)

        # return outputs, num_processed_tokens, is_prefill


    # add prompt string to the waiting queue by first transforming it to Sequence object
    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> None:
        """
        添加推理请求

        全局路由决策已在 Scheduler.schedule() 的 prefill 阶段完成，
        这里只负责把原始文本转为 Sequence 对象并放入 waiting 队列
        """
        self.scheduler.add_sequence(Sequence(
            token_ids=self.tokenizer.encode(prompt),
            block_size=self.config['block_size'],
            sampling_params=sampling_params
        ))

    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
        """批量推理入口"""
        for prompt in prompts:
            self.add_prompt(prompt, sampling_params)
        generated_tokens = {}
        # while not self.scheduler.is_finished():
        # 退出条件：本地空闲 + 所有远程 rank 都发回了 finished
        expected_num_outputs = len(prompts)
        while len(generated_tokens) < expected_num_outputs:
            start_t = time.time()
            # outputs, num_processed_tokens, is_prefill = self.step()
            try:
                outputs, num_processed_tokens, is_prefill = self.step()
            except Exception as e:
                import traceback
                print(f"\n[Rank {self.rank}] ====== Engine step error ======")
                traceback.print_exc()
                print(f"[Rank {self.rank}] ===============================\n")
                # 发生异常时，由于无法获取本轮数据，为了防止后续代码因变量未定义报错，这里直接中断或跳过
                break
            end_t = time.time()
            running_time = end_t - start_t + 1e-10
            phase = "prefilling" if is_prefill else "decoding"
            print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during", phase)
            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})
        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]
        output = {'text': [self.tokenizer.decode(tokens) for tokens in generated_tokens], 'token_ids': generated_tokens}
        return output
