import atexit
import torch.distributed as dist
import time
import torch.multiprocessing as mp

from myvllm.engine.sequence import Sequence
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.global_block_manager import GlobalBlockManager
from myvllm.engine.global_scheduler import GlobalScheduler
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer


def worker_process(config, rank, event):
    """Worker process function that initializes ModelRunner and enters loop."""
    # FIRST print before any other code
    import sys
    import os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # Line buffering
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    model_runner = ModelRunner(config, rank, event)
    model_runner.loop()


class LLMEngine:
    """
    LLM 推理引擎（编排层）

    职责：
    - 不亲自做路由决策或显存决策
    - 按正确顺序调用 GlobalScheduler → Scheduler → ModelRunner
    - 管理多 GPU worker 进程的生命周期
    """

    def __init__(self, config: dict):
        self.config = config
        world_size = config.get("world_size", 1)
        ctx = mp.get_context("spawn")
        self.processes = []
        self.events = []
        for i in range(1, world_size):
            event = ctx.Event()
            process = ctx.Process(target=worker_process, args=(config, i, event))
            self.events.append(event)
            self.processes.append(process)
            process.start()

        # start the engine only on the master thread with rank = 0
        self.model_runner = ModelRunner(config, rank=0, event=self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name_or_path", "gpt2"))

        # ------------------------------------------------------------ #
        # 初始化全局调度器（仅在启用全局池化时）
        # ------------------------------------------------------------ #
        self.enable_global_pool = config.get('enable_global_pool', False)
        if self.enable_global_pool:
            # GlobalBlockManager 已在 ModelRunner 中初始化，直接复用
            gbm = self.model_runner.gbm
            # GlobalScheduler 需要 BlockManager 的 compute_hash 接口
            # 但 BlockManager 在 Scheduler 中——先创建临时引用，后面设置
            self.global_scheduler = GlobalScheduler(
                gbm=gbm,
                block_manager=None,  # 暂时为空，下面补上
            )
        else:
            self.global_scheduler = None
        # ------------------------------------------------------------ #

        # scheduler needs to init after model_runner: when world_size > 1,
        # ModelRunner.__init__ calls dist.init_process_group() which is a
        # collective barrier — rank-0 blocks until all worker ranks have joined.
        # The scheduler should only be created after that rendezvous completes.
        # When world_size == 1 there is no barrier and no real dependency.
        self.scheduler = Scheduler(
            max_num_sequences=config.get("max_num_sequences", 16),
            max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
            max_cached_blocks=config.get("max_cached_blocks", 1024),
            block_size=config.get("block_size", 256),
            eos=config.get("eos", 50256),
            global_scheduler=self.global_scheduler,  # 传入全局调度器
        )

        # ------------------------------------------------------------ #
        # 回填 GlobalScheduler 的 block_manager 引用
        # ------------------------------------------------------------ #
        if self.global_scheduler is not None:
            self.global_scheduler.block_manager = self.scheduler.block_manager
        # ------------------------------------------------------------ #

        atexit.register(self.exit)


    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for process in self.processes:
            process.join()

    # call scheduler to schedule the next batch
    # return scheduled sequences and whether it is for prefilling
    # call model_runner.run() to run the model
    # call postprocessor to process the outputs and update sequences and update block manager
    def step(self) -> tuple[list[int], bool]:
        """
        推理

        流程：
        1. Scheduler.schedule() → 选出一批序列（内部已调用 GlobalScheduler 做路由和 rebalance）
        2. ModelRunner.run() → 执行模型 forward（内部已拉取远程块）
        3. postprocess → 追加 token、检查停止条件

        返回:
            (outputs, num_processed_tokens, is_prefill)
        """
        scheduled_sequences, is_prefill = self.scheduler.schedule()
        if not scheduled_sequences:
            return [], 0, is_prefill
        # run the model
        outputs = self.model_runner.call("run", scheduled_sequences, is_prefill)
        # Move outputs to CPU and convert them to a list
        if outputs is not None:
            outputs = outputs.cpu().tolist()
        # postprocess the outputs
        self.scheduler.postprocess(scheduled_sequences, outputs)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]
        num_processed_tokens = sum(len(seq) for seq in scheduled_sequences) if is_prefill else len(scheduled_sequences)

        return outputs, num_processed_tokens, is_prefill


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
        while not self.scheduler.is_finished():
            start_t = time.time()
            outputs, num_processed_tokens, is_prefill = self.step()
            end_t = time.time()
            running_time = end_t - start_t + 1e-10
            if is_prefill:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during prefilling")
            else:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during decoding")
            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})

        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]
        output = {'text': [self.tokenizer.decode(tokens) for tokens in generated_tokens], 'token_ids': generated_tokens}
        return output
