# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="PATH/to/Model"
    tokenizer_name: str=None
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=8
    batching_strategy: str="packing" #alternative: padding
    context_length: int=1024
    gradient_accumulation_steps: int=1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=3
    max_train_step: int=0
    max_eval_step: int=0
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=True
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "custom_dataset"
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=True
    output_dir: str = "PATH/to/output"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = True
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    use_wandb: bool = False # Enable wandb for experient tracking
    save_metrics: bool = False # saves training metrics to a json file for later plotting
    flop_counter: bool = False # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
    flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    use_profiler: bool = False # Enable pytorch profiler, can not be used with flop counter at the same time.
    profiler_dir: str = "PATH/to/save/profiler/results" # will be used if using profiler
