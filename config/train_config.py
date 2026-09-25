"""
Settings for train.py.

Edit the defaults here to change a run. The defaults reproduce the notebook:
GPT-2 small (124M) trained on data/shakespeare_input.txt with ~0.5M tokens per
optimizer step. That dataset is ~1.25M training tokens, so one epoch is 2 optimizer
steps and 25 epochs give the notebook's 50 steps.
"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Every setting used by train.py. Relative paths are resolved from the project root."""

    # I/O
    out_dir: str = "/content/drive/MyDrive/nanogpt_out/fineweb_modern_1ep"  # ckpt.pt (latest evaluation) and best.pt (lowest val loss) are written here
    log_interval: int = 10  # print and log training metrics every N optimizer steps
    eval_interval: int = 100  # also evaluate and checkpoint every N optimizer steps; 0: only at the end of each epoch
    eval_iters: int = 20  # max batches per split that estimate_loss averages over
    eval_only: bool = False  # if True, evaluate once and exit without training
    init_from: str = "resume"  # 'scratch', 'resume' (from out_dir/ckpt.pt) or 'gpt2', 'gpt2-medium', ...

    # wandb logging
    wandb_log: bool = True
    wandb_project: str = "nanogpt"
    wandb_run_name: str = "12L-768d-rmsnorm-swiglu-rope-1ep-7.4B"# empty: named from the architecture, e.g. "12L-768d-rmsnorm-swiglu-rope"


    # data
    data_dir: str = "/content/data"
    dataset: str = "fineweb"  # text file inside data_dir
    val_fraction: float = 0.1  # the last 10% of tokens form the validation split
    batch_size: int = 64  # micro-batch size: sequences per forward/backward passepoc
    block_size: int = 1024  # context length in tokens
    total_batch_size: int = 524288  # tokens per optimizer step (2**19, ~0.5M as in the GPT-3 paper)
    num_workers: int = 0  # DataLoader worker processes; 0 loads batches in the main process

    # model
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50304  # GPT-2's 50257 tokens padded up to a multiple of 64
    dropout: float = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = True  # bias inside LayerNorm and Linear layers, like GPT-2
    norm: str = "rmsnorm"  # 'layernorm' (like GPT-2) or 'rmsnorm'
    mlp: str = "swiglu"  # 'gelu' (like GPT-2) or 'swiglu'
    pos_emb: str = "rope"  # 'learned' (like GPT-2) or 'rope'


    # AdamW optimizer
    learning_rate: float = 6e-4  # max learning rate, GPT-3 paper value for the 124M model
    max_epochs: int = 1 # one epoch is one full pass over the training split
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip the global gradient norm at this value, 0.0 disables clipping

    # learning rate schedule: linear warmup, then cosine decay to min_lr at the last step
    decay_lr: bool = True  # if False, learning_rate is used for every step
    warmup_frac: float = 0.05  # fraction of all optimizer steps spent warming up
    min_lr: float = 6e-5  # ~= learning_rate / 10 per Chinchilla

    # DDP
    backend: str = "nccl"  # 'nccl' for GPUs, 'gloo' for CPU

    # system
    device: str = "auto"  # 'auto' (cuda if available, else cpu), 'cpu', 'cuda', 'cuda:0', ...
    dtype: str = "bfloat16"  # 'bfloat16' (autocast) or 'float32'
    compile: bool = True  # torch.compile the model (needs Triton, i.e. Linux + GPU in practice)
    seed: int = 1337
