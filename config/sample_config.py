"""
Settings for sample.py.
"""

from dataclasses import dataclass


@dataclass
class SampleConfig:
    """Every setting used by sample.py. Relative paths are resolved from the project root."""

    init_from: str = "resume"  # 'resume' (load out_dir/ckpt_name) or a GPT-2 variant, e.g. 'gpt2-xl'
    out_dir: str = "out"  # ignored unless init_from is 'resume'
    ckpt_name: str = "best.pt"  # 'best.pt' (lowest val loss) or 'ckpt.pt' (latest epoch)
    start: str = "<|endoftext|>Machine Learning is " # prompt, or "<|endoftext|>", or "FILE:prompt.txt" to read the prompt from a file
    num_samples: int = 3  # number of samples to draw
    max_new_tokens: int = 300  # tokens generated per sample
    temperature: float = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random
    top_k: int = 200  # keep only the top_k most likely tokens, the rest get zero probability
    seed: int = 1337
    device: str = "auto"  # 'auto' (cuda if available, else cpu), 'cpu', 'cuda', 'cuda:0', ...
    dtype: str = "float32"  # 'bfloat16' (autocast) or 'float32'
    compile: bool = False  # torch.compile the model
