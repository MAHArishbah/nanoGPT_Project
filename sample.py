"""
Generate text from a model trained with train.py, or from OpenAI GPT-2 weights.

$ python sample.py

Every setting lives in config/sample_config.py.
"""

from contextlib import nullcontext
from pathlib import Path

import tiktoken
import torch

from config import SampleConfig
from model import GPT

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_device(name):
    """Turn the 'auto' device setting into 'cuda' or 'cpu'; any other value is returned unchanged."""
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def get_autocast_context(device_type, dtype):
    """Return the mixed-precision context for generation: bf16 autocast, or a no-op for float32."""
    if dtype == "bfloat16" and device_type == "cuda" and not torch.cuda.is_bf16_supported():
        print("WARNING: this GPU does not support bfloat16, falling back to float32")
        dtype = "float32"
    if dtype == "float32":
        return nullcontext()
    if dtype == "bfloat16":
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    raise ValueError(f"unsupported dtype {dtype!r}, use 'bfloat16' or 'float32'")


def load_model(cfg, device):
    """Build the model in eval mode on device, from a train.py checkpoint or from GPT-2 weights."""
    if cfg.init_from == "resume":
        ckpt_path = PROJECT_ROOT / cfg.out_dir / cfg.ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"no checkpoint at {ckpt_path}; run train.py first")
        # weights_only=True only unpickles tensors and plain Python types, never arbitrary objects
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        model = GPT.from_checkpoint(checkpoint)
        print(
            f"loaded {ckpt_path.name}: epoch {checkpoint['epoch']}, step {checkpoint['iter_num']}, "
            f"val loss {checkpoint['val_loss']:.4f}"
        )
    elif cfg.init_from.startswith("gpt2"):
        model = GPT.from_pretrained(cfg.init_from, dict(dropout=0.0))
    else:
        raise ValueError(f"unknown init_from {cfg.init_from!r}")

    model.eval()
    model.to(device)
    if cfg.compile:
        model = torch.compile(model)
    return model


def read_prompt(start):
    """Return the prompt text; a value of the form 'FILE:path' is replaced by that file's contents."""
    if start.startswith("FILE:"):
        with open(start[5:], "r", encoding="utf-8") as f:
            return f.read()
    return start


def main(cfg):
    """Load the model, encode the prompt and print cfg.num_samples generated continuations."""
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")  # TF32 matmuls on Ampere and newer GPUs
    torch.backends.cudnn.allow_tf32 = True
    device = resolve_device(cfg.device)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    ctx = get_autocast_context(device_type, cfg.dtype)

    model = load_model(cfg, device)

    # the models are trained on GPT-2 BPE tokens, so the GPT-2 tokenizer encodes and decodes
    enc = tiktoken.get_encoding("gpt2")
    start_ids = enc.encode(read_prompt(cfg.start), allowed_special={"<|endoftext|>"})
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]  # (1, T)

    with torch.no_grad(), ctx:
        for _ in range(cfg.num_samples):
            y = model.generate(x, cfg.max_new_tokens, temperature=cfg.temperature, top_k=cfg.top_k)
            # vocab_size is padded to 50304; ids above GPT-2's 50257 real tokens have no text, so drop them
            tokens = [t for t in y[0].tolist() if t < enc.n_vocab]
            print(enc.decode(tokens))
            print("---------------")


if __name__ == "__main__":
    main(SampleConfig())
