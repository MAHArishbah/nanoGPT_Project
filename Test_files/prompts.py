"""
Run a list of prompts through a trained model, loading the checkpoint only once.

$ python prompts.py                                  # the built-in probe set below
$ python prompts.py --max-new-tokens 250             # longer continuations
$ python prompts.py --prompts-file my_prompts.txt    # one prompt per line
$ python prompts.py --out results/probes.txt         # save everything

Every prompt is given a leading <|endoftext|> unless it already has one: as training data
has docs which starts with that token and prompting without it starts the model in a state it never saw.
"""

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import tiktoken
import torch

from model import GPT

PROJECT_ROOT = Path(__file__).resolve().parent
EOT = "<|endoftext|>"

# Each group probes a different ability. Edit freely, or use --prompts-file instead.
PROMPTS = [
    # topic retrieval: well covered by FineWeb-Edu, should be among the better outputs
    "Artificial intelligence is",
    "A neural network is a",
    "Machine learning allows computers to",
    # factual recall: does 7.4B tokens memorise anything? high-frequency facts only
    "The capital of France is",
    "Water boils at",
    # in-context learning: can it infer a pattern from three examples? 124M is right at the edge
    "apple: fruit\ncarrot: vegetable\nsalmon: fish\noak:",
    # base model check: it completes documents, it does not answer you. Expect more Q&A pairs
    "Q: What causes rain?\nA:",
    # document structure, separate from content
    "# Introduction to",
    # outside its strength: FineWeb-Edu filters *for* educational value, so little code
    "def fibonacci(n):",
]


def resolve_device(name):
    """Turn the 'auto' device setting into 'cuda' or 'cpu'; any other value is returned unchanged."""
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def get_autocast_context(device_type, dtype):
    """Return the mixed-precision context for generation: bf16 autocast, or a no-op for float32."""
    if dtype == "auto":
        dtype = "bfloat16" if device_type == "cuda" else "float32"
    if dtype == "bfloat16" and device_type == "cuda" and not torch.cuda.is_bf16_supported():
        print("WARNING: this GPU does not support bfloat16, falling back to float32")
        dtype = "float32"
    if dtype == "float32":
        return nullcontext()
    if dtype == "bfloat16":
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    raise ValueError(f"unsupported dtype {dtype!r}, use 'auto', 'bfloat16' or 'float32'")


def load_prompts(prompts_file):
    """Return the prompts to run: the file's non-blank, non-comment lines, or the built-in list.

    In a file, '\\n' is turned into a real newline so multi-line prompts fit on one line.
    """
    if not prompts_file:
        return PROMPTS
    path = Path(prompts_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.replace("\\n", "\n") for ln in lines if ln.strip() and not ln.startswith("#")]


def load_model(ckpt_path, device):
    """Load a train.py checkpoint in eval mode on device and report what it was."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")
    # weights_only=True only unpickles tensors and plain Python types, never arbitrary objects
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = GPT.from_checkpoint(checkpoint)
    model.eval()
    model.to(device)
    print(
        f"loaded {ckpt_path.name}: epoch {checkpoint['epoch']}, step {checkpoint['iter_num']}, "
        f"val loss {checkpoint['val_loss']:.4f}\n"
    )
    return model


@torch.no_grad()
def complete(model, enc, prompt, args, device, ctx):
    """Return the model's continuation of prompt (the prompt itself is stripped off)."""
    text = prompt if prompt.startswith(EOT) or args.no_eot else EOT + prompt
    start_ids = enc.encode(text, allowed_special={EOT})
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]  # (1, T)

    with ctx:
        y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)

    # keep only what was generated, and drop the ids above GPT-2's 50257 real tokens:
    # vocab_size is padded to 50304 for speed, and the padding ids decode to nothing
    new_tokens = [t for t in y[0].tolist()[len(start_ids):] if t < enc.n_vocab]
    return enc.decode(new_tokens)


def main(args):
    device = resolve_device(args.device)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    ctx = get_autocast_context(device_type, args.dtype)
    torch.set_float32_matmul_precision("high")  # TF32 matmuls on Ampere and newer GPUs

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    model = load_model(ckpt_path, device)

    enc = tiktoken.get_encoding("gpt2")
    prompts = load_prompts(args.prompts_file)
    print(f"{len(prompts)} prompts x {args.num_samples} sample(s), {args.max_new_tokens} tokens each\n")

    lines = []
    t_start = time.time()
    for i, prompt in enumerate(prompts, 1):
        for s in range(args.num_samples):
            # re-seed per sample so each one is reproducible on its own and the order
            # of the prompt list does not change any individual result
            torch.manual_seed(args.seed + s)
            t0 = time.time()
            continuation = complete(model, enc, prompt, args, device, ctx)
            label = f"[{i}/{len(prompts)}]" + (f" sample {s + 1}" if args.num_samples > 1 else "")
            block = (
                f"{'=' * 70}\n{label}  PROMPT: {prompt!r}\n{'-' * 70}\n"
                f"{prompt}{continuation}\n"
                f"({time.time() - t0:.1f}s)\n"
            )
            print(block, flush=True)  # flush: the output stays live when redirected to a file
            lines.append(block)

    print(f"done in {time.time() - t_start:.0f}s")
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(lines), encoding="utf-8")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--ckpt", default="out/best.pt", help="checkpoint to sample from")
    parser.add_argument("--prompts-file", default="", help="text file with one prompt per line")
    parser.add_argument("--max-new-tokens", type=int, default=150, help="tokens generated per prompt")
    parser.add_argument("--num-samples", type=int, default=1, help="samples per prompt")
    parser.add_argument("--temperature", type=float, default=0.8, help="<1.0 less random, >1.0 more random")
    parser.add_argument("--top-k", type=int, default=200, help="keep only the top k tokens (0: no limit)")
    parser.add_argument("--no-eot", action="store_true", help="do not prepend <|endoftext|> to prompts")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', ...")
    parser.add_argument("--dtype", default="auto", help="'auto' (bf16 on GPU, fp32 on CPU), 'bfloat16', 'float32'")
    parser.add_argument("--out", default="", help="also write everything to this text file")
    args = parser.parse_args()
    if args.top_k == 0:
        args.top_k = None
    main(args)
