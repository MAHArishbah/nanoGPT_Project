"""
Evaluate a model trained with train.py on HellaSwag (the validation split, 10,042 examples).

$ python hellaswag.py --limit 1000          # quick subset, ~3 min on 4 CPU threads
$ python hellaswag.py                       # full set: ~35 min on 4 CPU threads, ~1 min on an A100

Each example is a context plus four candidate endings, one of which is correct. The model
never sees the choices as choices: every ending is scored on its own by the average
cross-entropy the model assigns to the ending's tokens, and the lowest-loss ending is the
model's answer. Two numbers come out of that:

  acc       lowest *total* loss wins  - biases towards short endings
  acc_norm  lowest *per-token* loss wins - the number everyone reports

Random guessing scores 25%. GPT-2 124M scores ~31%, and llm.c's 124M trained on
FineWeb-Edu scores ~29.5%, so anything near 30% is the expected result at this scale.

The scoring follows Karpathy's llm.c so the numbers are comparable to that repo.
"""

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F
from datasets import load_dataset

from model import GPT

PROJECT_ROOT = Path(__file__).resolve().parent
# rowanz/hellaswag on GitHub is blocked, so the data comes from the Hugging Face mirror
DATASET = "Rowan/hellaswag"


def resolve_device(name):
    """Turn the 'auto' device setting into 'cuda' or 'cpu'; any other value is returned unchanged."""
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def get_autocast_context(device_type, dtype):
    """Return the mixed-precision context for scoring: bf16 autocast on GPU, a no-op for float32."""
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


def load_examples(data_dir, limit):
    """Return the validation split (10,042 examples), or its first limit examples.

    The download is cached under data_dir rather than the default ~/.cache/huggingface,
    which keeps it off the C: drive.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    split = f"validation[:{limit}]" if limit else "validation"
    return load_dataset(DATASET, split=split, cache_dir=str(data_dir))


def render_example(example, enc):
    """Tokenise one example into (tokens, mask, label).

    tokens is (4, T): row i is the context followed by ending i, right-padded to the
    longest of the four. mask is (4, T) with 1 exactly on the ending tokens, so the
    shared context and the padding are both left out of the score. Endings get a
    leading space because GPT-2 BPE encodes a word differently with and without one.
    """
    ctx_tokens = enc.encode(example["ctx"])
    rows, masks = [], []
    for ending in example["endings"]:
        end_tokens = enc.encode(" " + ending)
        rows.append(ctx_tokens + end_tokens)
        masks.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    max_len = max(len(r) for r in rows)
    tokens = torch.zeros(4, max_len, dtype=torch.long)
    mask = torch.zeros(4, max_len, dtype=torch.long)
    for i, (row, row_mask) in enumerate(zip(rows, masks)):
        tokens[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, : len(row_mask)] = torch.tensor(row_mask, dtype=torch.long)
    return tokens, mask, int(example["label"])


@torch.no_grad()
def score_example(model, tokens, mask, block_size, ctx):
    """Return (argmin total loss, argmin per-token loss) over the four endings."""
    if tokens.size(1) > block_size:
        # keep the tail: the ending is what gets scored, the front of the context is not
        tokens, mask = tokens[:, -block_size:], mask[:, -block_size:]

    with ctx:
        # targets is passed only to make forward return logits at *every* position:
        # model(idx) alone returns the last position only (an inference shortcut, see
        # model.py). The loss it computes here is a batch mean, which cannot be split
        # back into the four per-ending scores, so it is discarded and recomputed below.
        logits, _ = model(tokens, targets=tokens)

    # position i predicts token i+1, so drop the last logit and the first token
    shift_logits = logits[:, :-1, :].float()  # float32: bf16 loses ties between close endings
    shift_tokens = tokens[:, 1:]
    shift_mask = mask[:, 1:]

    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_tokens.reshape(-1),
        reduction="none",
    ).view(shift_tokens.shape)

    losses = losses * shift_mask  # zero out the context and the padding
    sum_loss = losses.sum(dim=1)  # (4,) total loss of each ending
    mean_loss = sum_loss / shift_mask.sum(dim=1)  # (4,) per-token loss of each ending
    return sum_loss.argmin().item(), mean_loss.argmin().item()


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
        f"val loss {checkpoint['val_loss']:.4f}"
    )
    return model


def main(args):
    device = resolve_device(args.device)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    ctx = get_autocast_context(device_type, args.dtype)
    torch.set_float32_matmul_precision("high")  # TF32 matmuls on Ampere and newer GPUs

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    model = load_model(ckpt_path, device)
    block_size = model.config.block_size

    enc = tiktoken.get_encoding("gpt2")
    examples = load_examples(PROJECT_ROOT / args.data_dir, args.limit)
    total = len(examples)
    print(f"scoring {total:,} examples on {device} (random guessing scores 25%)")

    n = n_correct = n_correct_norm = 0
    t0 = time.time()
    for example in examples:
        tokens, mask, label = render_example(example, enc)
        pred, pred_norm = score_example(
            model, tokens.to(device), mask.to(device), block_size, ctx
        )
        n += 1
        n_correct += int(pred == label)
        n_correct_norm += int(pred_norm == label)

        if n % args.progress == 0 or n == total:
            elapsed = time.time() - t0
            eta = (total - n) * elapsed / n
            print(
                f"{n:>6,} / {total:,} | acc {n_correct / n:.4f} | "
                f"acc_norm {n_correct_norm / n:.4f} | {elapsed / 60:.1f} min elapsed, "
                f"~{eta / 60:.0f} min left"
            )

    result = {
        "checkpoint": str(ckpt_path),
        "examples": n,
        "acc": n_correct / n,
        "acc_norm": n_correct_norm / n,
        "minutes": (time.time() - t0) / 60,
    }
    print(f"\nHellaSwag over {n:,} examples: acc {result['acc']:.4f}, acc_norm {result['acc_norm']:.4f}")

    if args.out:
        out_path = PROJECT_ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--ckpt", default="out/best.pt", help="checkpoint to score (default: out/best.pt)")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N examples (0: all)")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', 'cuda:0', ...")
    parser.add_argument("--dtype", default="auto", help="'auto' (bf16 on GPU, fp32 on CPU), 'bfloat16', 'float32'")
    parser.add_argument("--data-dir", default="data/hellaswag", help="where hellaswag_val.jsonl is cached")
    parser.add_argument("--progress", type=int, default=100, help="print progress every N examples")
    parser.add_argument("--out", default="", help="also write the result as JSON to this path")
    main(parser.parse_args())
