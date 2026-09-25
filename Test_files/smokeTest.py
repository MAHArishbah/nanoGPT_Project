"""
Smoke test on a tiny model: every architecture variant builds and runs, and train.py and
sample.py work end to end (train, resume, resume refused on a changed setting, sample). by claude code

$ python smokeTest.py

"""

import itertools
import math
import shutil
from pathlib import Path

import torch

import sample
import train
from config import SampleConfig, TrainConfig
from model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parent
TINY_MODEL = dict(n_layer=2, n_head=2, n_embd=64, block_size=64)
LLAMA_STYLE = dict(norm="rmsnorm", mlp="swiglu", pos_emb="rope", bias=False)


def tiny_train_config(**overrides):
    """A TrainConfig for a run that takes seconds: tiny model, 10% of the text, CPU-friendly."""
    settings = dict(
        **TINY_MODEL,
        data_dir="data",
        dataset="shakespeare_input.txt",
        wandb_log=False,
        compile=False,
        dtype="float32",
        val_fraction=0.9,  # train on only the first 10% of the text, so an epoch takes seconds
        batch_size=16,
        total_batch_size=16 * 64 * 4,  # 4 gradient-accumulation steps per optimizer step
        eval_iters=5,
        log_interval=10,
        max_epochs=1,
    )
    settings.update(overrides)
    return TrainConfig(**settings)


def header(title):
    print(f"\n{'=' * 20} {title}")


# ---------------------------------------------------------------- 1. every variant builds and runs
header("1. every combination of norm x mlp x pos_emb")
torch.manual_seed(0)
idx = torch.randint(0, 50304, (2, 32))
targets = torch.randint(0, 50304, (2, 32))
print(f"expected starting loss ~ ln(50304) = {math.log(50304):.2f}")
for norm, mlp, pos_emb in itertools.product(("layernorm", "rmsnorm"), ("gelu", "swiglu"), ("learned", "rope")):
    args = dict(**TINY_MODEL, norm=norm, mlp=mlp, pos_emb=pos_emb)
    model = GPT(GPTConfig(**args))
    _, loss = model(idx, targets)
    loss.backward()
    keys = model.state_dict().keys()
    block = model.transformer.h[0]
    # save and reload, then compare outputs (eval mode, so dropout can't differ)
    reloaded = GPT.from_checkpoint({"model_args": args, "model": model.state_dict()})
    model.eval(), reloaded.eval()
    same = torch.allclose(model(idx)[0], reloaded(idx)[0])
    print(
        f"{norm:9} {mlp:6} {pos_emb:7} | loss {loss.item():.2f} | norm {type(block.ln_1).__name__:9} "
        f"| mlp {type(block.mlp).__name__:9} c_fc {tuple(block.mlp.c_fc.weight.shape)} "
        f"| wpe {'yes' if any('.wpe.' in k for k in keys) else 'no '} | reload identical: {same}"
    )

old_args = dict(TINY_MODEL)  # no norm/mlp/pos_emb keys, like a checkpoint saved before those settings existed
old = GPT.from_checkpoint({"model_args": old_args, "model": GPT(GPTConfig(**old_args)).state_dict()})
print(f"checkpoint without the new keys loads as: {old.config.norm}, {old.config.mlp}, {old.config.pos_emb}")

for d in ("out-test-gpt2", "out-test-llama"):
    shutil.rmtree(ROOT / d, ignore_errors=True)

# ---------------------------------------------------------------- 2. train both styles
header("2a. train, GPT-2 style (layernorm, gelu, learned)")
train.main(tiny_train_config(out_dir="out-test-gpt2"))

header("2b. train, LLaMA style (rmsnorm, swiglu, rope, no bias)")
train.main(tiny_train_config(out_dir="out-test-llama", **LLAMA_STYLE))

# ---------------------------------------------------------------- 3. resume
header("3. resume the LLaMA-style run for a second epoch")
train.main(tiny_train_config(out_dir="out-test-llama", **LLAMA_STYLE, init_from="resume", max_epochs=2))
ckpt = torch.load(ROOT / "out-test-llama" / "ckpt.pt", map_location="cpu", weights_only=True)
print("model_args saved in ckpt.pt:", ckpt["model_args"])

# ---------------------------------------------------------------- 4. a changed architecture must be refused
header("4. resume with pos_emb changed to 'learned' (must be refused)")
try:
    changed = {**LLAMA_STYLE, "pos_emb": "learned"}
    train.main(tiny_train_config(out_dir="out-test-llama", **changed, init_from="resume", max_epochs=3))
    print("FAIL: the resume was not refused")
except ValueError as e:
    print(f"refused as expected:\n{e}")

# ---------------------------------------------------------------- 5. sample
header("5. sample from the LLaMA-style best.pt")
sample.main(SampleConfig(out_dir="out-test-llama", num_samples=2, max_new_tokens=40, dtype="float32", start="ROMEO:"))
header("5. sample from the GPT2-style best.pt")
sample.main(SampleConfig(out_dir="out-test-gpt2", num_samples=2, max_new_tokens=40, dtype="float32", start="ROMEO:"))

print("\nsmoke test finished; delete out-test-gpt2/ and out-test-llama/ when you're done looking")
