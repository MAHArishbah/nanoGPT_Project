"""Check the KV-cache model (model_kv.py) against the original model.py on the real checkpoints."""
import sys, time
from pathlib import Path

import torch
import tiktoken

PROJECT = Path(r"E:\AI_ML\Prac\nanogpt_project")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).parent))
import model as orig          # the user's untouched model.py
import model as kvm        # the copy with the KV cache

torch.set_num_threads(4)
enc = tiktoken.get_encoding("gpt2")
prompt = "<|endoftext|>The theory of evolution by natural selection explains"
ids = torch.tensor([enc.encode(prompt, allowed_special={"<|endoftext|>"})])


def load(mod, ck):
    m = mod.GPT.from_checkpoint(ck)  # strict load_state_dict: fails on any missing/extra key
    return m.eval()


for name in ["gpt-2 arch", "Modern arch"]:
    path = PROJECT / "out" / name / "best.pt"
    ck = torch.load(path, map_location="cpu", weights_only=True)
    print(f"\n=== {name}  ({ck['model_args'].get('norm')}, {ck['model_args'].get('mlp')}, {ck['model_args'].get('pos_emb')}), "
          f"step {ck['iter_num']}, val {ck['val_loss']:.4f}")
    print("checkpoint keys:", sorted(ck.keys()))

    a, b = load(orig, ck), load(kvm, ck)
    print("1. strict load into original AND kv model: OK")

    # 2. training path unchanged: full-sequence logits and loss identical
    x = ids[:, :-1]; y = ids[:, 1:]
    with torch.no_grad():
        la, lossa = a(x, y); lb, lossb = b(x, y)
    print(f"2. training forward: max|dlogits| = {(la - lb).abs().max().item():.2e}, loss {lossa.item():.5f} vs {lossb.item():.5f}")

    # 3. per-step logits: cached decode vs full recompute, over 20 steps
    with torch.no_grad():
        cache = kvm.KVCache(b.config.n_layer)
        seq = ids.clone()
        lc, _ = b(seq, kv_cache=cache)
        worst = 0.0
        for _ in range(20):
            lf, _ = b(seq)  # full recompute, last position
            worst = max(worst, (lc[:, -1] - lf[:, -1]).abs().max().item())
            nxt = lf[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat((seq, nxt), 1)
            lc, _ = b(nxt, kv_cache=cache)
    print(f"3. cached vs full logits over 20 decode steps: max|diff| = {worst:.2e}")

    # 4. greedy generation: identical tokens with and without cache; timing
    n = 200
    with torch.no_grad():
        t0 = time.time(); g0 = b.generate(ids, n, top_k=1, use_cache=False); t_nc = time.time() - t0
        t0 = time.time(); g1 = b.generate(ids, n, top_k=1, use_cache=True); t_c = time.time() - t0
    print(f"4. greedy {n} tokens identical: {torch.equal(g0, g1)} | no cache {t_nc:.1f}s, cache {t_c:.1f}s -> {t_nc / t_c:.1f}x")

    # 5. context overflow: crop to block 32 so 60 new tokens must wrap past the limit
    c0, c1 = load(orig, ck), load(kvm, ck)
    c0.crop_block_size(32); c1.crop_block_size(32)
    with torch.no_grad():
        o0 = c0.generate(ids, 60, top_k=1)                 # original, uncached
        o1 = c1.generate(ids, 60, top_k=1, use_cache=True)  # kv, cache rebuilt when full
    print(f"5. past block_size (32): identical to original = {torch.equal(o0, o1)}")

    # 6. manual (non-flash) attention path with the cache
    for blk in b.transformer.h:
        blk.attn.flash = False
        blk.attn.bias = torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024)
    with torch.no_grad():
        g2 = b.generate(ids, 40, top_k=1, use_cache=True)
    print(f"6. manual-attention path + cache matches flash greedy: {torch.equal(g2, g1[:, :g2.size(1)])}")

    print("sample:", repr(enc.decode([t for t in g1[0].tolist() if t < enc.n_vocab])[:300]))
