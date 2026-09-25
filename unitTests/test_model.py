"""Unit tests for model.py: the building blocks, the GPT model, the KV cache and generation."""

import math
from dataclasses import asdict

import pytest
import torch
from torch.nn import functional as F

from model import (
    GPT,
    MLP,
    Block,
    CausalSelfAttention,
    GPTConfig,
    KVCache,
    LayerNorm,
    RMSNorm,
    SwiGLUMLP,
    apply_rope,
    make_mlp,
    make_norm,
    rope_cos_sin,
)

VOCAB = 64

# the two architectures train.py is used with
ARCHS = {
    "gpt2": dict(norm="layernorm", mlp="gelu", pos_emb="learned"),
    "modern": dict(norm="rmsnorm", mlp="swiglu", pos_emb="rope"),
}


def tiny_config(**overrides):
    """A GPTConfig small enough to run on a CPU in milliseconds."""
    args = dict(block_size=16, vocab_size=VOCAB, n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=True)
    args.update(overrides)
    return GPTConfig(**args)


def use_manual_attention(module):
    """Switch every attention layer inside module to the non-flash path (the PyTorch < 2.0 one)."""
    for m in module.modules():
        if isinstance(m, CausalSelfAttention):
            bs = 16
            m.flash = False
            m.register_buffer("bias", torch.tril(torch.ones(bs, bs)).view(1, 1, bs, bs))


@pytest.fixture(params=list(ARCHS))
def arch(request):
    """Run the test once for the GPT-2 architecture and once for the modern one."""
    return ARCHS[request.param]


# ---------------------------------------------------------------- config


def test_default_config_is_gpt2_small():
    """The GPTConfig defaults describe GPT-2 small (124M) with a vocab padded to a multiple of 64."""
    c = GPTConfig()
    assert (c.n_layer, c.n_head, c.n_embd, c.block_size) == (12, 12, 768, 1024)
    assert c.vocab_size >= 50257 and c.vocab_size % 64 == 0
    assert (c.norm, c.mlp, c.pos_emb) == ("layernorm", "gelu", "learned")


# ---------------------------------------------------------------- normalisation


@pytest.mark.parametrize("bias", [True, False])
def test_layernorm_matches_formula(bias):
    """LayerNorm subtracts the mean, divides by the std, then scales (and shifts when bias is on)."""
    ln = LayerNorm(8, bias=bias)
    with torch.no_grad():
        ln.weight.normal_()
        if bias:
            ln.bias.normal_()
    x = torch.randn(3, 5, 8)
    expected = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-5)
    expected = expected * ln.weight + (ln.bias if bias else 0)
    torch.testing.assert_close(ln(x), expected)


def test_layernorm_without_bias_has_only_a_scale():
    """With bias=False the only learnable parameter is the scale."""
    ln = LayerNorm(8, bias=False)
    assert ln.bias is None
    assert [name for name, _ in ln.named_parameters()] == ["weight"]


def test_rmsnorm_matches_formula():
    """RMSNorm divides by the root-mean-square and scales, with no mean subtraction and no shift."""
    norm = RMSNorm(8)
    with torch.no_grad():
        norm.weight.normal_()
    x = torch.randn(3, 5, 8)
    expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * norm.weight
    torch.testing.assert_close(norm(x), expected)
    assert [name for name, _ in norm.named_parameters()] == ["weight"]  # no shift


def test_make_norm_picks_the_configured_layer():
    """make_norm builds the layer config.norm names and rejects unknown names."""
    assert isinstance(make_norm(tiny_config(norm="layernorm")), LayerNorm)
    assert isinstance(make_norm(tiny_config(norm="rmsnorm")), RMSNorm)
    with pytest.raises(ValueError, match="unknown norm"):
        make_norm(tiny_config(norm="batchnorm"))


# ---------------------------------------------------------------- RoPE


def test_rope_tables_shape_and_values():
    """The cos/sin tables have one row per position, start unrotated and turn the first pair 1 rad per step."""
    cos, sin = rope_cos_sin(8, 10)
    assert cos.shape == sin.shape == (10, 8)
    # position 0 is not rotated
    torch.testing.assert_close(cos[0], torch.ones(8))
    torch.testing.assert_close(sin[0], torch.zeros(8))
    torch.testing.assert_close(cos**2 + sin**2, torch.ones(10, 8))
    # dims i and i + hs/2 share an angle
    torch.testing.assert_close(cos[:, :4], cos[:, 4:])
    # the first pair turns fastest: one radian per position
    torch.testing.assert_close(sin[:, 0], torch.sin(torch.arange(10.0)))


def test_rope_needs_an_even_head_size():
    """RoPE rotates pairs of dimensions, so an odd head size is rejected."""
    with pytest.raises(AssertionError):
        rope_cos_sin(7, 10)


def test_apply_rope_is_a_rotation():
    """apply_rope keeps the shape and the vector lengths, and leaves position 0 unchanged."""
    cos, sin = rope_cos_sin(8, 6)
    x = torch.randn(2, 3, 6, 8)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))  # rotations keep lengths
    torch.testing.assert_close(y[..., 0, :], x[..., 0, :])  # position 0 is unchanged


def test_apply_rope_scores_depend_only_on_relative_position():
    """After RoPE, q . k depends on how far apart the two tokens are, not on where they sit."""
    T, hs = 10, 8
    cos, sin = rope_cos_sin(hs, T)
    q = torch.randn(hs).expand(1, 1, T, hs)  # the same query at every position
    k = torch.randn(hs).expand(1, 1, T, hs)  # the same key at every position
    scores = apply_rope(q, cos, sin)[0, 0] @ apply_rope(k, cos, sin)[0, 0].T  # (T, T)
    # same distance, same score, wherever the pair sits
    torch.testing.assert_close(scores[3, 1], scores[7, 5])
    torch.testing.assert_close(scores[2, 0], scores[9, 7])
    torch.testing.assert_close(scores[4, 4], scores[0, 0])


def test_apply_rope_keeps_the_input_dtype():
    """float32 tables applied to a bf16 input still give bf16, as needed under autocast."""
    cos, sin = rope_cos_sin(8, 4)  # float32 tables
    x = torch.randn(1, 1, 4, 8, dtype=torch.bfloat16)
    assert apply_rope(x, cos, sin).dtype == torch.bfloat16


# ---------------------------------------------------------------- KV cache


def test_kv_cache_starts_empty():
    """A new cache holds nothing for any layer and starts at position 0."""
    cache = KVCache(3)
    assert cache.k == [None] * 3 and cache.v == [None] * 3 and cache.pos == 0


def test_kv_cache_appends_along_the_time_dimension():
    """update appends new k/v after the cached ones, per layer, without moving pos."""
    cache = KVCache(2)
    k1, v1 = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)
    k, v = cache.update(0, k1, v1)
    assert torch.equal(k, k1) and torch.equal(v, v1)

    k2, v2 = torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4)
    k, v = cache.update(0, k2, v2)
    assert k.shape == v.shape == (1, 2, 4, 4)
    assert torch.equal(k[:, :, :3], k1) and torch.equal(k[:, :, 3:], k2)
    assert torch.equal(v[:, :, :3], v1) and torch.equal(v[:, :, 3:], v2)
    assert cache.k[1] is None  # layers are cached separately
    assert cache.pos == 0  # GPT.forward moves pos, not update


# ---------------------------------------------------------------- attention


def test_attention_needs_n_embd_divisible_by_n_head():
    """The embedding must split evenly into heads."""
    with pytest.raises(AssertionError):
        CausalSelfAttention(tiny_config(n_embd=30, n_head=4))


@pytest.mark.parametrize("manual", [False, True])
def test_attention_is_causal(manual):
    """Changing later tokens never changes the outputs at earlier positions, on both attention paths."""
    attn = CausalSelfAttention(tiny_config()).eval()
    if manual:
        use_manual_attention(attn)
    x = torch.randn(2, 8, 32)
    changed = x.clone()
    changed[:, 5:] = torch.randn(2, 3, 32)  # change only the future
    with torch.no_grad():
        y, y_changed = attn(x), attn(changed)
    assert y.shape == x.shape
    torch.testing.assert_close(y[:, :5], y_changed[:, :5])  # the past cannot see the change
    assert not torch.allclose(y[:, 5:], y_changed[:, 5:])


@pytest.mark.parametrize("rope", [False, True])
def test_flash_and_manual_attention_agree(rope):
    """The fused scaled_dot_product_attention kernel and the manual masked softmax give the same output."""
    attn = CausalSelfAttention(tiny_config()).eval()
    x = torch.randn(2, 8, 32)
    tables = rope_cos_sin(16, 8) if rope else None
    with torch.no_grad():
        y_flash = attn(x, tables)
        use_manual_attention(attn)
        y_manual = attn(x, tables)
    torch.testing.assert_close(y_flash, y_manual, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------- MLPs and blocks


@pytest.mark.parametrize("mlp_cls", [MLP, SwiGLUMLP])
def test_mlp_is_position_wise(mlp_cls):
    """Both MLPs keep the shape and transform each position on its own."""
    mlp = mlp_cls(tiny_config()).eval()
    x = torch.randn(2, 6, 32)
    changed = x.clone()
    changed[:, 3] = torch.randn(2, 32)
    y, y_changed = mlp(x), mlp(changed)
    assert y.shape == x.shape
    keep = [0, 1, 2, 4, 5]
    torch.testing.assert_close(y[:, keep], y_changed[:, keep])


def test_swiglu_hidden_size():
    """SwiGLU's hidden size is 8/3 * n_embd rounded up to a multiple of 64, with gate and up fused."""
    mlp = SwiGLUMLP(tiny_config(n_embd=768, n_head=12))
    assert mlp.c_proj.in_features == 2048  # 8 * 768 / 3 = 2048 exactly
    assert mlp.c_fc.out_features == 2 * 2048  # gate and up fused
    # rounded up to a multiple of 64
    assert SwiGLUMLP(tiny_config(n_embd=32)).c_proj.in_features == 128


def test_swiglu_has_as_many_weights_as_gelu_mlp():
    """At n_embd=768 the three SwiGLU matrices hold exactly as many weights as GELU's two."""
    cfg = tiny_config(n_embd=768, n_head=12, bias=False)
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(SwiGLUMLP(cfg)) == count(MLP(cfg))


def test_make_mlp_picks_the_configured_layer():
    """make_mlp builds the MLP config.mlp names and rejects unknown names."""
    assert isinstance(make_mlp(tiny_config(mlp="gelu")), MLP)
    assert isinstance(make_mlp(tiny_config(mlp="swiglu")), SwiGLUMLP)
    with pytest.raises(ValueError, match="unknown mlp"):
        make_mlp(tiny_config(mlp="relu"))


def test_block_keeps_the_shape(arch):
    """A transformer block maps (B, T, C) to (B, T, C) for both architectures."""
    block = Block(tiny_config(**arch))
    x = torch.randn(2, 8, 32)
    rope = rope_cos_sin(16, 8) if arch["pos_emb"] == "rope" else None
    assert block(x, rope).shape == x.shape


# ---------------------------------------------------------------- GPT: construction


def test_unknown_pos_emb_is_rejected():
    """GPT only accepts 'learned' or 'rope' positions."""
    with pytest.raises(ValueError, match="unknown pos_emb"):
        GPT(tiny_config(pos_emb="sinusoidal"))


def test_embedding_and_lm_head_share_weights(arch):
    """The token embedding and the LM head are one tied matrix."""
    model = GPT(tiny_config(**arch))
    assert model.lm_head.weight is model.transformer.wte.weight


def test_learned_positions_have_a_table_and_rope_does_not():
    """Learned positions use a wpe table; RoPE has none and keeps its cos/sin tables out of the state dict."""
    learned = GPT(tiny_config(**ARCHS["gpt2"]))
    assert learned.transformer.wpe.weight.shape == (16, 32)
    assert not hasattr(learned, "rope_cos")

    rope = GPT(tiny_config(**ARCHS["modern"]))
    assert rope.transformer.wpe is None
    assert rope.rope_cos.shape == rope.rope_sin.shape == (16, 16)  # (block_size, head_size)
    # the tables follow from the config, so checkpoints do not store them
    assert not any("rope" in k for k in rope.state_dict())


def test_get_num_params():
    """get_num_params leaves out the position table only; RoPE models have none to leave out."""
    learned = GPT(tiny_config(**ARCHS["gpt2"]))
    total = sum(p.numel() for p in learned.parameters())
    assert learned.get_num_params(non_embedding=False) == total
    assert learned.get_num_params() == total - learned.transformer.wpe.weight.numel()

    rope = GPT(tiny_config(**ARCHS["modern"]))
    assert rope.get_num_params() == rope.get_num_params(non_embedding=False)


def test_weight_init():
    """Weights start at std 0.02, residual projections at 0.02 / sqrt(2 * n_layer), biases at zero."""
    model = GPT(tiny_config(n_embd=64, n_layer=2))
    block = model.transformer.h[0]
    assert block.attn.c_attn.weight.std().item() == pytest.approx(0.02, rel=0.1)
    # residual projections are scaled down by sqrt(2 * n_layer)
    assert block.attn.c_proj.weight.std().item() == pytest.approx(0.02 / math.sqrt(4), rel=0.1)
    assert block.mlp.c_proj.weight.std().item() == pytest.approx(0.02 / math.sqrt(4), rel=0.1)
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.bias is not None:
            assert torch.count_nonzero(m.bias) == 0


# ---------------------------------------------------------------- GPT: forward


def test_forward_shapes(arch):
    """With targets: logits for every position and a scalar loss. Without: logits for the last position only."""
    model = GPT(tiny_config(**arch))
    idx = torch.randint(0, VOCAB, (2, 10))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 10, VOCAB)
    assert loss.shape == ()

    logits_last, no_loss = model(idx)
    assert logits_last.shape == (2, 1, VOCAB) and no_loss is None
    torch.testing.assert_close(logits_last[:, 0], logits[:, -1])


def test_initial_loss_is_close_to_uniform(arch):
    """A freshly initialised model predicts close to uniformly, so its loss is about ln(vocab_size)."""
    model = GPT(tiny_config(**arch))
    idx = torch.randint(0, VOCAB, (4, 16))
    _, loss = model(idx, torch.randint(0, VOCAB, (4, 16)))
    assert loss.item() == pytest.approx(math.log(VOCAB), abs=0.1)


def test_targets_of_minus_one_are_ignored():
    """Positions whose target is -1 do not count towards the loss."""
    model = GPT(tiny_config())
    idx = torch.randint(0, VOCAB, (2, 8))
    targets = torch.randint(0, VOCAB, (2, 8))
    targets[:, :4] = -1
    logits, loss = model(idx, targets)
    expected = F.cross_entropy(logits[:, 4:].reshape(-1, VOCAB), targets[:, 4:].reshape(-1))
    torch.testing.assert_close(loss, expected)


def test_sequences_longer_than_block_size_are_rejected():
    """The model cannot run on more tokens than block_size."""
    model = GPT(tiny_config())
    with pytest.raises(AssertionError, match="block size"):
        model(torch.zeros(1, 17, dtype=torch.long))


def test_model_is_causal(arch):
    """Changing later tokens leaves the logits at earlier positions unchanged."""
    model = GPT(tiny_config(**arch)).eval()
    idx = torch.randint(0, VOCAB, (1, 12))
    changed = idx.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % VOCAB
    with torch.no_grad():
        logits, _ = model(idx, idx)
        logits_changed, _ = model(changed, changed)
    torch.testing.assert_close(logits[:, :8], logits_changed[:, :8])


def test_dropout_only_in_train_mode():
    """Dropout makes repeated forwards differ in train mode and is switched off in eval mode."""
    model = GPT(tiny_config(dropout=0.5))
    idx = torch.randint(0, VOCAB, (2, 8))
    model.train()
    assert not torch.allclose(model(idx, idx)[0], model(idx, idx)[0])
    model.eval()
    torch.testing.assert_close(model(idx, idx)[0], model(idx, idx)[0])


def test_model_can_overfit_one_batch(arch):
    """Gradients flow end to end: 200 AdamW steps on one batch cut its loss by more than half."""
    model = GPT(tiny_config(**arch))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, VOCAB, (2, 8))
    y = torch.randint(0, VOCAB, (2, 8))
    _, first = model(x, y)
    for _ in range(200):
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.5 * first.item()


# ---------------------------------------------------------------- KV cache inside the model


@pytest.mark.parametrize("manual", [False, True])
def test_cached_forward_matches_full_forward(arch, manual):
    """Prefill plus one-token decode steps give the same logits as recomputing the whole prefix."""
    model = GPT(tiny_config(**arch)).eval()
    if manual:
        use_manual_attention(model)
    idx = torch.randint(0, VOCAB, (2, 12))
    cache = KVCache(model.config.n_layer)
    with torch.no_grad():
        logits, _ = model(idx[:, :5], kv_cache=cache)  # prefill
        assert cache.pos == 5
        for t in range(5, 12):
            full, _ = model(idx[:, :t])
            torch.testing.assert_close(logits, full, atol=1e-5, rtol=1e-4)
            logits, _ = model(idx[:, t : t + 1], kv_cache=cache)  # decode one token
            assert cache.pos == t + 1
        full, _ = model(idx)
        torch.testing.assert_close(logits, full, atol=1e-5, rtol=1e-4)


def test_cache_takes_one_token_at_a_time_after_prefill():
    """Once the cache holds tokens, feeding several new tokens at once is rejected."""
    model = GPT(tiny_config()).eval()
    cache = KVCache(model.config.n_layer)
    with torch.no_grad():
        model(torch.randint(0, VOCAB, (1, 4)), kv_cache=cache)
        with pytest.raises(AssertionError, match="one token at a time"):
            model(torch.randint(0, VOCAB, (1, 2)), kv_cache=cache)


def test_cache_cannot_grow_past_block_size():
    """A full cache (block_size tokens) cannot take another token."""
    model = GPT(tiny_config()).eval()
    cache = KVCache(model.config.n_layer)
    with torch.no_grad():
        model(torch.randint(0, VOCAB, (1, 16)), kv_cache=cache)
        with pytest.raises(AssertionError, match="block size"):
            model(torch.randint(0, VOCAB, (1, 1)), kv_cache=cache)


# ---------------------------------------------------------------- generation


@pytest.mark.parametrize("use_cache", [False, True])
def test_generate_extends_the_prompt(use_cache):
    """generate keeps the prompt and appends max_new_tokens valid token ids to every row."""
    model = GPT(tiny_config()).eval()
    prompt = torch.randint(0, VOCAB, (3, 4))
    out = model.generate(prompt, 6, use_cache=use_cache)
    assert out.shape == (3, 10)
    assert torch.equal(out[:, :4], prompt)
    assert out.min() >= 0 and out.max() < VOCAB


def greedy(model, idx, n):
    """Reference greedy decoding: full recompute, argmax, crop to block_size."""
    bs = model.config.block_size
    for _ in range(n):
        logits, _ = model(idx[:, -bs:])
        idx = torch.cat((idx, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
    return idx


@pytest.mark.parametrize("manual", [False, True])
def test_cached_and_uncached_greedy_generation_agree(arch, manual):
    """With top_k=1, generate gives the reference greedy tokens with and without the cache, past block_size too."""
    model = GPT(tiny_config(**arch)).eval()
    if manual:
        use_manual_attention(model)
    prompt = torch.randint(0, VOCAB, (2, 5))
    # 30 new tokens run well past block_size (16), so the cache is rebuilt along the way
    with torch.no_grad():
        expected = greedy(model, prompt, 30)
    torch.testing.assert_close(model.generate(prompt, 30, top_k=1, use_cache=False), expected)
    torch.testing.assert_close(model.generate(prompt, 30, top_k=1, use_cache=True), expected)


def test_top_k_only_samples_from_the_k_most_likely_tokens():
    """Even at a high temperature, top_k=3 samples only among the 3 highest logits."""
    model = GPT(tiny_config()).eval()
    prompt = torch.randint(0, VOCAB, (1, 5))
    with torch.no_grad():
        logits, _ = model(prompt)
    allowed = set(logits[0, -1].topk(3).indices.tolist())
    # a high temperature flattens the distribution, so without top_k many tokens would appear
    out = model.generate(prompt.expand(500, -1), 1, temperature=5.0, top_k=3)
    sampled = set(out[:, -1].tolist())
    assert sampled <= allowed
    assert len(sampled) > 1


def test_generate_is_reproducible_with_a_seed():
    """The same seed gives the same sampled tokens."""
    model = GPT(tiny_config()).eval()
    prompt = torch.randint(0, VOCAB, (1, 4))
    torch.manual_seed(42)
    a = model.generate(prompt, 10, temperature=1.0)
    torch.manual_seed(42)
    b = model.generate(prompt, 10, temperature=1.0)
    assert torch.equal(a, b)


# ---------------------------------------------------------------- helpers


def test_crop_block_size():
    """crop_block_size shrinks wpe and the causal mask, keeps short-sequence outputs, and cannot grow."""
    model = GPT(tiny_config(**ARCHS["gpt2"])).eval()
    use_manual_attention(model)
    idx = torch.randint(0, VOCAB, (1, 6))
    with torch.no_grad():
        before, _ = model(idx, idx)
    model.crop_block_size(8)
    assert model.config.block_size == 8
    assert model.transformer.wpe.weight.shape == (8, 32)
    assert model.transformer.h[0].attn.bias.shape == (1, 1, 8, 8)
    with torch.no_grad():
        after, _ = model(idx, idx)
    torch.testing.assert_close(before, after)  # the first positions are kept as they were
    with pytest.raises(AssertionError):
        model(torch.zeros(1, 9, dtype=torch.long))
    with pytest.raises(AssertionError):
        model.crop_block_size(12)  # can only shrink


def test_from_checkpoint_strips_the_compile_prefix(arch):
    """A checkpoint saved from a torch.compile'd model ('_orig_mod.' keys) loads into an identical model."""
    cfg = tiny_config(**arch)
    src = GPT(cfg).eval()
    checkpoint = {
        "model_args": asdict(cfg),
        "model": {"_orig_mod." + k: v for k, v in src.state_dict().items()},
    }
    dst = GPT.from_checkpoint(checkpoint).eval()
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v)
    idx = torch.randint(0, VOCAB, (1, 8))
    with torch.no_grad():
        torch.testing.assert_close(src(idx, idx)[0], dst(idx, idx)[0])


def test_configure_optimizers_decays_only_matrices():
    """AdamW decays 2D+ tensors only, holds every parameter exactly once, and is not fused on CPU."""
    model = GPT(tiny_config())
    opt = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, betas=(0.9, 0.95), device_type="cpu")
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])
    # every parameter exactly once (the tied embedding is not counted twice)
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(model.parameters()))
    assert decay["lr"] == 3e-4 and decay["betas"] == (0.9, 0.95)
    assert not opt.defaults.get("fused")  # fused AdamW is for CUDA only


def test_estimate_mfu():
    """estimate_mfu follows the PaLM FLOP formula and scales inversely with the step time."""
    model = GPT(tiny_config())
    cfg = model.config
    N = model.get_num_params()
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
    expected = (6 * N + 12 * L * H * Q * T) * T * 4 / 0.5 / 312e12
    assert model.estimate_mfu(4, 0.5) == pytest.approx(expected)
    assert model.estimate_mfu(4, 0.25) == pytest.approx(2 * expected)  # twice as fast, twice the MFU
