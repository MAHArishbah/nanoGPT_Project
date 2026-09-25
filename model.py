"""
GPT-2 style decoder-only transformer.

Holds the building blocks (LayerNorm, CausalSelfAttention, MLP, Block), the
architecture hyper-parameters (GPTConfig) and the full GPT model, including
helpers to build the optimizer, estimate MFU, load OpenAI GPT-2 weights,
rebuild a model from a training checkpoint and generate text.
"""

import inspect
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Architecture hyper-parameters of the GPT model. The defaults give GPT-2 small (124M)."""

    block_size: int = 1024  # maximum context length in tokens
    vocab_size: int = 50304  # GPT-2 vocab size of 50257, padded up to the nearest multiple of 64 for speed
    n_layer: int = 12  # number of transformer blocks
    n_head: int = 12  # attention heads per block
    n_embd: int = 768  # embedding / hidden dimension
    dropout: float = 0.0  # 0.0 is good for pretraining, try 0.1+ for finetuning
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    norm: str = "layernorm"  # 'layernorm' (like GPT-2) or 'rmsnorm' (LLaMA-style: no mean subtraction, no bias)
    mlp: str = "gelu"  # 'gelu' (like GPT-2) or 'swiglu' (LLaMA-style gated MLP)
    pos_emb: str = "learned"  # 'learned' (GPT-2's position table, wpe) or 'rope' (rotary, LLaMA-style)





class LayerNorm(nn.Module):
    """LayerNorm with an optional bias (older nn.LayerNorm versions cannot turn the bias off) now it does , so class is redundant
    nn.LayerNorm can be used as drop in replacement TODO."""

    def __init__(self, ndim, bias):
        """Create a learnable scale of size ndim and, when bias is True, a learnable shift."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        """Normalise the last dimension of input, then apply the scale (and shift)."""
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """RMSNorm: divide x by the root-mean-square of its last dimension, then apply a learnable scale.
    TODO remove class to use nn.RMSNorm dropin"""

    def __init__(self, ndim, eps=1e-5):
        """Create a learnable scale of size ndim (no bias: RMSNorm only rescales, it never shifts)."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        """Normalise the last dimension of x by its RMS, then apply the scale."""
        return F.rms_norm(x,self.weight.shape,self.weight,self.eps)

    
def make_norm(config):
    """Build the normalisation layer that config.norm names."""
    if config.norm == "layernorm":
        return LayerNorm(config.n_embd, bias=config.bias)
    if config.norm == "rmsnorm":
        return RMSNorm(config.n_embd)
    raise ValueError(f"unknown norm {config.norm!r}, use 'layernorm' or 'rmsnorm'")

def rope_cos_sin(head_size, max_len, base=10000.0):
    """cos and sin tables of shape (max_len, head_size) for rotary position embeddings (RoPE)."""
    assert head_size % 2 == 0, "RoPE rotates pairs of dimensions, so the head size must be even"
    # one rotation speed per pair of dimensions: the first pairs turn fast, the last ones slowly
    inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2).float() / head_size))  # (hs/2,)
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)  # (max_len, hs/2): position x speed
    angles = torch.cat((angles, angles), dim=-1)  # (max_len, hs): dims i and i + hs/2 share an angle
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate every (x[..., i], x[..., i + hs/2]) pair of x (B, nh, T, hs) by its position's angle."""
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    # the tables are float32; cast back so q and k keep v's dtype under bf16 autocast
    return (x * cos + rotated * sin).type_as(x)

class KVCache:
    """ k and v tokens seen so far one (k,v) per layer for faster generation
    holds activations not weights, lives for one generate call and is never saved so chkpts are unaffected
    pos is the number of tokens already cached that is the position of the next token"""
    def __init__(self,n_layer):
        """ starting empty no layer has cached anything """
        self.k=[None]* n_layer
        self.v=[None] * n_layer
        self.pos=0

    def update(self,layer,k,v):
        """append this step's k,v to layer's cache and return the full k,v"""
        if self.k[layer] is not None:
            k=torch.cat((self.k[layer],k),dim=2) #k is (B,nh,T,hs)
            v=torch.cat((self.v[layer],v),dim=2) #v is (B,nh,T,hs)
        self.k[layer],self.v[layer]=k,v
        return k,v




class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention: each position attends only to itself and earlier positions."""

    def __init__(self, config):
        """Build the fused q/k/v projection, the output projection and the dropout layers."""
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        # one Linear computes q, k and v for all heads in a single matmul
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # projects the concatenated head outputs back into the residual stream
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularisation
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # flash attention (fused kernel) ships with PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash attention requires PyTorch >= 2.0")
            # lower-triangular mask so a token can never attend to future tokens
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x,rope=None,kv_cache=None,layer=0):
        """Apply causal self-attention to x of shape (B, T, C); the output has the same shape.
          rope is the (cos, sin) pair of RoPE tables for the T positions, or None for learned positions.
          kv cache (inference only ) holds the k/v for earlier tokens, layer is this block's index in it"""
        B, T, C = x.size()  # batch size, sequence length, embedding dimension (n_embd)
        past=kv_cache.pos if kv_cache is not None else 0 #tokens already in cache

        # compute q, k, v for all heads and move the head dimension next to the batch dimension
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        if rope is not None:
        # rotate q and k by their positions, so q . k depends on how far apart the two tokens are
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        if kv_cache is not None:
            assert past == 0 or T==1 , 'with a non empty kv cache feed one token at a time'
            #cache after rope , so cached keys are never rotated twice, attend over past plus new keys
            k,v=kv_cache.update(layer,k,v) #(B,nh,past+ T,hs)


        if self.flash:
            # fused attention kernel with built-in causal masking, is causal aligns its mask to the top left so with 1
            #query and past+1 keys it would let the query see only key0 and a single new token may see every cached token , hence no mask in decode. rows are query and keys are columns
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=(past==0)
            )
        else:
            # manual attention: (B, nh, T, hs) x (B, nh, hs, past + T) -> (B, nh, T, past+T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, past:past +T, :past + T] == 0, float("-inf"))  # type: ignore
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Position-wise feed-forward network: expand 4x, GELU, project back, dropout."""

    def __init__(self, config):
        """Build the expansion layer, the activation, the projection and dropout."""
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU() #costs 8d^2 with 2 matrices
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """Transform every position of x of shape (B, T, C) independently."""
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class SwiGLUMLP(nn.Module):
    """Gated feed-forward network (SwiGLU, as in LLaMA): SiLU(gate) * up, project back, dropout."""

    def __init__(self, config):
        """Build the fused gate/up projection, the output projection and dropout."""
        super().__init__()
        #costs 3dh. equilising , we get 8d/3 , so the three matrices hold as many weights as GELU's two;
        # rounded up to a multiple of 64 for fast matmuls (768 -> 2048 exactly)
        hidden = 64 * math.ceil(8 * config.n_embd / 3 / 64)
        # one Linear computes the gate and the up projection in a single matmul
        self.c_fc = nn.Linear(config.n_embd, 2 * hidden, bias=config.bias)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """Transform every position of x of shape (B, T, C) independently."""
        gate, up = self.c_fc(x).chunk(2, dim=-1)  # each (B, T, hidden)
        x = F.silu(gate) * up
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

def make_mlp(config):
    """Build the feed-forward network that config.mlp names."""
    if config.mlp == "gelu":
        return MLP(config)
    if config.mlp == "swiglu":
        return SwiGLUMLP(config)
    raise ValueError(f"unknown mlp {config.mlp!r}, use 'gelu' or 'swiglu'")




class Block(nn.Module):
    """One pre-norm transformer block: attention then MLP, each added to the residual stream."""

    def __init__(self, config):
        """Build the two LayerNorms, the attention layer and the MLP."""
        super().__init__()
        self.ln_1 = make_norm(config=config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = make_norm(config=config)
        self.mlp = make_mlp(config)

    def forward(self, x,rope=None,kv_cache=None,layer=0):
        """Run x through attention (tokens communicate) then the MLP (each token computes)."""
        x = x + self.attn(self.ln_1(x),rope,kv_cache,layer)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT-2 language model: token + position embeddings, n_layer blocks, final LayerNorm, LM head."""

    def __init__(self, config):
        """Build all layers from config, tie the embedding and LM-head weights and initialise weights."""
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        if config.pos_emb not in ("learned", "rope"):
            raise ValueError(f"unknown pos_emb {config.pos_emb!r}, use 'learned' or 'rope'")

        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
                wpe=nn.Embedding(config.block_size, config.n_embd) if config.pos_emb == "learned" else None,  # position embeddings
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=make_norm(config),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying: the input embedding and the output projection share one matrix
        self.transformer.wte.weight = self.lm_head.weight
        if config.pos_emb == "rope":
        # RoPE tables for every position, shared by all blocks. They follow from the config,
        # so they are rebuilt on load instead of being saved in checkpoints (persistent=False)
            cos, sin = rope_cos_sin(config.n_embd // config.n_head, config.block_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)


        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per the GPT-2 paper.
        # the 2 is there because every block adds to the residual stream twice (attn and mlp)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model.

        With non_embedding=True the position embeddings are subtracted. The token
        embeddings are kept, because they are tied to the LM head and so are real
        parameters of the final layer.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.transformer.wpe is not None:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialise Linear and Embedding weights from N(0, 0.02) and biases to zero, like GPT-2."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None,kv_cache=None):
        """Run the model on token ids idx of shape (B, T).

        With targets (B, T): returns logits for every position and the cross-entropy loss.
        Without targets (inference): returns logits for the last position only, and loss=None.
        kv cache( inference only): idx continues the tokens already cached, and is appended to it
        """
        device = idx.device
        b, t = idx.size()
        start=kv_cache.pos if kv_cache is not None else 0 #position of idx;s first token
        assert start+t <= self.config.block_size, (
            f"cannot forward a sequence of length {start+t}, block size is only {self.config.block_size}"
        )
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        if self.config.pos_emb == "learned":
            pos = torch.arange(start, start+t, dtype=torch.long, device=device)  # (t,)
            x = tok_emb + self.transformer.wpe(pos)  # position added once, at the input
            rope = None
        else:
            x = tok_emb  # RoPE adds no position here; every attention layer rotates q and k instead
            rope = (self.rope_cos[start:start+t], self.rope_sin[start:start+t])  # each (t, head_size)
        x = self.transformer.drop(x)
        for i,block in enumerate( self.transformer.h):
            x = block(x, rope,kv_cache,i)
        if kv_cache is not None:
            kv_cache.pos += t
        x = self.transformer.ln_f(x)

        if targets is not None:
            # training: logits at every position and the loss against the targets
            logits = self.lm_head(x)  # (b, t, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimisation: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss

    def crop_block_size(self, block_size):
        """Shrink the context length, e.g. to use pretrained GPT-2 (1024) with a smaller block size."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if self.transformer.wpe is not None:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """Build a model and load OpenAI GPT-2 weights from Hugging Face (requires `transformers`)."""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}
        assert all(k == "dropout" for k in override_args), "only dropout can be overridden"
        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_type}")
        # n_layer, n_head and n_embd are determined by model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True,norm=layernorm,mlp=gelu,pos_embedding=learned")
        config_args["vocab_size"] = 50257  # always 50257 for GPT-2 checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT-2 checkpoints
        config_args["bias"] = True  # always True for GPT-2 checkpoints
        config_args["norm"] = "layernorm"
        config_args["mlp"] = "gelu"  # GPT-2 uses a GELU MLP
        config_args["pos_emb"] = "learned"  # GPT-2 learns a position table (wpe)


        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]

        # a from-scratch model whose weights will be overwritten
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]  # the causal mask is a buffer, not a param

        # the Hugging Face model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith(".attn.masked_bias")]  # buffer, skip
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]  # the mask (buffer), skip
        # OpenAI checkpoints use a "Conv1D" module; we use a plain Linear, so these weights must be transposed
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    @classmethod
    def from_checkpoint(cls, checkpoint):
        """Rebuild a model from a checkpoint dict holding 'model_args' and 'model' (the state dict)."""
        model = cls(GPTConfig(**checkpoint["model_args"]))
        # weights saved from a torch.compile'd model carry an '_orig_mod.' prefix on every key
        unwanted_prefix = "_orig_mod."
        state_dict = {
            (k[len(unwanted_prefix):] if k.startswith(unwanted_prefix) else k): v
            for k, v in checkpoint["model"].items()
        }
        model.load_state_dict(state_dict)
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Create an AdamW optimizer that applies weight decay only to 2D+ tensors.

        Matmul weights and embeddings (2D) are decayed; biases and LayerNorm
        parameters (1D) are not. The fused CUDA kernel is used when available.
        """
        # all parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # fused AdamW does the whole update in one kernel, much faster on GPU
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model FLOPs utilisation (MFU) as a fraction of A100 bfloat16 peak FLOPS.

        fwdbwd_per_iter is the number of sequences processed per optimizer step and dt the
        step time in seconds. FLOP count follows the PaLM paper, Appendix B.
        """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak is 312 TFLOPS
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,use_cache=True):
        """Extend the token ids idx of shape (B, T) by max_new_tokens sampled tokens.

        Each new token is sampled from the softmax of the last-position logits
        (divided by temperature, optionally restricted to the top_k most likely
        tokens) and fed back in. Call model.eval() first.
        with use_cache the prompt is run once (prefill) and every later step feeds only the newest token 
        reusing the cachd k/v of the earlier ones.

        """
        block_size=self.config.block_size
        kv_cache=None
        for _ in range(max_new_tokens):
            if not use_cache:
                # if the context grows too long, crop it to the last block_size tokens
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
            elif kv_cache is None or kv_cache.pos==block_size:
                #prefill:first step, or the cache is full . cached keys carry their
                #absolute positions, so the oldest token cannot just be dropped;rebuild from the last
                #block_size tokens instead exactly in uncached case.
                kv_cache=KVCache(self.config.n_layer)
                logits,_=self(idx[:,-block_size:],kv_cache=kv_cache)
            else:
                #decode: only the newset toke , its k/v are appended to the cache
                logits,_=self(idx[:,-1:],kv_cache=kv_cache)
            # take the logits at the final step and scale by the temperature
            logits = logits[:, -1, :] / temperature
            # optionally keep only the top_k most likely tokens
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
