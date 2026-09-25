"""
Train a GPT model on a text dataset, evaluating and checkpointing every eval_interval
optimizer steps and at the end of every epoch. A resumed run continues from the exact step
of its checkpoint, also in the middle of an epoch.

Single GPU or CPU:
$ python train.py

Several GPUs on one machine (DDP), e.g. 4:
$ torchrun --standalone --nproc_per_node=4 train.py

Several machines: run torchrun on each with --nnodes, --node_rank, --master_addr and
--master_port (prefix NCCL_IB_DISABLE=1 if the cluster has no Infiniband).

Every setting lives in config/train_config.py.
"""

import math
import os
import time
import dataclasses
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import TrainConfig
from dataset import build_dataloaders
from model import GPT, GPTConfig

PROJECT_ROOT = Path(__file__).resolve().parent

# config keys that define the model architecture (the fields of GPTConfig)
# MODEL_ARG_KEYS = ("n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size", "dropout","norm","mlp","pos_emb")
MODEL_ARG_KEYS = tuple(f.name for f in dataclasses.fields(GPTConfig))
# config keys that may change between a run and its resume,fields only affecting I/O, logging,
# h/w , how long training runs. Every other key must match.
RESUME_MAY_DIFFER = {
    "out_dir", "data_dir", "log_interval", "eval_interval", "eval_iters", "eval_only",
    "wandb_log", "wandb_project", "wandb_run_name", "max_epochs",
    "num_workers", "backend", "device", "dtype", "compile",
}


def resolve_device(name):
    """Turn the 'auto' device setting into 'cuda' or 'cpu' any other value is returned unchanged."""
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def setup_distributed(cfg):
    """Initialise DDP when launched by torchrun, and pick this process's device.Mehtod called once in each process ( as each process is on one GPU), 
    so once per GPU.

    Returns (ddp, rank, local_rank, world_size, device). Without torchrun there is a
    single process with rank 0 and world size 1.
    """
    device = resolve_device(cfg.device)
    ddp = int(os.environ.get("RANK", -1)) != -1  # torchrun --nproc_per_node=4 train.py sets RANK,LOCAL_RANK,WORLD_SIZE for every process it starts
    if not ddp:
        return False, 0, 0, 1, device
    dist.init_process_group(backend=cfg.backend)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if device.startswith("cuda"):
        # one GPU per process, picked by the process's rank on this machine
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    return True, rank, local_rank, world_size, device


def cleanup_distributed(ddp):
    """Wait until every DDP process is done, then shut the process group down."""
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


def get_autocast_context(device_type, dtype):
    """Return the mixed-precision context used around forward passes.

    bfloat16 has the same exponent range as float32, so unlike float16 it needs no
    GradScaler. GPUs without bfloat16 support fall back to plain float32.
    """
    if dtype == "bfloat16" and device_type == "cuda" and not torch.cuda.is_bf16_supported():
        print("WARNING: this GPU does not support bfloat16, falling back to float32")
        dtype = "float32"
    if dtype == "float32":
        return nullcontext()
    if dtype == "bfloat16":
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    raise ValueError(f"unsupported dtype {dtype!r}, use 'bfloat16' or 'float32'")


def get_lr(it, cfg, warmup_iters, lr_decay_iters):
    """Learning rate for optimizer step it: linear warmup, then cosine decay down to min_lr."""
    # linear warmup
    if it < warmup_iters:
        return cfg.learning_rate * (it + 1) / (warmup_iters + 1)
    # past the decay window, hold at the minimum
    if it > lr_decay_iters:
        return cfg.min_lr
    # cosine decay from learning_rate down to min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # goes from 1 to 0
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# @torch.no_grad()
# def estimate_loss(model, loaders, eval_iters, ctx, device):
#     """Average the loss over up to eval_iters batches of each split, with dropout off.

#     loaders maps a split name ('train', 'val') to its DataLoader. Returns {split: mean loss}.
#     """
#     out = {}
#     model.eval()
#     for split, loader in loaders.items():
#         n_batches = min(eval_iters, len(loader))
#         losses = torch.zeros(n_batches)
#         for k, (X, Y) in zip(range(n_batches), loader):
#             X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
#             with ctx:
#                 _, loss = model(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean().item()
#     model.train()
#     return out
@torch.no_grad()
def estimate_loss(model, loaders, eval_iters, ctx, device):
    """Average the loss and next-token accuracy over up to eval_iters batches of each split, with dropout off.

    loaders maps a split name ('train', 'val') to its DataLoader. Returns {split: mean loss} plus
    {split + '_acc': top-1 accuracy, split + '_top5': top-5 accuracy}, the accuracies as fractions.
    """
    out = {}
    model.eval()
    for split, loader in loaders.items():
        n_batches = min(eval_iters, len(loader))
        losses, accs, top5s = torch.zeros(n_batches), torch.zeros(n_batches), torch.zeros(n_batches)
        for k, (X, Y) in zip(range(n_batches), loader):
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
            top5 = logits.topk(5, dim=-1).indices  # (B, T, 5), most likely token first
            accs[k] = (top5[..., 0] == Y).float().mean().item()
            top5s[k] = (top5 == Y.unsqueeze(-1)).any(dim=-1).float().mean().item()
        out[split] = losses.mean().item()
        out[f"{split}_acc"] = accs.mean().item()
        out[f"{split}_top5"] = top5s.mean().item()
    model.train()
    return out



def save_checkpoint(obj, path):
    """Save obj to path atomically: write a temporary file, then swap it into place.

    If the process dies mid-write, the previous checkpoint at path is still intact
    instead of being replaced by a half-written file.
    """
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def load_resume_checkpoint(ckpt_path, device):
    """Load the rolling checkpoint that train.py writes at every evaluation."""
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"init_from='resume' but there is no checkpoint at {ckpt_path}; train with init_from='scratch' first"
        )
    print(f"resuming training from {ckpt_path}")
    # weights_only=True only unpickles tensors and plain Python types, never arbitrary objects
    return torch.load(ckpt_path, map_location=device, weights_only=True)


def verify_resume_config(checkpoint, run_config, data_stats):
    """Make sure the current settings match the ones the checkpoint was trained with.

    Checks the model architecture, every setting outside RESUME_MAY_DIFFER and the
    dataset's token counts. Allowed differences are printed; any other difference
    raises a ValueError listing all of them, so a resume never quietly turns into a
    different run.
    """
    mismatches = []
    saved_args = {"norm": "layernorm", "mlp": "gelu","pos_emb": "learned",**checkpoint["model_args"]}
    # a GPT-2-initialised run takes its architecture from the pretrained weights, not from the config;
    # only block_size (cropped) and dropout (overridden) come from the config
    pretrained = checkpoint["config"].get("init_from", "").startswith("gpt2")
    for key in MODEL_ARG_KEYS:
        if pretrained and key not in ("block_size", "dropout"):
            continue
        saved = saved_args[key]
        if saved != run_config[key]:
            mismatches.append(f"{key}: checkpoint={saved!r}, current={run_config[key]!r}")

    saved_config = checkpoint["config"]
    for key in sorted(set(saved_config) | set(run_config)):
        # model keys were compared above, and init_from is always 'resume' at this point
        if key in MODEL_ARG_KEYS or key == "init_from":
            continue
        old, new = saved_config.get(key, "<missing>"), run_config.get(key, "<missing>")
        if old == new:
            continue
        if key in RESUME_MAY_DIFFER:
            print(f"note: {key} changed since the checkpoint: {old!r} -> {new!r}")
        else:
            mismatches.append(f"{key}: checkpoint={old!r}, current={new!r}")

    if checkpoint["data_stats"] != data_stats:
        mismatches.append(f"dataset token counts: checkpoint={checkpoint['data_stats']}, current={data_stats}")

    if mismatches:
        raise ValueError(
            "cannot resume, the current config differs from the checkpoint:\n  "
            + "\n  ".join(mismatches)
            + "\nset these back in config/train_config.py, or start a new run with init_from='scratch'"
        )


def main(cfg):
    """Train the model described by cfg, evaluating and checkpointing every cfg.eval_interval steps and after every epoch."""
    ddp, rank, local_rank, world_size, device = setup_distributed(cfg)
    master_process = rank == 0  # only rank 0 evaluates, logs and writes checkpoints
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    # gradient accumulation: micro-batches on every rank add up to total_batch_size tokens per step
    tokens_per_micro_step = cfg.batch_size * cfg.block_size * world_size #batch here is micro batch
    if cfg.total_batch_size % tokens_per_micro_step != 0:
        raise ValueError(
            f"total_batch_size ({cfg.total_batch_size}) must be divisible by "
            f"batch_size * block_size * world_size ({tokens_per_micro_step})"
        )
    grad_accum_steps = cfg.total_batch_size // tokens_per_micro_step

    out_dir = PROJECT_ROOT / cfg.out_dir
    ckpt_path = out_dir / "ckpt.pt"  # latest evaluation, overwritten at every evaluation; used to resume
    best_path = out_dir / "best.pt"  # weights of the evaluation with the lowest val loss; used by sample.py
    if master_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"tokens per optimizer step: {cfg.total_batch_size:,} = {grad_accum_steps} grad accum steps "
            f"x {world_size} process(es) x {cfg.batch_size} sequences x {cfg.block_size} tokens"
        )

    torch.manual_seed(cfg.seed + rank)  # each DDP process gets its own seed (for dropout)
    torch.set_float32_matmul_precision("high")  # TF32 matmuls on Ampere and newer GPUs
    torch.backends.cudnn.allow_tf32 = True
    ctx = get_autocast_context(device_type, cfg.dtype)

    # ---------------------------------------------------------------- data
    # data_stats (token counts per split) is stored in the checkpoint so a resume can detect a changed dataset
    train_loader, val_loader, train_sampler, data_stats = build_dataloaders(
        PROJECT_ROOT / cfg.data_dir / cfg.dataset,
        cfg.block_size,
        cfg.batch_size,
        cfg.val_fraction,
        cfg.seed,
        rank=rank,
        world_size=world_size,
        num_workers=cfg.num_workers,
        pin_memory=device_type == "cuda",
    )  
    # print(data_stats)
    
    # micro-batches left over at the end of an epoch, too few for a full step, are skipped
    steps_per_epoch = len(train_loader) // grad_accum_steps
    if steps_per_epoch == 0:
        raise ValueError(
            f"the train split has {len(train_loader)} micro-batches per process, fewer than the "
            f"{grad_accum_steps} needed for one optimizer step; lower total_batch_size or use more data"
        )
    total_iters = cfg.max_epochs * steps_per_epoch
    warmup_iters = int(cfg.warmup_frac * total_iters)
    if master_process:
        print(f"train tokens: {data_stats['train_tokens']:,} | val tokens: {data_stats['val_tokens']:,}")
        print(
            f"1 epoch = {len(train_loader)} micro-batches per process = {steps_per_epoch} optimizer steps; "
            f"Total {cfg.max_epochs} epochs = {total_iters} steps ({warmup_iters} warmup)"
        )
    # import sys; sys.exit(0)
    # ---------------------------------------------------------------- model
    run_config = asdict(cfg)  # the settings stored in every checkpoint and sent to wandb
    model_args = {k: getattr(cfg, k) for k in MODEL_ARG_KEYS} #use getattr whenever the attribute name is only known while the program run
    start_epoch, iter_num = 0, 0 #iternum- step counter acrros all epochs and resumes.
    best_val_loss, best_epoch, best_iter = float("inf"), -1, -1
    wandb_run_id = None
    checkpoint = None
    if cfg.init_from == "scratch":
        if master_process:
            print("initializing a new model from scratch")
        model = GPT(GPTConfig(**model_args))
    elif cfg.init_from == "resume":
        checkpoint = load_resume_checkpoint(ckpt_path, device)
        verify_resume_config(checkpoint, run_config, data_stats)
        model = GPT.from_checkpoint(checkpoint)
        iter_num = checkpoint["iter_num"]
        # continue in the epoch the checkpoint was saved in, or in the next one if it was saved at an epoch's end
        start_epoch = iter_num // steps_per_epoch
        best_val_loss = checkpoint["best_val_loss"]
        best_epoch = checkpoint["best_epoch"]
        # checkpoints from before eval_interval existed only evaluated at the end of an epoch
        best_iter = checkpoint.get("best_iter", (best_epoch + 1) * steps_per_epoch)
        wandb_run_id = checkpoint["wandb_run_id"]
        if master_process:
            print(
                f"resuming at epoch {start_epoch}, step {iter_num} "
                f"(best val loss {best_val_loss:.4f} at step {best_iter})"
            )
    elif cfg.init_from.startswith("gpt2"):
        if master_process:
            print(f"initializing from OpenAI GPT-2 weights: {cfg.init_from}")
        model = GPT.from_pretrained(cfg.init_from, dict(dropout=cfg.dropout))
        if cfg.block_size > model.config.block_size:
            raise ValueError(f"block_size {cfg.block_size} is larger than {cfg.init_from}'s {model.config.block_size}")
        # the architecture comes from the pretrained model, not from cfg
        model_args = {k: getattr(model.config, k) for k in MODEL_ARG_KEYS}
    else:
        raise ValueError(f"unknown init_from {cfg.init_from!r}")
    # shorten the context if the config asks for less than the model supports (GPT-2 has 1024)
    if cfg.block_size < model.config.block_size:
        model.crop_block_size(cfg.block_size)
        model_args["block_size"] = cfg.block_size
    run_config.update(model_args)  # checkpoints record the real architecture
    model.to(device)

    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])  # restores AdamW's moment estimates
    checkpoint = None  # free the memory held by the loaded checkpoint

    raw_model = model  # the plain module: clean state_dict keys for saving, and estimate_mfu
    if cfg.compile:
        if master_process:
            print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)
    # evaluation runs on rank 0 alone, so it uses the model without the DDP wrapper;
    # a DDP forward pass may start collective ops that the other ranks would never join
    eval_model = model
    if ddp:
        model = DDP(model, device_ids=[local_rank] if device_type == "cuda" else None)
    eval_loaders = {"train": train_loader, "val": val_loader}

    if cfg.eval_only:
        if master_process:
            losses = estimate_loss(eval_model, eval_loaders, cfg.eval_iters, ctx, device)
            print(f"eval only: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        cleanup_distributed(ddp)
        return
    if start_epoch >= cfg.max_epochs:
        if master_process:
            print(f"nothing to do: all {cfg.max_epochs} epochs are done; raise max_epochs to keep training")
        cleanup_distributed(ddp)
        return

    # ---------------------------------------------------------------- logging
    use_wandb = cfg.wandb_log and master_process
    if use_wandb:
        import wandb,inspect

        arch = f"{model_args['norm']}-{model_args['mlp']}-{model_args['pos_emb']}"
        n_params = raw_model.get_num_params()
        # shown in W&B but kept out of run_config: run_config goes into checkpoints and is compared
        # on resume, and these may change legitimately (a resumed run can land on another GPU)
        wandb_extra = {
            # model
            "arch": arch,
            "num_params": n_params,  # without the position table, like the printout
            "num_params_total": raw_model.get_num_params(non_embedding=False),
            "head_size": model_args["n_embd"] // model_args["n_head"],
            "mlp_hidden_dim": raw_model.get_parameter("transformer.h.0.mlp.c_proj.weight").shape[1],
            "weight_tying": True,  # wte and lm_head share one matrix
            "attention": "flash (scaled_dot_product_attention)"
            if hasattr(torch.nn.functional, "scaled_dot_product_attention")
            else "manual",
            # data
            "tokenizer": "GPT-2 BPE (tiktoken)",
            "train_tokens": data_stats["train_tokens"],
            "val_tokens": data_stats["val_tokens"],
            # optimisation, derived from the settings
            "optimizer": "AdamW (fused)" if optimizer.defaults.get("fused") else "AdamW", #reporting if optimiser is actually using fused
            "lr_schedule": "linear warmup + cosine decay to min_lr, per step" if cfg.decay_lr else "constant",
            "grad_accum_steps": grad_accum_steps,
            "world_size": world_size,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_iters,
            "warmup_steps": warmup_iters,
            "tokens_total": total_iters * cfg.total_batch_size,
            "tokens_per_param": total_iters * cfg.total_batch_size / n_params,  # Chinchilla-optimal is ~20
            # system
            "device_name": torch.cuda.get_device_name(device) if device_type == "cuda" else "cpu",
            "torch_version": torch.__version__,
            "matmul_precision": "high (TF32)",
        }
        # a resumed run reuses the stored run id, so it keeps logging to the same W&B run
        run = wandb.init(
            project=cfg.wandb_project,
            # an empty wandb_run_name gets one from the architecture, e.g. "12L-768d-rmsnorm-swiglu-rope"
            name=cfg.wandb_run_name or f"{model_args['n_layer']}L-{model_args['n_embd']}d-{arch}",# commenting for 1 ep of 7b tokens
            tags=[model_args["norm"], model_args["mlp"], model_args["pos_emb"]],
            config=run_config,
            id=wandb_run_id,
            resume="allow",
        )
        run.config.update(wandb_extra, allow_val_change=True)
        wandb_run_id = run.id
        # plot every metric against the optimizer step
        wandb.define_metric("iter")
        wandb.define_metric("train/*", step_metric="iter")
        wandb.define_metric("eval/*", step_metric="iter")
        wandb.define_metric("sys/*", step_metric="iter")
        # the runs table shows each run's lowest val loss instead of its last one
        wandb.define_metric("eval/val_loss", step_metric="iter", summary="min")

    # ---------------------------------------------------------------- training loop
    t0 = time.time()
    local_iter_num = 0  # steps taken by this process; MFU is skipped for the first few
    running_mfu = -1.0
    for epoch in range(start_epoch, cfg.max_epochs):
        train_sampler.set_epoch(epoch)  # new shuffle each epoch, fixed by (seed, epoch)
        batches = iter(train_loader)
        # steps of this epoch that are already done: non-zero only right after a mid-epoch resume.
        # The batch order is fixed by (seed, epoch), so skipping those batches continues exactly
        # where the checkpoint left off
        done_steps = iter_num - epoch * steps_per_epoch #number of steps done in this epoch
        if done_steps > 0:
            if master_process:
                print(f"skipping the {done_steps} steps of epoch {epoch} that the checkpoint already trained on")
            for _ in range(done_steps * grad_accum_steps):
                next(batches)
            t0 = time.time()  # keep the skipping out of the first step's timing
        for _ in range(done_steps, steps_per_epoch):
            # set the learning rate for this step
            lr = get_lr(iter_num, cfg, warmup_iters, total_iters) if cfg.decay_lr else cfg.learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # forward and backward over grad_accum_steps micro-batches; gradients add up in .grad
            loss_accum = torch.zeros((), device=device) # loss over all tokens in one opt step 
            for micro_step in range(grad_accum_steps):
                X, Y = next(batches)
                X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
                if ddp:
                    # all-reduce gradients across ranks only on the last micro-step
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                with ctx:
                    _, loss = model(X, Y)
                # scale so the summed gradients equal the gradient of the mean loss over the full batch
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()
            if ddp:
                # average the logged loss over ranks (gloo has no AVG op, so sum and divide)
                dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
                loss_accum /= world_size

            # clip the global gradient norm (an infinite limit just measures it), then update
            max_norm = cfg.grad_clip if cfg.grad_clip > 0.0 else float("inf")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # release gradient memory until the next backward

            # timing and logging
            if device_type == "cuda":
                torch.cuda.synchronize()  # wait for queued GPU work so the timing is real
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % cfg.log_interval == 0 and master_process:
                lossf = loss_accum.item()
                normf = norm.item()
                if local_iter_num >= 5:  # skip the first steps (compilation, warm-up) when measuring MFU
                    mfu = raw_model.estimate_mfu(cfg.batch_size * grad_accum_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                tokens_per_sec = cfg.total_batch_size / dt
                mfu_str = f"{running_mfu * 100:.2f}%" if running_mfu >= 0 else "n/a"
                print(
                    f"epoch {epoch} | step {iter_num} | loss {lossf:.4f} | lr {lr:.4e} | norm {normf:.4f} | "
                    f"dt {dt * 1000:.2f}ms | tok/s {tokens_per_sec:,.0f} | mfu {mfu_str}"
                )
                if use_wandb:
                    metrics = {
                        "iter": iter_num,
                        "epoch": epoch,
                        "train/step_loss": lossf,
                        "train/perplexity": math.exp(lossf),
                        "train/lr": lr,
                        "train/grad_norm": normf,  # measured before clipping
                        "train/grad_clipped": float(0.0 < cfg.grad_clip < normf),  # 1 when clipping kicked in
                        "train/tokens_seen": (iter_num + 1) * cfg.total_batch_size,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/step_time_ms": dt * 1000,
                    }
                    if running_mfu >= 0:
                        metrics["train/mfu"] = running_mfu * 100  # as a percentage
                    if device_type == "cuda":
                        metrics["sys/gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
                    wandb.log(metrics)

            iter_num += 1
            local_iter_num += 1

            # evaluate, log and checkpoint every eval_interval steps and at the end of every epoch (rank 0 only)
            end_of_epoch = iter_num == (epoch + 1) * steps_per_epoch #if we are at last step in epoch
            if end_of_epoch or (cfg.eval_interval > 0 and iter_num % cfg.eval_interval == 0):
                if master_process:
                    losses = estimate_loss(eval_model, eval_loaders, cfg.eval_iters, ctx, device)
                    is_best = losses["val"] < best_val_loss
                    if is_best:
                        best_val_loss, best_epoch, best_iter = losses["val"], epoch, iter_num
                    label = f"epoch {epoch} done" if end_of_epoch else f"step {iter_num} eval"
                    print(
                        f"{label}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} "
                        f"(best {best_val_loss:.4f} at step {best_iter})"
                    )
                    if use_wandb:
                        wandb.log({
                            "iter": iter_num,
                            "epoch": epoch,
                            "eval/train_loss": losses["train"],
                            "eval/val_loss": losses["val"],
                            "eval/val_perplexity": math.exp(losses["val"]),
                            "eval/train_acc": 100 * losses["train_acc"],
                            "eval/val_acc": 100 * losses["val_acc"],
                            "eval/val_top5_acc": 100 * losses["val_top5"],
                            "eval/overfit_gap": losses["val"] - losses["train"],
                            "eval/best_val_loss": best_val_loss,
                            "eval/best_epoch": best_epoch,
                            "eval/best_step": best_iter,
                        })

                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "config": run_config,
                        "data_stats": data_stats,
                        "epoch": epoch,  # the epoch in progress (finished, if iter_num is at its end)
                        "iter_num": iter_num,  # optimizer steps done; a resume continues from here
                        "val_loss": losses["val"],
                        "best_val_loss": best_val_loss,
                        "best_epoch": best_epoch,
                        "best_iter": best_iter,
                        "wandb_run_id": wandb_run_id,
                    }
                    if is_best:
                        # best.pt holds what sample.py needs: weights and architecture, no optimizer state
                        best = {k: checkpoint[k] for k in ("model", "model_args", "config", "epoch", "iter_num", "val_loss")}
                        save_checkpoint(best, best_path)
                    save_checkpoint(checkpoint, ckpt_path)
                    print(f"saved {ckpt_path.name}{' and ' + best_path.name if is_best else ''} to {out_dir}")
                    checkpoint = None
                t0 = time.time()  # keep evaluation and checkpointing out of the next step's timing

    if master_process:
        print(f"training finished: best val loss {best_val_loss:.4f} at step {best_iter} (epoch {best_epoch})")
    if use_wandb:
        wandb.finish()
    cleanup_distributed(ddp)


if __name__ == "__main__":
    main(TrainConfig())
