"""Unit tests for train.py: the helpers, and main() end to end on a tiny model and random tokens."""

import dataclasses
from contextlib import nullcontext
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

import dataset
import train
from config import TrainConfig
from model import GPT, GPTConfig

VOCAB = 64
BLOCK = 8
BATCH = 2
GRAD_ACCUM = 2
STEPS_PER_EPOCH = 3  # 12 train blocks = 6 micro-batches = 3 optimizer steps


def tiny_train_config(tmp_path, **overrides):
    """A TrainConfig for a model and a dataset small enough to train in about a second on a CPU."""
    args = dict(
        out_dir=str(tmp_path / "out"),  # absolute, so PROJECT_ROOT / out_dir lands in tmp_path
        data_dir=str(tmp_path),
        dataset="data",  # never read: the fake_data fixture replaces load_tokens
        log_interval=1,
        eval_interval=0,
        eval_iters=2,
        init_from="scratch",
        wandb_log=False,
        batch_size=BATCH,
        block_size=BLOCK,
        total_batch_size=BATCH * BLOCK * GRAD_ACCUM,
        num_workers=0,
        n_layer=1,
        n_head=2,
        n_embd=16,
        vocab_size=VOCAB,
        dropout=0.0,
        norm="rmsnorm",
        mlp="swiglu",
        pos_emb="rope",
        learning_rate=1e-3,
        min_lr=1e-4,
        max_epochs=2,
        warmup_frac=0.2,
        backend="gloo",
        device="cpu",
        dtype="float32",
        compile=False,
        seed=1337,
    )
    args.update(overrides)
    return TrainConfig(**args)


@pytest.fixture
def fake_data(monkeypatch):
    """Serve random tokens instead of reading a dataset, and make sure train.py does not think it runs under torchrun."""
    rng = np.random.default_rng(0)
    train_tokens = rng.integers(0, VOCAB, size=12 * BLOCK + 1).astype(np.uint16)
    val_tokens = rng.integers(0, VOCAB, size=4 * BLOCK + 1).astype(np.uint16)
    monkeypatch.setattr(dataset, "load_tokens", lambda data_path, val_fraction: (train_tokens, val_tokens))
    monkeypatch.delenv("RANK", raising=False)
    return {"train_tokens": len(train_tokens), "val_tokens": len(val_tokens)}


def load(path):
    """Load a checkpoint onto the CPU the way train.py and sample.py do (tensors and plain types only)."""
    return torch.load(path, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------- config keys


def test_model_arg_keys_are_the_gpt_config_fields():
    """MODEL_ARG_KEYS follows GPTConfig, so a new architecture field is saved in checkpoints automatically."""
    assert set(train.MODEL_ARG_KEYS) == {f.name for f in dataclasses.fields(GPTConfig)}


def test_every_model_arg_and_resume_key_is_a_train_setting():
    """Every architecture key and every key allowed to change on resume exists in TrainConfig."""
    settings = {f.name for f in dataclasses.fields(TrainConfig)}
    assert set(train.MODEL_ARG_KEYS) <= settings
    assert train.RESUME_MAY_DIFFER <= settings


# ---------------------------------------------------------------- small helpers


def test_resolve_device():
    """'auto' becomes cuda or cpu; any explicit device is returned unchanged."""
    assert train.resolve_device("cpu") == "cpu"
    assert train.resolve_device("cuda:1") == "cuda:1"
    assert train.resolve_device("auto") == ("cuda" if torch.cuda.is_available() else "cpu")


def test_setup_distributed_without_torchrun(monkeypatch):
    """Without torchrun's RANK variable there is one process: no DDP, rank 0, world size 1."""
    monkeypatch.delenv("RANK", raising=False)
    cfg = TrainConfig(device="cpu")
    assert train.setup_distributed(cfg) == (False, 0, 0, 1, "cpu")


def test_autocast_context():
    """float32 needs no autocast, bfloat16 gets torch.autocast, and other dtypes are rejected."""
    assert isinstance(train.get_autocast_context("cpu", "float32"), nullcontext)
    assert isinstance(train.get_autocast_context("cpu", "bfloat16"), torch.autocast)
    with pytest.raises(ValueError, match="unsupported dtype"):
        train.get_autocast_context("cpu", "float16")


def test_lr_schedule():
    """Linear warmup up to learning_rate, cosine decay down to min_lr, then held at min_lr."""
    cfg = TrainConfig(learning_rate=1.0, min_lr=0.1)
    lr = lambda it: train.get_lr(it, cfg, warmup_iters=10, lr_decay_iters=100)
    # linear warmup
    assert lr(0) == pytest.approx(1 / 11)
    assert lr(9) == pytest.approx(10 / 11)
    assert all(lr(i) < lr(i + 1) for i in range(9))
    # cosine decay: peak right after warmup, halfway at the middle, min_lr at the end
    assert lr(10) == pytest.approx(1.0)
    assert lr(55) == pytest.approx(0.55)
    assert lr(100) == pytest.approx(0.1)
    assert all(lr(i) >= lr(i + 1) for i in range(10, 100))
    # held at min_lr afterwards
    assert lr(150) == pytest.approx(0.1)


def test_save_checkpoint_replaces_the_file_and_leaves_no_temp(tmp_path):
    """save_checkpoint overwrites the old file and leaves no .tmp file behind."""
    path = tmp_path / "ckpt.pt"
    train.save_checkpoint({"iter_num": 1}, path)
    train.save_checkpoint({"iter_num": 2}, path)
    assert load(path) == {"iter_num": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt.pt"]


def test_load_resume_checkpoint(tmp_path):
    """A missing checkpoint raises a helpful FileNotFoundError; an existing one is loaded."""
    path = tmp_path / "ckpt.pt"
    with pytest.raises(FileNotFoundError, match="init_from='scratch'"):
        train.load_resume_checkpoint(path, "cpu")
    torch.save({"iter_num": 7}, path)
    assert train.load_resume_checkpoint(path, "cpu") == {"iter_num": 7}


# ---------------------------------------------------------------- verify_resume_config


DATA_STATS = {"train_tokens": 1000, "val_tokens": 100}


def make_checkpoint(cfg):
    """The parts of a train.py checkpoint that verify_resume_config reads."""
    run_config = asdict(cfg)
    return {
        "model_args": {k: run_config[k] for k in train.MODEL_ARG_KEYS},
        "config": dict(run_config, init_from="scratch"),
        "data_stats": dict(DATA_STATS),
    }


def test_resume_with_the_same_settings_passes(tmp_path):
    """Resuming with unchanged settings is accepted (init_from itself may differ)."""
    cfg = tiny_train_config(tmp_path)
    checkpoint = make_checkpoint(cfg)
    train.verify_resume_config(checkpoint, asdict(replace(cfg, init_from="resume")), DATA_STATS)


def test_resume_allows_io_and_hardware_changes(tmp_path, capsys):
    """Settings in RESUME_MAY_DIFFER may change; each change is printed as a note."""
    cfg = tiny_train_config(tmp_path)
    checkpoint = make_checkpoint(cfg)
    changed = replace(cfg, out_dir="elsewhere", max_epochs=10, eval_iters=50, compile=True)
    train.verify_resume_config(checkpoint, asdict(changed), DATA_STATS)
    printed = capsys.readouterr().out
    assert "note: out_dir changed" in printed and "note: max_epochs changed" in printed


@pytest.mark.parametrize(
    "change", [dict(learning_rate=1.0), dict(n_layer=4), dict(norm="layernorm"), dict(seed=1), dict(batch_size=4)]
)
def test_resume_rejects_other_changes(tmp_path, change):
    """Changing an architecture or optimisation setting makes the resume fail, naming the setting."""
    cfg = tiny_train_config(tmp_path)
    checkpoint = make_checkpoint(cfg)
    key = next(iter(change))
    with pytest.raises(ValueError, match=key):
        train.verify_resume_config(checkpoint, asdict(replace(cfg, **change)), DATA_STATS)


def test_resume_rejects_a_changed_dataset(tmp_path):
    """Different token counts mean a different dataset, so the resume fails."""
    cfg = tiny_train_config(tmp_path)
    checkpoint = make_checkpoint(cfg)
    with pytest.raises(ValueError, match="dataset token counts"):
        train.verify_resume_config(checkpoint, asdict(cfg), {"train_tokens": 999, "val_tokens": 100})


def test_resume_lists_every_mismatch(tmp_path):
    """All mismatches are reported in one error, not just the first."""
    cfg = tiny_train_config(tmp_path)
    checkpoint = make_checkpoint(cfg)
    with pytest.raises(ValueError) as err:
        train.verify_resume_config(checkpoint, asdict(replace(cfg, n_embd=32, weight_decay=0.0)), DATA_STATS)
    assert "n_embd" in str(err.value) and "weight_decay" in str(err.value)


def test_resume_of_an_old_checkpoint_defaults_to_the_gpt2_architecture(tmp_path):
    """Checkpoints saved before norm / mlp / pos_emb existed are treated as GPT-2 style."""
    cfg = tiny_train_config(tmp_path, norm="layernorm", mlp="gelu", pos_emb="learned")
    checkpoint = make_checkpoint(cfg)
    for key in ("norm", "mlp", "pos_emb"):
        del checkpoint["model_args"][key]
        del checkpoint["config"][key]
    train.verify_resume_config(checkpoint, asdict(cfg), DATA_STATS)
    with pytest.raises(ValueError, match="norm"):
        train.verify_resume_config(checkpoint, asdict(replace(cfg, norm="rmsnorm")), DATA_STATS)


# ---------------------------------------------------------------- estimate_loss


def test_estimate_loss():
    """estimate_loss returns loss and accuracies per split, averages eval_iters batches and restores train mode."""
    model = GPT(GPTConfig(block_size=BLOCK, vocab_size=VOCAB, n_layer=1, n_head=2, n_embd=16))
    batches = [(torch.randint(0, VOCAB, (2, BLOCK)), torch.randint(0, VOCAB, (2, BLOCK))) for _ in range(3)]
    loaders = {"train": batches, "val": batches[:2]}
    model.train()
    out = train.estimate_loss(model, loaders, eval_iters=2, ctx=nullcontext(), device="cpu")

    assert set(out) == {"train", "train_acc", "train_top5", "val", "val_acc", "val_top5"}
    assert model.training  # put back in train mode
    for split in ("train", "val"):
        assert 0.0 <= out[f"{split}_acc"] <= out[f"{split}_top5"] <= 1.0
    # eval_iters=2: both splits average the same first two batches
    with torch.no_grad():
        model.eval()
        expected = sum(model(x, y)[1].item() for x, y in batches[:2]) / 2
    assert out["train"] == pytest.approx(expected) and out["val"] == pytest.approx(expected)


# ---------------------------------------------------------------- main, end to end


def test_training_from_scratch_writes_checkpoints(tmp_path, fake_data):
    """A full run writes ckpt.pt (everything needed to resume) and best.pt (weights only)."""
    cfg = tiny_train_config(tmp_path)
    train.main(cfg)

    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == ["best.pt", "ckpt.pt"]  # no leftover .tmp files
    ckpt = load(out / "ckpt.pt")
    assert ckpt["iter_num"] == 2 * STEPS_PER_EPOCH
    assert ckpt["epoch"] == 1
    assert ckpt["data_stats"] == fake_data
    assert ckpt["model_args"] == {k: getattr(cfg, k) for k in train.MODEL_ARG_KEYS}
    assert ckpt["best_val_loss"] <= ckpt["val_loss"]

    best = load(out / "best.pt")
    assert set(best) == {"model", "model_args", "config", "epoch", "iter_num", "val_loss"}  # no optimizer state
    assert best["val_loss"] == ckpt["best_val_loss"]
    GPT.from_checkpoint(best)  # strict load: every key present, none extra


def test_training_lowers_the_loss(tmp_path, fake_data):
    """The trained model has a lower loss on a training block than a freshly initialised one."""
    train.main(tiny_train_config(tmp_path, max_epochs=10, learning_rate=1e-2))
    ckpt = load(tmp_path / "out" / "ckpt.pt")
    model = GPT.from_checkpoint(ckpt)
    fresh = GPT(GPTConfig(**ckpt["model_args"]))
    x, y = dataset.TokenBlockDataset(dataset.load_tokens(None, None)[0], BLOCK)[0]
    with torch.no_grad():
        assert model(x[None], y[None])[1] < fresh(x[None], y[None])[1]


class Crash(Exception):
    """Raised to simulate the training process dying right after a checkpoint is written."""


def test_mid_epoch_resume_matches_an_uninterrupted_run(tmp_path, fake_data, monkeypatch):
    """A run killed in the middle of an epoch and resumed ends with the same weights as one never interrupted."""
    # uninterrupted: 2 epochs of 3 steps
    straight = tiny_train_config(tmp_path, out_dir=str(tmp_path / "straight"), eval_interval=1)
    train.main(straight)

    # interrupted after step 4 (in the middle of epoch 1), then resumed
    real_save = train.save_checkpoint

    def crash_after_step_4(obj, path):
        """Save as usual, then crash once the step-4 ckpt.pt is on disk."""
        real_save(obj, path)
        if path.name == "ckpt.pt" and obj["iter_num"] == 4:
            raise Crash

    interrupted = replace(straight, out_dir=str(tmp_path / "interrupted"))
    monkeypatch.setattr(train, "save_checkpoint", crash_after_step_4)
    with pytest.raises(Crash):
        train.main(interrupted)
    monkeypatch.setattr(train, "save_checkpoint", real_save)
    assert load(tmp_path / "interrupted" / "ckpt.pt")["iter_num"] == 4

    train.main(replace(interrupted, init_from="resume"))

    a = load(tmp_path / "straight" / "ckpt.pt")
    b = load(tmp_path / "interrupted" / "ckpt.pt")
    assert a["iter_num"] == b["iter_num"] == 2 * STEPS_PER_EPOCH
    for k in a["model"]:
        torch.testing.assert_close(a["model"][k], b["model"][k])
    assert a["best_val_loss"] == pytest.approx(b["best_val_loss"])


def test_resume_with_a_changed_setting_fails(tmp_path, fake_data):
    """main refuses to resume when a setting outside RESUME_MAY_DIFFER has changed."""
    cfg = tiny_train_config(tmp_path, max_epochs=1)
    train.main(cfg)
    with pytest.raises(ValueError, match="learning_rate"):
        train.main(replace(cfg, init_from="resume", learning_rate=5e-3))


def test_resume_when_every_epoch_is_done(tmp_path, fake_data, capsys):
    """Resuming a finished run trains nothing and leaves the checkpoint as it was."""
    cfg = tiny_train_config(tmp_path, max_epochs=1)
    train.main(cfg)
    before = load(tmp_path / "out" / "ckpt.pt")
    train.main(replace(cfg, init_from="resume"))
    assert "nothing to do" in capsys.readouterr().out
    assert load(tmp_path / "out" / "ckpt.pt")["iter_num"] == before["iter_num"]


def test_resume_without_a_checkpoint_fails(tmp_path, fake_data):
    """init_from='resume' with no ckpt.pt in out_dir raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        train.main(tiny_train_config(tmp_path, init_from="resume"))


def test_eval_only_writes_nothing(tmp_path, fake_data, capsys):
    """eval_only evaluates once and exits without training or writing checkpoints."""
    train.main(tiny_train_config(tmp_path, eval_only=True))
    assert "eval only" in capsys.readouterr().out
    assert not list((tmp_path / "out").iterdir())


def test_total_batch_size_must_split_into_micro_batches(tmp_path, fake_data):
    """total_batch_size must be a whole number of batch_size * block_size * world_size micro-batches."""
    with pytest.raises(ValueError, match="must be divisible"):
        train.main(tiny_train_config(tmp_path, total_batch_size=BATCH * BLOCK * 2 + 1))


def test_too_little_data_for_one_step_fails(tmp_path, fake_data):
    """An epoch with fewer micro-batches than one optimizer step needs is rejected."""
    with pytest.raises(ValueError, match="fewer than"):
        train.main(tiny_train_config(tmp_path, total_batch_size=BATCH * BLOCK * 100))


def test_unknown_init_from_fails(tmp_path, fake_data):
    """init_from must be 'scratch', 'resume' or a GPT-2 variant."""
    with pytest.raises(ValueError, match="unknown init_from"):
        train.main(tiny_train_config(tmp_path, init_from="llama"))
