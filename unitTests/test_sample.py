"""Unit tests for sample.py: the helpers, loading a checkpoint and main() end to end on a tiny model."""

from contextlib import nullcontext
from dataclasses import asdict

import pytest
import tiktoken
import torch

import sample
from config import SampleConfig
from model import GPT, GPTConfig

SEPARATOR = "---------------"


@pytest.fixture
def enc():
    """The GPT-2 tokenizer sample.py encodes and decodes with."""
    return tiktoken.get_encoding("gpt2")


@pytest.fixture
def checkpoint_dir(tmp_path):
    """Write a best.pt like train.py does, for a tiny model over the full (padded) GPT-2 vocabulary."""
    cfg = GPTConfig(block_size=32, vocab_size=50304, n_layer=1, n_head=2, n_embd=16)
    model = GPT(cfg)
    torch.save(
        {
            "model": model.state_dict(),
            "model_args": asdict(cfg),
            "config": {},
            "epoch": 0,
            "iter_num": 5,
            "val_loss": 1.2345,
        },
        tmp_path / "best.pt",
    )
    return tmp_path


def sample_config(out_dir, **overrides):
    """A SampleConfig that loads out_dir/best.pt on the CPU and generates a few short samples."""
    args = dict(
        init_from="resume",
        out_dir=str(out_dir),  # absolute, so PROJECT_ROOT / out_dir lands in tmp_path
        ckpt_name="best.pt",
        start="Hello world",
        num_samples=2,
        max_new_tokens=5,
        temperature=0.8,
        top_k=10,
        device="cpu",
        dtype="float32",
        compile=False,
    )
    args.update(overrides)
    return SampleConfig(**args)


# ---------------------------------------------------------------- small helpers


def test_resolve_device():
    """'auto' becomes cuda or cpu; any explicit device is returned unchanged."""
    assert sample.resolve_device("cpu") == "cpu"
    assert sample.resolve_device("auto") == ("cuda" if torch.cuda.is_available() else "cpu")


def test_autocast_context():
    """float32 needs no autocast, bfloat16 gets torch.autocast, and other dtypes are rejected."""
    assert isinstance(sample.get_autocast_context("cpu", "float32"), nullcontext)
    assert isinstance(sample.get_autocast_context("cpu", "bfloat16"), torch.autocast)
    with pytest.raises(ValueError, match="unsupported dtype"):
        sample.get_autocast_context("cpu", "int8")


def test_read_prompt_returns_plain_text():
    """A prompt without the FILE: prefix is used as it is."""
    assert sample.read_prompt("<|endoftext|>Once upon a time") == "<|endoftext|>Once upon a time"


def test_read_prompt_reads_a_file(tmp_path):
    """'FILE:path' is replaced by the file's UTF-8 contents."""
    path = tmp_path / "prompt.txt"
    path.write_text("naïve café\nsecond line", encoding="utf-8")
    assert sample.read_prompt(f"FILE:{path}") == "naïve café\nsecond line"


# ---------------------------------------------------------------- load_model


def test_load_model_from_a_checkpoint(checkpoint_dir, capsys):
    """load_model restores the saved weights exactly and returns the model in eval mode."""
    model = sample.load_model(sample_config(checkpoint_dir), "cpu")
    assert not model.training  # ready for generation
    saved = torch.load(checkpoint_dir / "best.pt", weights_only=True)["model"]
    for k, v in saved.items():
        assert torch.equal(model.state_dict()[k], v)
    assert "step 5" in capsys.readouterr().out


def test_load_model_without_a_checkpoint_fails(tmp_path):
    """A missing checkpoint raises FileNotFoundError telling you to train first."""
    with pytest.raises(FileNotFoundError, match="run train.py first"):
        sample.load_model(sample_config(tmp_path, ckpt_name="missing.pt"), "cpu")


def test_load_model_rejects_an_unknown_source(tmp_path):
    """init_from must be 'resume' or a GPT-2 variant."""
    with pytest.raises(ValueError, match="unknown init_from"):
        sample.load_model(sample_config(tmp_path, init_from="llama"), "cpu")


# ---------------------------------------------------------------- main, end to end


def printed_samples(out):
    """The samples main() printed, without the model-loading lines before them."""
    body = out[out.index("val loss") :].split("\n", 1)[1]
    return [s.strip("\n") for s in body.split(SEPARATOR)[:-1]]


def test_main_prints_every_sample(checkpoint_dir, capsys):
    """main prints num_samples continuations, each starting with the prompt."""
    sample.main(sample_config(checkpoint_dir, num_samples=3))
    samples = printed_samples(capsys.readouterr().out)
    assert len(samples) == 3
    assert all(s.startswith("Hello world") for s in samples)


def test_main_is_reproducible_with_a_seed(checkpoint_dir, capsys):
    """Two runs with the same seed print the same samples."""
    cfg = sample_config(checkpoint_dir, seed=7)
    sample.main(cfg)
    first = printed_samples(capsys.readouterr().out)
    sample.main(cfg)
    assert printed_samples(capsys.readouterr().out) == first


def test_main_drops_padded_token_ids(tmp_path, enc, monkeypatch, capsys):
    """Ids from 50257 up exist only because vocab_size is padded to 50304; they have no text."""

    class FakeModel:
        """Stands in for GPT: always 'generates' a padded id, ' there', and another padded id."""

        def generate(self, x, max_new_tokens, temperature, top_k):
            """Return the prompt followed by the fixed tokens."""
            new = [50300] + enc.encode(" there") + [50257]
            return torch.cat((x, torch.tensor([new])), dim=1)

    monkeypatch.setattr(sample, "load_model", lambda cfg, device: FakeModel())
    sample.main(sample_config(tmp_path, start="Hello", num_samples=1))
    assert capsys.readouterr().out == f"Hello there\n{SEPARATOR}\n"
