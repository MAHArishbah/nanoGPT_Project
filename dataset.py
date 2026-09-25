"""
Data pipeline: loads GPT-2 BPE tokens, either from the train.bin / val.bin files that
prepare_fineweb.py writes or by tokenising a text file and splitting it into train /
validation tokens, and serves (input, target) blocks through torch.utils.data.DataLoader.
"""

import os

import numpy as np
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def load_tokens(data_path, val_fraction):
    """Return (train_tokens, val_tokens) for data_path, as uint16 arrays of GPT-2 token ids.

    data_path is either a folder written by prepare_fineweb.py, holding train.bin and
    val.bin, or a text file. The folder's files are opened as memory maps: nothing is read
    until a batch needs it, so a dataset larger than RAM works, and val_fraction is not
    used. A text file is encoded with GPT-2 BPE here; the first (1 - val_fraction) of its
    tokens form the train split and the rest the validation split. uint16 (GPT-2 ids are
    < 65536) is 4x smaller than int64.
    """
    if os.path.isdir(data_path):
        train = np.memmap(os.path.join(data_path, "train_1ep.bin"), dtype=np.uint16, mode="r")
        val = np.memmap(os.path.join(data_path, "val.bin"), dtype=np.uint16, mode="r")
        return train, val
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"dataset not found: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    enc = tiktoken.get_encoding("gpt2")
    tokens = np.array(enc.encode_ordinary(text), dtype=np.uint16)
    split = len(tokens) - int(len(tokens) * val_fraction)
    return tokens[:split], tokens[split:]


class TokenBlockDataset(Dataset):
    """Cuts a 1-D token array into non-overlapping blocks of block_size tokens.

    Item i is (x, y): x = tokens[i*T : i*T + T] and y is x shifted one token to
    the left, i.e. the next-token targets. The blocks do not overlap, so one pass
    over the dataset (one epoch) sees every token once.
    """

    def __init__(self, tokens, block_size):
        """Store the token array and the block length; fail early if not even one block fits."""
        self.tokens = tokens
        self.block_size = block_size
        if len(self) == 0:
            raise ValueError(f"{len(tokens)} tokens are too few for a single block of {block_size}")

    def __len__(self):
        """Number of complete blocks. The -1 keeps room for the target of the last input token."""
        return (len(self.tokens) - 1) // self.block_size

    def __getitem__(self, idx):
        """Return block idx as int64 tensors (x, y), each of shape (block_size,)."""
        start = idx * self.block_size
        chunk = torch.from_numpy(self.tokens[start : start + self.block_size + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


def build_dataloaders(data_path, block_size, batch_size, val_fraction, seed,
                      rank=0, world_size=1, num_workers=0, pin_memory=False):
    """Create the train and validation DataLoaders.

    Returns (train_loader, val_loader, train_sampler, token_counts), where token_counts
    is {'train_tokens': int, 'val_tokens': int}. The train data goes through a
    DistributedSampler: under DDP it gives every rank its own shard, and with a single
    process (world_size=1) it is a plain shuffle. Either way the order depends only on
    (seed, epoch), so call train_sampler.set_epoch(epoch) before each epoch; a resumed
    run then sees the same batches as an uninterrupted one. The validation loader is
    not shuffled, so every evaluation scores the same batches and epochs compare fairly.
    """
    train_tokens, val_tokens = load_tokens(data_path, val_fraction)
    train_ds = TokenBlockDataset(train_tokens, block_size)
    val_ds = TokenBlockDataset(val_tokens, block_size)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True,  # keep every micro-batch the same shape
        num_workers=num_workers,
        pin_memory=pin_memory,  # page-locked memory allows async host-to-GPU copies
        persistent_workers=False,  # evaluation also iterates this loader; a persistent loader would reset the training loop's iterator
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers >0
    )
    token_counts = {"train_tokens": len(train_tokens), "val_tokens": len(val_tokens)}
    return train_loader, val_loader, train_sampler, token_counts
