"""
Download FineWeb-Edu, tokenise it with the GPT-2 tokenizer and save it as the token files
that train.py reads: a folder holding train.bin and val.bin (see dataset.load_tokens).

$ python prepare_fineweb.py

The training budget follows Chinchilla: about 20 training tokens per model parameter(updated to 60 tokens per paramter in the script after
runs with 2o tokens per paramter for 3 epochs), so the
124M model gets ~2.5B(*3) tokens. Docs streamed from the Hugging Face Hub, so only the
part that is used gets downloaded, and files that Hugging Face's malware scan marked unsafe
are skipped(visited the site , saw the tag and then updated the script ). Every document starts with <|endoftext|>, which tells the model where one
document ends and the next begins. config is in the same file .
"""

import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from huggingface_hub import HfApi

# ---------------------------------------------------------------- settings
DATASET = "HuggingFaceFW/fineweb-edu"  #repo id for dataset
SAMPLE_DIR = "sample/10BT"  # a random ~10B-token sample of the full dataset (the "sample-10BT" subset)
N_PARAMS = 124_000_000  # parameters of the model the budget is for (GPT-2 small)
TOKENS_PER_PARAM = 60  # 3x Chinchilla's 20: same number of tokens seen as the 3-epoch runs, but all unique
TRAIN_TOKENS = N_PARAMS * TOKENS_PER_PARAM  # 7.44B tokens, ~14.9 GB on disk
VAL_TOKENS = 10_000_000  # held out; each evaluation reads only eval_iters batches of it
OUT_DIR = Path(__file__).resolve().parent / "data" / "fineweb"  # train.py: data_dir="data", dataset="fineweb"
NUM_PROCS = max(1, (os.cpu_count() or 2) - 1)  # tokeniser processes; one core streams and writes

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token  # 50256, <|endoftext|>


def safe_data_files():
    """Return (files to read, files skipped): the sample's parquet files in order excluding any flagged unsafe.

    Hf scans every uploaded file for malware. Web text sometimes matches a virus
    signature (a forum post quoting exploit code, say); such a file is skipped. Files the
    scan has not reached ("unscanned", usually because they are large) are kept.
    """
    files = sorted(
        (f for f in HfApi().list_repo_tree(DATASET, repo_type="dataset", path_in_repo=SAMPLE_DIR, expand=True)
         if f.path.endswith(".parquet")),
        key=lambda f: f.path,
    )#sorting to make the order reproducible under different runs
    unsafe = [f.path for f in files if f.security is not None and f.security.get("status") == "unsafe"] #not none has to be called first as some files didnt have security attribute , gave error the other way 
    return [f.path for f in files if f.path not in unsafe], unsafe #paths relative to root repo


def tokenize(text):
    """GPT-2 token ids of one document as uint16, with <|endoftext|> in front.Converted to np array to have 2 bytes per token rather than 28+bytes for python int object.
    Fucntion runs inside the workers. [EOT] + enc.encode_ordinary(text) this costs a little speed as list is copied again"""
    return np.array([EOT] + enc.encode_ordinary(text), dtype=np.uint16) #runs BPE returns int, changed to uint16 as its below  2^16-1 (65535) above this need uint32


def write_tokens(path, docs, n_tokens, label):
    """Write the next n_tokens tokens from the docs iterator to path and return the number of documents.

    The last document is cut to fit exactly. The file is written under a temporary name and
    renamed at the end, so an interrupted run never leaves a half-written file behind.
    """
    tmp_path = path.with_name(path.name + ".tmp") #rename is atomic with in a single filesysetm 
    written, n_docs, t0 = 0, 0, time.time()
    report_every = max(n_tokens // 20, 1)  # progress every 5%
    next_report = report_every
    with open(tmp_path, "wb") as f:
        #docs is a single stateful iterator, the imap results, and both calls to write_tokens share it.
        #When the val call breaks, the iterator stays positioned at the next document. The train call then continues from exactly that point.
        #  for  "val first, then train, with no document in both" without being  explicit.
        for tokens in docs:
            tokens = tokens[: n_tokens - written] #cutoff wil be discarded
            f.write(tokens.tobytes())
            # tokens.tofile(f) #little faster as tokens.tobytes() makes another copy
            written += len(tokens)
            n_docs += 1
            if written >= next_report:
                rate = written / (time.time() - t0)
                eta = (n_tokens - written) / rate
                print(f"{label}: {written:>13,} / {n_tokens:,} tokens | {rate / 1e6:.2f}M tokens/s | ~{eta / 60:.0f} min left")
                next_report += report_every
            if written == n_tokens:
                break
    if written < n_tokens:
        raise RuntimeError(f"{DATASET} ran out after {written:,} {label} tokens; {n_tokens:,} were wanted")
    os.replace(tmp_path, path,)
    return n_docs


def main():
    """Stream the dataset, tokenise it in parallel and write val.bin, train.bin and info.json."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_files, skipped = safe_data_files()
    print(f"{DATASET} ({SAMPLE_DIR}, {len(data_files)} files) -> {OUT_DIR}")
    if skipped:
        print(f"skipping {len(skipped)} file(s) flagged unsafe by Hugging Face's malware scan: {skipped}")
    print(f"{VAL_TOKENS:,} val tokens + {TRAIN_TOKENS:,} train tokens, {NUM_PROCS} tokeniser processes")
    dataset = load_dataset(DATASET, data_files=data_files, split="train", streaming=True) # here 'train' doesnt has anything to do with our splits. Its HF convention it just puts all under train when passing files directly
    texts = (row["text"] for row in dataset) #dropping other rows in dataset like metadata, score etc an dlazy loading via generaotr
    with Pool(NUM_PROCS) as pool:
        # imap keeps the documents in order, so the split is the same on every run; val comes
        # first and train continues with the next document, so no document is in both
        docs = pool.imap(tokenize, texts, chunksize=64) #a lazy, ordered iterator of token arrays, one per document.
        val_docs = write_tokens(OUT_DIR / "val.bin", docs, VAL_TOKENS, "val")
        train_docs = write_tokens(OUT_DIR / "train_1ep.bin", docs, TRAIN_TOKENS, "train")
    info = {
        "dataset": DATASET,
        "sample": SAMPLE_DIR,
        "skipped_unsafe_files": skipped,
        "tokenizer": "gpt2 (tiktoken), <|endoftext|> before every document",
        "dtype": "uint16",
        "val_tokens": VAL_TOKENS,
        "val_documents": val_docs,
        "train_tokens": TRAIN_TOKENS,
        "train_documents": train_docs,
    }
    (OUT_DIR / "info.json").write_text(json.dumps(info, indent=2))
    print(f"done: {val_docs:,} val and {train_docs:,} train documents in {OUT_DIR}")

#under spawn or forkserver, every worker re-imports this file. W/o this  each worker would call main(), 
# which creates a Pool which will create more workers. Python will throw error as it detets this.
# Inside a worker, __name__ is not "__main__", so this skips main(). The workers still see the module-level enc and EOT and the tokenize function, 
# which are the only things they need.
if __name__ == "__main__":
    main()



#More on Pool
# Pool is a fixed team of reusable worker processes, plus the machinery to hand them work and collect what they produce.

# The problem it solves

# To use several CPUs from Python, you need several processes, because the GIL limits each process to one CPU of Python work. The raw tool for that is multiprocessing.Process: you start a process, give it a function, and wait for it to finish. Doing that by hand for your job raises three problems:

# Startup cost. Starting a process takes milliseconds, and each new one would have to set up its own tokenizer. You have millions of documents, so one process per document would spend most of its time starting processes.
# Distribution. Something has to decide which process gets which document, and keep every process busy.
# Collection. Results come back from separate memory spaces, and something has to gather them and match each one to its input.

# Pool solves all three once, so we don't have to.

# What it does
# Starts N workers once and keeps them alive. Pool(7) launches 7 processes when the with block begins. Each one then loops, taking a task, running it, and sending back the result, until the pool shuts down. Startup cost is paid 7 times in total, not millions of times.
# Uses a shared task queue. Tasks go into one queue that all workers read from, and a worker takes the next task when it becomes free. Nobody assigns documents in advance. That's automatic load balancing: a worker stuck on a huge document doesn't hold the others up, because they keep pulling tasks. It's like a bank with one queue for all tellers, rather than one line per teller.
# Moves data between processes for us. It pickles tasks into the workers' pipes and unpickles results on the way back. Those are the helper threads in your main process from the earlier diagram.
# Matches results to inputs. Every task carries an index, so results can be put back into input order even though workers finish in random order.
# The main methods
# Method	|Returns |	Order kept?|	Memory
# map(f, items)}|	a full list, only after everything finishes	|yes|	all results at once
# imap(f, items, chunksize)|	an iterator, results as they arrive|	yes|	streams
# imap_unordered(f, items, chunksize)|	an iterator|	no, completion order	|streams, slightly faster
# apply_async(f, args)|	a handle for one task|	n/a|	one result

# this script uses imap. It needs streaming, because 7.44B tokens of results can't sit in one list, and it needs input order, because that makes the val/train split reproducible.

# Shutting down
# close(): no more tasks will be submitted; workers finish what's queued and then exit.
# join(): wait for them to exit. It must come after close() or terminate().
# terminate(): kill the workers immediately, abandoning unfinished tasks. This is what leaving a with Pool(...) block does, which is fine for this script because all needed results are already consumed.