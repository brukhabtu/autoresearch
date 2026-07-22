"""
TinyStories data prep for the CPU toy net (separate from prepare.py, which is
left untouched). Downloads karpathy/tinystories-gpt4-clean and trains a SMALL
BPE tokenizer on it — simple English needs far fewer merges than climbmix, and
a smaller vocab keeps the model lean and fast on CPU.

Data + tokenizer live in ~/.cache/autoresearch/tinystories/.
Run: uv run prepare_stories.py
"""

import os
import sys
import time
import pickle

import requests
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

from prepare import SPLIT_PATTERN, SPECIAL_TOKENS, BOS_TOKEN

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "tinystories")
DATA_DIR = os.path.join(CACHE, "data")
TOK_DIR = os.path.join(CACHE, "tokenizer")
PARQUET = os.path.join(DATA_DIR, "tinystories_gpt4_clean.parquet")
URL = "https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean/resolve/main/tinystories_gpt4_clean.parquet"

VOCAB_SIZE = 2048          # small vocab for simple English
VAL_STORIES = 2000         # first N stories held out for validation
TOK_TRAIN_CHARS = 100_000_000   # chars used to train the tokenizer

_TEXT_CANDIDATES = ("text", "story", "content", "completion", "output")


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(PARQUET):
        print(f"Data: already present ({os.path.getsize(PARQUET)/1e6:.0f} MB)")
        return
    print(f"Data: downloading {URL} ...")
    tmp = PARQUET + ".tmp"
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    os.rename(tmp, PARQUET)
    print(f"Data: done ({os.path.getsize(PARQUET)/1e6:.0f} MB)")


def _text_column(schema):
    names = [f.name for f in schema]
    for cand in _TEXT_CANDIDATES:
        if cand in names:
            return cand
    # fall back to first string-typed field
    for f in schema:
        if "string" in str(f.type) or "utf8" in str(f.type):
            return f.name
    raise RuntimeError(f"No text column found among {names}")


def _numbered_stories():
    pf = pq.ParquetFile(PARQUET)
    col = _text_column(pf.schema_arrow)
    idx = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=[col])
        for s in tbl.column(col).to_pylist():
            yield idx, s
            idx += 1


def story_iterator(split, max_chars=None):
    """Yield story texts for 'train' or 'val'. Val = first VAL_STORIES stories."""
    assert split in ("train", "val")
    nchars = 0
    for idx, s in _numbered_stories():
        is_val = idx < VAL_STORIES
        if (split == "val") != is_val:
            continue
        if not s:
            continue
        yield s
        if max_chars is not None:
            nchars += len(s)
            if nchars >= max_chars:
                return


def train_tokenizer():
    tok_pkl = os.path.join(TOK_DIR, "tokenizer.pkl")
    tb_path = os.path.join(TOK_DIR, "token_bytes.pt")
    if os.path.exists(tok_pkl) and os.path.exists(tb_path):
        print(f"Tokenizer: already trained at {TOK_DIR}")
        return
    os.makedirs(TOK_DIR, exist_ok=True)

    print(f"Tokenizer: training BPE (vocab={VOCAB_SIZE}) on up to {TOK_TRAIN_CHARS/1e6:.0f}M chars...")
    t0 = time.time()
    tok = rustbpe.Tokenizer()
    tok.train_from_iterator(story_iterator("train", max_chars=TOK_TRAIN_CHARS),
                            VOCAB_SIZE - len(SPECIAL_TOKENS), pattern=SPLIT_PATTERN)
    mergeable = {bytes(k): v for k, v in tok.get_mergeable_ranks()}
    offset = len(mergeable)
    specials = {name: offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(name="tinystories", pat_str=tok.get_pattern(),
                            mergeable_ranks=mergeable, special_tokens=specials)
    with open(tok_pkl, "wb") as f:
        pickle.dump(enc, f)
    print(f"Tokenizer: trained in {time.time()-t0:.1f}s (vocab={enc.n_vocab})")

    special_set = set(SPECIAL_TOKENS)
    tb = [0 if enc.decode([i]) in special_set else len(enc.decode([i]).encode("utf-8"))
          for i in range(enc.n_vocab)]
    torch.save(torch.tensor(tb, dtype=torch.int32), tb_path)

    test = "Once upon a time, there was a little cat."
    assert enc.decode(enc.encode_ordinary(test)) == test, "roundtrip failed"
    print(f"Tokenizer: token_bytes + sanity check ok -> {TOK_DIR}")


if __name__ == "__main__":
    print(f"Cache: {CACHE}\n")
    download()
    print()
    train_tokenizer()
    print("\nDone! Ready to train with train_stories.py")
