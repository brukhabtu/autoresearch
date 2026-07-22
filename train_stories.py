"""
Train the CPU toy net on TinyStories, and SAMPLE from it so we can read what it
writes. Bigger than the toy (depth 4, dim 256) but small vocab (2048), which
keeps it fast on CPU. Prints generated stories at three points during training
so you can watch coherence emerge.

prepare_stories.py must be run first (downloads data, trains tokenizer).
Run: uv run train_stories.py
"""

import math
import time

import torch

from train_cpu import GPT, GPTConfig
from prepare import Tokenizer
from prepare_stories import story_iterator, TOK_DIR
import os

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEQ_LEN = 256
DEVICE_BATCH_SIZE = 16
TOTAL_BATCH_SIZE = SEQ_LEN * DEVICE_BATCH_SIZE
BUDGET = 1800.0                # seconds of training (30 min)
EVAL_STEPS = 40
SAMPLE_AT = [0.1, 0.35, 0.65]  # fractions of budget for mid-training samples
PROMPT = "Once upon a time"
CKPT = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "tinystories", "model.pt")

DEPTH, DIM, HEADS = 4, 256, 4


def get_token_bytes():
    return torch.load(os.path.join(TOK_DIR, "token_bytes.pt"), map_location="cpu")


def make_stories_dataloader(tokenizer, B, T, split, buffer_size=600):
    row_capacity = T + 1
    bos = tokenizer.get_bos_token_id()

    def story_batches():
        while True:                       # infinite epochs
            batch = []
            for s in story_iterator(split):
                batch.append(s)
                if len(batch) == 128:
                    yield batch; batch = []
            if batch:
                yield batch

    gen = story_batches()
    doc_buffer = []
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)

    def refill():
        doc_buffer.extend(tokenizer.encode(next(gen), prepend=bos))

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill()
                remaining = row_capacity - pos
                best_idx, best_len = -1, 0
                for i, doc in enumerate(doc_buffer):
                    dl = len(doc)
                    if dl <= remaining and dl > best_len:
                        best_idx, best_len = i, dl
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    si = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(si)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining
        yield row_buffer[:, :-1].contiguous(), row_buffer[:, 1:].contiguous()


@torch.no_grad()
def sample(model, tokenizer, prompt, max_new=180, temp=0.8, topk=40):
    was_training = model.training
    model.eval()
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode(prompt, prepend=bos)
    x = torch.tensor([ids], dtype=torch.long)
    for _ in range(max_new):
        ctx = x[:, -model.config.sequence_len:]
        logits = model(ctx)[0, -1] / temp
        if topk:
            v, _ = torch.topk(logits, min(topk, logits.size(-1)))
            logits = logits.masked_fill(logits < v[-1], float("-inf"))
        nxt = torch.multinomial(logits.softmax(-1), 1)
        if nxt.item() == bos:
            break
        x = torch.cat([x, nxt.view(1, 1)], dim=1)
    if was_training:
        model.train()
    txt = tokenizer.decode(x[0, 1:].tolist())   # drop BOS
    return txt


@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size, seq_len, steps):
    tb = get_token_bytes()
    loader = make_stories_dataloader(tokenizer, batch_size, seq_len, "val")
    total_nats, total_bytes = 0.0, 0
    for _ in range(steps):
        x, y = next(loader)
        loss = model(x, y, reduction="none").view(-1)
        yb = tb[y.view(-1)]
        mask = yb > 0
        total_nats += (loss * mask).sum().item()
        total_bytes += yb.sum().item()
    return total_nats / (math.log(2) * total_bytes)


def main():
    t_start = time.time()
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    tokenizer = Tokenizer.from_directory(TOK_DIR)
    vocab = tokenizer.get_vocab_size()
    print(f"Vocab: {vocab}")

    config = GPTConfig(sequence_len=SEQ_LEN, vocab_size=vocab, n_layer=DEPTH,
                       n_head=HEADS, n_kv_head=HEADS, n_embd=DIM, window_pattern="L")
    model = GPT(config)
    model.init_weights()
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Model: depth {DEPTH}, dim {DIM}, {nparams/1e6:.1f}M params")

    optimizer = model.setup_optimizer()
    loader = make_stories_dataloader(tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, "train")
    x, y = next(loader)

    print(f"Time budget: {BUDGET:.0f}s\nTraining...\n")
    smooth, total_time, step, warmdown = 0.0, 0.0, 0, 0.4
    sample_i = 0

    while True:
        t0 = time.time()
        loss = model(x, y)
        train_loss = loss.detach().item()
        loss.backward()
        x, y = next(loader)
        progress = min(total_time / BUDGET, 1.0)
        lrm = 1.0 if progress < 1.0 - warmdown else (1.0 - progress) / warmdown
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * lrm
        optimizer.step()
        model.zero_grad(set_to_none=True)
        if math.isnan(train_loss):
            print("\nNaN, aborting"); return
        dt = time.time() - t0
        if step > 2:
            total_time += dt
        smooth = 0.9 * smooth + 0.1 * train_loss
        debiased = smooth / (1 - 0.9 ** (step + 1))
        if step % 20 == 0:
            print(f"\rstep {step:04d} | {total_time:4.0f}s ({100*progress:3.0f}%) | loss {debiased:.3f}   ",
                  end="", flush=True)
        step += 1

        while sample_i < len(SAMPLE_AT) and progress >= SAMPLE_AT[sample_i]:
            print(f"\n\n--- sample @ {100*progress:.0f}%  (loss {debiased:.3f}) ---")
            print(f'"{PROMPT}{sample(model, tokenizer, PROMPT)}"')
            print("-" * 60)
            sample_i += 1

        if step > 2 and total_time >= BUDGET:
            break

    # Final (best) sample + checkpoint
    print(f"\n\n--- sample @ 100%  (loss {debiased:.3f}) ---")
    print(f'"{PROMPT}{sample(model, tokenizer, PROMPT)}"')
    print("-" * 60)
    torch.save({"state_dict": model.state_dict(),
                "config": {"sequence_len": SEQ_LEN, "vocab_size": vocab, "n_layer": DEPTH,
                           "n_head": HEADS, "n_kv_head": HEADS, "n_embd": DIM, "window_pattern": "L"}},
               CKPT)
    print(f"Saved checkpoint -> {CKPT}")

    print("\nEvaluating on held-out stories...")
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, EVAL_STEPS)
    print("---")
    print(f"val_bpb:          {val_bpb:.4f}")
    print(f"training_seconds: {total_time:.0f}")
    print(f"total_seconds:    {time.time()-t_start:.0f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {nparams/1e6:.1f}")


if __name__ == "__main__":
    main()
