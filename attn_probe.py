"""
Attention probe for the CPU toy net.

Trains the same tiny model as train_cpu.py, and at several wall-clock
snapshots captures the model's REAL attention distributions on a fixed probe
sentence — recomputing softmax(QK^T / sqrt(head_dim) + causal mask) from the
actual c_q/c_k weights, with RoPE + QK-norm applied exactly as in the forward
pass. Writes everything to a JSON the attention artifact renders.

Run: uv run attn_probe.py
"""

import json
import math
import time

import torch

from train_cpu import (
    GPT, GPTConfig, MuonAdamW, norm, apply_rotary_emb,
    make_cpu_dataloader, Tokenizer,
    SEQ_LEN, DEVICE_BATCH_SIZE, TOTAL_BATCH_SIZE,
)

OUT = "/tmp/claude-0/-home-user-autoresearch/0dbd08b6-ed5d-51b7-9018-63f7204a07df/scratchpad/attn_data.json"
PROBE_TEXT = "The quick brown fox jumps over the lazy dog."
BUDGET = 240.0                       # seconds of training
SNAP_SECS = [20.0, 75.0, 240.0]      # capture points (init captured separately)


@torch.no_grad()
def capture_attn(model, idx):
    """Return attn[layer][head] = TxT lower-triangular probability matrix."""
    B, T = idx.size()
    cos_sin = (model.cos[:, :T], model.sin[:, :T])
    cos, sin = cos_sin
    x = model.transformer.wte(idx)
    x = norm(x)
    x0 = x
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    layers = []
    for i, block in enumerate(model.transformer.h):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        attn = block.attn
        xin = norm(x)
        q = attn.c_q(xin).view(B, T, attn.n_head, attn.head_dim)
        k = attn.c_k(xin).view(B, T, attn.n_kv_head, attn.head_dim)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        qh, kh = q.transpose(1, 2), k.transpose(1, 2)          # (B, H, T, D)
        scores = (qh @ kh.transpose(-2, -1)) / (attn.head_dim ** 0.5)
        scores = scores.masked_fill(mask, float("-inf"))
        probs = scores.softmax(dim=-1)[0]                       # (H, T, T)
        heads = [[[round(float(probs[h, i2, j]), 4) for j in range(T)]
                  for i2 in range(T)] for h in range(probs.size(0))]
        layers.append(heads)
        # advance the real state so the next layer sees correct activations
        ve = model.value_embeds[str(i)](idx) if str(i) in model.value_embeds else None
        x = block(x, ve, cos_sin, model.windows[i])
    return layers


def main():
    torch.manual_seed(42)
    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()

    # probe input: BOS + sentence
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode(PROBE_TEXT, prepend=bos)
    idx = torch.tensor([ids], dtype=torch.long)
    tok_strs = ["<bos>"] + [tokenizer.decode([t]).replace(" ", "·") for t in ids[1:]]
    print(f"Probe: {len(ids)} tokens -> {tok_strs}")

    config = GPTConfig(vocab_size=vocab_size)
    model = GPT(config)
    model.init_weights()
    optimizer = model.setup_optimizer()
    train_loader = make_cpu_dataloader(tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, "train")
    x, y, _ = next(train_loader)

    snapshots = []

    def snap(label, step, secs, loss):
        snapshots.append({
            "label": label, "step": step, "secs": round(secs, 1),
            "loss": None if loss is None else round(loss, 3),
            "layers": capture_attn(model, idx),
        })
        print(f"  snapshot: {label} (step {step}, {secs:.0f}s, loss {loss})")

    snap("init", 0, 0.0, None)  # random weights, before any training

    smooth, total_time, step, snap_i = 0.0, 0.0, 0, 0
    warmdown = 0.5
    print("Training...")
    while True:
        t0 = time.time()
        loss = model(x, y)
        train_loss = loss.detach().item()
        loss.backward()
        x, y, _ = next(train_loader)
        progress = min(total_time / BUDGET, 1.0)
        lrm = 1.0 if progress < 1.0 - warmdown else (1.0 - progress) / warmdown
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] * lrm
        optimizer.step()
        model.zero_grad(set_to_none=True)
        if math.isnan(train_loss):
            print("NaN, aborting"); break
        dt = time.time() - t0
        if step > 2:
            total_time += dt
        smooth = 0.9 * smooth + 0.1 * train_loss
        debiased = smooth / (1 - 0.9 ** (step + 1))
        if step % 20 == 0:
            print(f"\rstep {step} | {total_time:.0f}s | loss {debiased:.3f}   ", end="", flush=True)
        step += 1
        while snap_i < len(SNAP_SECS) and total_time >= SNAP_SECS[snap_i]:
            lbl = f"{int(SNAP_SECS[snap_i])}s"
            print()
            snap(lbl, step, total_time, debiased)
            snap_i += 1
        if step > 2 and total_time >= BUDGET:
            break
    print()

    data = {"tokens": tok_strs, "probe_text": PROBE_TEXT,
            "n_layer": config.n_layer, "n_head": config.n_head,
            "snapshots": snapshots}
    with open(OUT, "w") as f:
        json.dump(data, f)
    print(f"Wrote {OUT}  ({len(snapshots)} snapshots, {len(tok_strs)} tokens)")


if __name__ == "__main__":
    main()
