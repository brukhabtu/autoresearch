"""Quick CPU throughput benchmark across model sizes. Times real fwd+bwd+opt
steps on random data (isolates compute) to find the largest size that still
trains at a practical step time. Run: uv run bench_sizes.py"""
import time, torch
from train_cpu import GPT, GPTConfig

torch.manual_seed(0)
VOCAB = 8192

CONFIGS = [
    # (name, depth, dim, heads, seq, batch)
    ("toy (current)", 2, 128, 2, 256, 8),
    ("small",         4, 256, 4, 256, 8),
    ("medium",        6, 384, 6, 512, 16),
    ("large",         8, 512, 8, 512, 16),
    ("xl",           10, 640, 10, 512, 16),
]

def bench(name, depth, dim, heads, seq, batch):
    cfg = GPTConfig(sequence_len=seq, vocab_size=VOCAB, n_layer=depth,
                    n_head=heads, n_kv_head=heads, n_embd=dim, window_pattern="L")
    model = GPT(cfg)
    model.init_weights()
    opt = model.setup_optimizer()
    nparams = sum(p.numel() for p in model.parameters())
    nmatrix = sum(p.numel() for p in model.transformer.h.parameters())
    x = torch.randint(0, VOCAB, (batch, seq))
    y = torch.randint(0, VOCAB, (batch, seq))
    # peak-ish memory estimate: params + grads + adam/muon state (~4x) in fp32
    approx_ram_gb = nparams * 4 * 5 / 1e9
    times = []
    for i in range(5):
        t0 = time.time()
        loss = model(x, y)
        loss.backward()
        opt.step()
        model.zero_grad(set_to_none=True)
        dt = time.time() - t0
        if i >= 2:
            times.append(dt)
    step = sum(times) / len(times)
    toks = batch * seq
    tps = toks / step
    steps_5min = int(300 / step)
    print(f"{name:16s} | d{depth:<2} dim{dim:<4} h{heads:<2} seq{seq:<4} b{batch:<3} "
          f"| {nparams/1e6:5.1f}M par ({nmatrix/1e6:4.1f}M matrix) "
          f"| {step*1000:6.0f} ms/step | {tps:6.0f} tok/s "
          f"| {toks/1000:4.0f}K tok/step | ~{steps_5min:4d} steps/5min "
          f"| ~{approx_ram_gb:.1f}GB")

print(f"threads={torch.get_num_threads()}\n")
for c in CONFIGS:
    try:
        bench(*c)
    except Exception as e:
        print(f"{c[0]:16s} | FAILED: {type(e).__name__}: {e}")
