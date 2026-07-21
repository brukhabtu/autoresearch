"""
CPU TOY FORK of train.py — for demonstration only, NOT a valid experiment.

Why this file exists:
  train.py is hard-bound to CUDA + FlashAttention-3 and cannot run without an
  NVIDIA GPU. This fork runs the *same architecture and optimizer* on CPU so you
  can watch the full setup -> train -> evaluate pipeline execute end to end.

What is DIFFERENT from the canonical run (and why the number is NOT comparable):
  - Device is CPU, everything in float32 (no bf16 autocast, no fused kernels).
  - FlashAttention-3 replaced by torch's scaled_dot_product_attention.
  - torch.compile disabled (eager mode).
  - Tiny model (DEPTH=2, dim=128, seq=256) so a CPU can do a few steps.
  - TIME_BUDGET and the eval are shrunk to finish in a minute or two on CPU.
    The real evaluate_bpb() streams ~21M tokens at seq-len 2048 — hours on CPU.

prepare.py and train.py are left completely untouched.
"""

import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the real, unmodified data/tokenizer plumbing from prepare.py.
# (get_token_bytes already supports device="cpu"; the dataloader/eval below are
#  CPU reimplementations of prepare's CUDA-bound versions with identical math.)
from prepare import Tokenizer, get_token_bytes, _document_batches

# ---------------------------------------------------------------------------
# Demo constants (shrunk for CPU — this is why results are not canonical)
# ---------------------------------------------------------------------------

SEQ_LEN = 256            # canonical is MAX_SEQ_LEN=2048
DEVICE_BATCH_SIZE = 8
TOTAL_BATCH_SIZE = SEQ_LEN * DEVICE_BATCH_SIZE  # 1 grad-accum step
TIME_BUDGET = 60         # canonical is 300s
EVAL_STEPS = 20          # canonical streams ~21M tokens; we sample a little

DEPTH = 2
device = torch.device("cpu")

# ---------------------------------------------------------------------------
# Model (mirrors train.py, fp32 + SDPA instead of FA3)
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = SEQ_LEN
    vocab_size: int = 8192
    n_layer: int = DEPTH
    n_head: int = 2
    n_kv_head: int = 2
    n_embd: int = 128
    window_pattern: str = "L"   # all-full attention (README's small-compute tip)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B,H,T,D)
        left = window  # FA3 window_size[0]: attend keys in [i-left, i]
        if left is None or left <= 0 or left >= T:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            idx = torch.arange(T)
            diff = idx[:, None] - idx[None, :]          # i - j
            allowed = (diff >= 0) & (diff <= left)
            attn_mask = torch.zeros(T, T, dtype=q.dtype).masked_fill(~allowed, float("-inf"))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window):
        x = x + self.attn(norm(x), ve, cos_sin, window)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        long_window = config.sequence_len
        short_window = long_window // 2
        c2w = {"L": long_window, "S": short_window}
        p = config.window_pattern.upper()
        self.windows = [c2w[p[i % len(p)]] for i in range(config.n_layer)]
        self.windows[-1] = long_window
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        cos, sin = self._rotary(config.sequence_len * 10, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3 ** 0.5 * n_embd ** -0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

    def _rotary(self, seq_len, head_dim, base=10000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return cos[None, :, None, :], sin[None, :, None, :]

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.6, matrix_lr=0.04,
                        weight_decay=0.2, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=[self.resid_lambdas], lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=[self.x0_lambdas], lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind='muon', params=group_params, lr=matrix_lr,
                                     momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.windows[i])
        x = norm(x)
        softcap = 15
        logits = self.lm_head(x).float()
        logits = softcap * torch.tanh(logits / softcap)
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
        return logits


# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, torch.compile stripped for CPU eager execution)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps, wd):
    p.mul_(1 - lr * wd)
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)
    bias1 = 1 - beta1 ** step
    bias2 = 1 - beta2 ** step
    denom = (exp_avg_sq / bias2).sqrt() + eps
    p.add_(exp_avg / denom, alpha=-(lr / bias1))


def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum, lr, wd, beta2, ns_steps, red_dim):
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.float()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            Bm = b * A + c * (A @ A)
            X = a * X + X @ Bm
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            Bm = b * A + c * (A @ A)
            X = a * X + Bm @ X
    g = X
    v_mean = g.square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
    second_momentum_buffer.lerp_(v_mean, 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            adamw_step_fused(p, p.grad, state['exp_avg'], state['exp_avg_sq'],
                             state['step'], group['lr'], group['betas'][0],
                             group['betas'][1], group['eps'], group['weight_decay'])

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device_, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device_)
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device_)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([q.grad for q in params])
        stacked_params = torch.stack(params)
        lr = group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5
        muon_step_fused(stacked_grads, stacked_params, state["momentum_buffer"],
                        state["second_momentum_buffer"], group["momentum"], lr,
                        group["weight_decay"], group["beta2"], group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)


# ---------------------------------------------------------------------------
# CPU dataloader + reduced eval (CPU reimplementations of prepare.py's versions)
# ---------------------------------------------------------------------------

def make_cpu_dataloader(tokenizer, B, T, split, buffer_size=500):
    row_capacity = T + 1
    batches = _document_batches(split)
    bos = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)

    def refill():
        nonlocal epoch
        db, epoch = next(batches)
        doc_buffer.extend(tokenizer.encode(db, prepend=bos))

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
        yield row_buffer[:, :-1].contiguous(), row_buffer[:, 1:].contiguous(), epoch


@torch.no_grad()
def evaluate_bpb_cpu(model, tokenizer, batch_size, seq_len, eval_steps):
    token_bytes = get_token_bytes(device="cpu")
    val_loader = make_cpu_dataloader(tokenizer, batch_size, seq_len, "val")
    total_nats, total_bytes = 0.0, 0
    for _ in range(eval_steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)


# ---------------------------------------------------------------------------
# Setup + training loop
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")

    config = GPTConfig(vocab_size=vocab_size)
    print(f"Model config: {asdict(config)}")
    model = GPT(config)
    model.init_weights()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"num_params_M: {num_params / 1e6:.2f}")

    optimizer = model.setup_optimizer()
    train_loader = make_cpu_dataloader(tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, "train")
    x, y, epoch = next(train_loader)

    print(f"Time budget: {TIME_BUDGET}s (DEMO — not the canonical 300s)")
    print("Training...")

    t_start_training = time.time()
    smooth = 0.0
    total_training_time = 0.0
    step = 0
    warmdown = 0.5

    while True:
        t0 = time.time()
        loss = model(x, y)
        train_loss = loss.detach().item()
        loss.backward()
        x, y, epoch = next(train_loader)

        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = 1.0 if progress < 1.0 - warmdown else (1.0 - progress) / warmdown
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        optimizer.step()
        model.zero_grad(set_to_none=True)

        if math.isnan(train_loss) or train_loss > 100:
            print("\nFAIL: loss exploded")
            return

        dt = time.time() - t0
        if step > 2:
            total_training_time += dt
        smooth = 0.9 * smooth + 0.1 * train_loss
        debiased = smooth / (1 - 0.9 ** (step + 1))
        remaining = max(0, TIME_BUDGET - total_training_time)
        print(f"\rstep {step:04d} ({100*progress:.0f}%) | loss: {debiased:.4f} | "
              f"lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | epoch {epoch} | remaining: {remaining:.0f}s   ",
              end="", flush=True)

        step += 1
        if step > 2 and total_training_time >= TIME_BUDGET:
            break
    print()

    total_tokens = step * TOTAL_BATCH_SIZE
    model.eval()
    print(f"Evaluating (reduced: {EVAL_STEPS} steps of {DEVICE_BATCH_SIZE}x{SEQ_LEN} tokens)...")
    val_bpb = evaluate_bpb_cpu(model, tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, EVAL_STEPS)

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}   (CPU toy — NOT comparable to GPU runs)")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {time.time() - t_start:.1f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.3f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.2f}")
    print(f"depth:            {DEPTH}")


if __name__ == "__main__":
    main()
