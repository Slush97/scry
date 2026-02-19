#!/usr/bin/env python3
"""PyTorch GPT-2 training step benchmark — matches scry-llm's mfu_bench config.

GPT-2: d=768, h=12, L=6, d_ff=3072, V=50257, seq=128
Reports tok/s and MFU for batch=4 and batch=32 in both fp32 and bf16.
"""

import time
import torch
import torch.nn as nn

VOCAB = 50257
D_MODEL = 768
N_HEADS = 12
N_LAYERS = 6
D_FF = 3072
SEQ_LEN = 128
PEAK_FP32 = 23.1  # RTX 5070 Ti
N_WARMUP = 3
N_STEPS = 10


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, D_FF)
        self.fc2 = nn.Linear(D_FF, D_MODEL)

    def forward(self, x):
        h = self.ln1(x)
        mask = nn.Transformer.generate_square_subsequent_mask(h.size(1), device=h.device)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + h
        h = self.ln2(x)
        h = nn.functional.gelu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class GPT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos_emb = nn.Embedding(1024, D_MODEL)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB, bias=False)
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, input_ids, targets):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = nn.functional.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))
        return loss


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def bench(batch_size, dtype, compile_model=False):
    device = "cuda"
    model = GPT2().to(device)
    if dtype == torch.bfloat16:
        model = model.to(dtype)
    if compile_model:
        model = torch.compile(model)

    n_params = count_params(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    dtype_label = "bf16" if dtype == torch.bfloat16 else "fp32"
    compiled_label = "+compile" if compile_model else ""
    label = f"PyTorch {dtype_label}{compiled_label} batch={batch_size}"

    def make_batch():
        ids = torch.randint(0, VOCAB, (batch_size, SEQ_LEN), device=device)
        tgt = torch.randint(0, VOCAB, (batch_size, SEQ_LEN), device=device)
        return ids, tgt

    # Warmup
    for _ in range(N_WARMUP):
        ids, tgt = make_batch()
        with torch.cuda.amp.autocast(enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16):
            loss = model(ids, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Timed
    total_loss = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        ids, tgt = make_batch()
        with torch.cuda.amp.autocast(enabled=(dtype == torch.bfloat16), dtype=torch.bfloat16):
            loss = model(ids, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens = N_STEPS * batch_size * SEQ_LEN
    tok_s = tokens / elapsed
    flops_per_tok = 6.0 * n_params
    achieved = tok_s * flops_per_tok / 1e12
    mfu = achieved / PEAK_FP32 * 100.0

    print(f"────────────────────────────────────────")
    print(f"[{label}] n_params={n_params}")
    print(f"[{label}] {N_STEPS} steps in {elapsed:.3f}s")
    print(f"[{label}] avg loss: {total_loss / N_STEPS:.4f}")
    print(f"[{label}] tok/s: {tok_s:.0f}")
    print(f"[{label}] achieved: {achieved:.2f} TFLOPS")
    print(f"[{label}] MFU: {mfu:.1f}% (vs {PEAK_FP32} FP32 peak)")
    print(f"────────────────────────────────────────")
    print()


if __name__ == "__main__":
    print(f"=== PyTorch GPT-2 MFU Benchmark ===")
    print(f"Config: d={D_MODEL}, h={N_HEADS}, L={N_LAYERS}, d_ff={D_FF}, V={VOCAB}")
    print(f"seq={SEQ_LEN}, warmup={N_WARMUP}, steps={N_STEPS}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # FP32
    bench(4, torch.float32)
    bench(32, torch.float32)

    # BF16 (autocast)
    bench(4, torch.bfloat16)
    bench(32, torch.bfloat16)

    # BF16 + torch.compile
    bench(4, torch.bfloat16, compile_model=True)
    bench(32, torch.bfloat16, compile_model=True)
