"""
PyTorch baseline benchmark for scry-llm comparison.

Runs GPT-2 training at two configs:
  1. Small  — d_model=256, n_heads=4, n_layers=4,  batch=4, seq=128  (matches scry-llm quick bench)
  2. Target — d_model=768, n_heads=12, n_layers=12, batch=4, seq=128  (124M, memory-safe batch)

Reports tok/s and MFU for each config at FP32, BF16, and BF16+torch.compile.
RTX 5070 Ti BF16 dense peak: 88 TFLOPS.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

GPU_PEAK_TFLOPS = 88.0  # RTX 5070 Ti BF16 dense tensor core throughput


class GPT2Config:
    def __init__(self, d_model, n_heads, n_layers, vocab_size=50257, max_seq_len=1024):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = 4 * d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // n_heads
        self.dropout = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight
        self.cfg = cfg

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def n_params(self):
        # Exclude position embeddings from count (standard practice)
        return sum(p.numel() for p in self.parameters()) - self.pos_emb.weight.numel()


def benchmark_config(label, cfg, batch_size, seq_len, dtype, compile_model, total_steps=50, warmup_steps=5):
    device = torch.device("cuda")
    torch.manual_seed(42)

    model = GPT2(cfg).to(device)
    if dtype == torch.bfloat16:
        model = model.to(dtype)
    n_params = model.n_params()

    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.1)

    # Synthetic data
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len + 1), device=device)

    tokens_per_step = batch_size * seq_len

    # Warmup
    for _ in range(warmup_steps):
        x = input_ids[:, :seq_len]
        y = input_ids[:, 1:seq_len + 1]
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    losses = []
    for step in range(total_steps):
        x = input_ids[:, :seq_len]
        y = input_ids[:, 1:seq_len + 1]
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tok_per_sec = (total_steps * tokens_per_step) / elapsed
    flops_per_step = 6 * n_params * tokens_per_step
    effective_tflops = (flops_per_step * total_steps) / elapsed / 1e12
    mfu = effective_tflops / GPU_PEAK_TFLOPS * 100

    dtype_label = {torch.float32: "FP32", torch.bfloat16: "BF16"}[dtype]
    compiled_label = "+compile" if compile_model else ""

    print(f"\n{'='*60}")
    print(f"  {label} | {dtype_label}{compiled_label}")
    print(f"{'='*60}")
    print(f"  Params:          {n_params/1e6:.1f}M")
    print(f"  Config:          d={cfg.d_model} h={cfg.n_heads} L={cfg.n_layers} B={batch_size} T={seq_len}")
    print(f"  Steps:           {total_steps} ({elapsed:.2f}s)")
    print(f"  Throughput:      {tok_per_sec:,.0f} tok/s")
    print(f"  Effective:       {effective_tflops:.1f} TFLOPS")
    print(f"  MFU:             {mfu:.1f}%")
    print(f"  Final loss:      {losses[-1]:.4f}")
    print(f"{'='*60}")

    return tok_per_sec, mfu


def main():
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA:   {torch.cuda.get_device_name(0)}")
    print(f"Peak:   {GPU_PEAK_TFLOPS} TFLOPS (BF16 dense)")
    print()

    # --- Config 1: Small (matches scry-llm quick bench) ---
    small_cfg = GPT2Config(d_model=256, n_heads=4, n_layers=4)
    batch, seq = 4, 128

    benchmark_config("Small model", small_cfg, batch, seq, torch.float32, False)
    benchmark_config("Small model", small_cfg, batch, seq, torch.bfloat16, False)
    benchmark_config("Small model", small_cfg, batch, seq, torch.bfloat16, True)

    # --- Config 2: GPT-2 124M (design doc target) ---
    target_cfg = GPT2Config(d_model=768, n_heads=12, n_layers=12)

    benchmark_config("GPT-2 124M", target_cfg, batch, seq, torch.float32, False)
    benchmark_config("GPT-2 124M", target_cfg, batch, seq, torch.bfloat16, False)
    benchmark_config("GPT-2 124M", target_cfg, batch, seq, torch.bfloat16, True)

    # --- Config 3: GPT-2 124M at production batch (if memory allows) ---
    try:
        torch.cuda.empty_cache()
        benchmark_config("GPT-2 124M (big batch)", target_cfg, 16, 1024, torch.bfloat16, True,
                         total_steps=20, warmup_steps=3)
    except torch.cuda.OutOfMemoryError:
        print("\nOOM on batch=16, seq=1024 — trying batch=8, seq=512")
        torch.cuda.empty_cache()
        try:
            benchmark_config("GPT-2 124M (med batch)", target_cfg, 8, 512, torch.bfloat16, True,
                             total_steps=20, warmup_steps=3)
        except torch.cuda.OutOfMemoryError:
            print("OOM on fallback too — skipping large batch test")


if __name__ == "__main__":
    main()
