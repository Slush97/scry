#!/usr/bin/env python3
"""Trace Whisper-tiny decoder for a 2s 440Hz sine wave, step by step from safetensors."""

import torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
import whisper
import math

torch.set_grad_enabled(False)

# ── Load weights ──────────────────────────────────────────────────────
f = safe_open('/home/esoc/scry/crates/scry-stt/models/whisper-tiny/model.safetensors', framework='pt')
state = {k: f.get_tensor(k) for k in f.keys()}

# ── Model hyperparams ────────────────────────────────────────────────
d_model = 384
n_heads = 6
head_dim = d_model // n_heads  # 64
n_decoder_layers = 4
scale = 1.0 / math.sqrt(head_dim)

# ── Generate audio ────────────────────────────────────────────────────
sr = 16000
t = np.arange(2 * sr) / sr
audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
audio_padded = whisper.pad_or_trim(audio)

# ── Mel spectrogram ──────────────────────────────────────────────────
mel = whisper.log_mel_spectrogram(audio_padded)  # [80, 3000]
print(f"Mel shape: {list(mel.shape)}")

# ── Run encoder (same as trace_encoder.py) ───────────────────────────
x = mel.unsqueeze(0).float()  # [1, 80, 3000]

# Conv1 + GELU
x = F.gelu(F.conv1d(x, state['model.encoder.conv1.weight'], state['model.encoder.conv1.bias'], padding=1))
# Conv2 + GELU
x = F.gelu(F.conv1d(x, state['model.encoder.conv2.weight'], state['model.encoder.conv2.bias'], stride=2, padding=1))
# Transpose + positional embedding
x = x.permute(0, 2, 1) + state['model.encoder.embed_positions.weight']

# Encoder layers
for li in range(4):  # whisper-tiny has 4 encoder layers
    p = f'model.encoder.layers.{li}'
    # Self-attention
    res = x
    xn = F.layer_norm(x, [d_model], state[f'{p}.self_attn_layer_norm.weight'], state[f'{p}.self_attn_layer_norm.bias'])
    Q = F.linear(xn, state[f'{p}.self_attn.q_proj.weight'], state[f'{p}.self_attn.q_proj.bias'])
    K = F.linear(xn, state[f'{p}.self_attn.k_proj.weight'])
    V = F.linear(xn, state[f'{p}.self_attn.v_proj.weight'], state[f'{p}.self_attn.v_proj.bias'])
    sl = x.shape[1]
    Q = Q.view(1, sl, n_heads, head_dim).transpose(1, 2)
    K = K.view(1, sl, n_heads, head_dim).transpose(1, 2)
    V = V.view(1, sl, n_heads, head_dim).transpose(1, 2)
    aw = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * scale, dim=-1)
    ao = torch.matmul(aw, V).transpose(1, 2).contiguous().view(1, sl, d_model)
    ao = F.linear(ao, state[f'{p}.self_attn.out_proj.weight'], state[f'{p}.self_attn.out_proj.bias'])
    x = res + ao
    # MLP
    res = x
    xn = F.layer_norm(x, [d_model], state[f'{p}.final_layer_norm.weight'], state[f'{p}.final_layer_norm.bias'])
    h = F.gelu(F.linear(xn, state[f'{p}.fc1.weight'], state[f'{p}.fc1.bias']))
    h = F.linear(h, state[f'{p}.fc2.weight'], state[f'{p}.fc2.bias'])
    x = res + h

# Final encoder layer norm
encoder_out = F.layer_norm(x, [d_model], state['model.encoder.layer_norm.weight'], state['model.encoder.layer_norm.bias'])
print(f"Encoder output shape: {list(encoder_out.shape)}")
print(f"Encoder output first 5: {encoder_out[0, 0, :5].tolist()}")

# ── Pre-compute cross-attention K,V for each decoder layer ───────────
# These are constant across all decoder steps
cross_kv = []
for li in range(n_decoder_layers):
    p = f'model.decoder.layers.{li}'
    cK = F.linear(encoder_out, state[f'{p}.encoder_attn.k_proj.weight'])  # no bias
    cV = F.linear(encoder_out, state[f'{p}.encoder_attn.v_proj.weight'], state[f'{p}.encoder_attn.v_proj.bias'])
    enc_sl = encoder_out.shape[1]
    cK = cK.view(1, enc_sl, n_heads, head_dim).transpose(1, 2)  # [1, n_heads, 1500, 64]
    cV = cV.view(1, enc_sl, n_heads, head_dim).transpose(1, 2)
    cross_kv.append((cK, cV))

print(f"\nCross-attn KV precomputed for {n_decoder_layers} layers")
print(f"Cross K shape per layer: {list(cross_kv[0][0].shape)}")

# ── Decoder: process tokens one at a time ────────────────────────────
# Prompt tokens: SOT=50258, en=50259, transcribe=50359, notimestamps=50363
tokens = [50258, 50259, 50359, 50363]
token_emb = state['model.decoder.embed_tokens.weight']  # [51864, 384]
pos_emb = state['model.decoder.embed_positions.weight']  # [448, 384]

# KV cache for self-attention: list of (K_cache, V_cache) per layer
# Each starts empty, grows by 1 each step
self_kv_cache = [None] * n_decoder_layers  # will be (K, V) tensors

print("\n" + "=" * 70)
print("DECODER TRACE -- feeding tokens one at a time")
print("=" * 70)

for step, tok in enumerate(tokens):
    print(f"\n--- Step {step}: token {tok} ---")

    # Token embedding + positional embedding
    x = token_emb[tok].unsqueeze(0).unsqueeze(0)  # [1, 1, 384]
    x = x + pos_emb[step].unsqueeze(0).unsqueeze(0)  # position = step
    
    print(f"  After embed+pos first 5: {x[0, 0, :5].tolist()}")

    for li in range(n_decoder_layers):
        p = f'model.decoder.layers.{li}'

        # ── Self-attention ──
        res = x
        xn = F.layer_norm(x, [d_model],
                          state[f'{p}.self_attn_layer_norm.weight'],
                          state[f'{p}.self_attn_layer_norm.bias'])

        # Q for current position only
        Q = F.linear(xn, state[f'{p}.self_attn.q_proj.weight'], state[f'{p}.self_attn.q_proj.bias'])
        # K, V for current position
        K_new = F.linear(xn, state[f'{p}.self_attn.k_proj.weight'])
        V_new = F.linear(xn, state[f'{p}.self_attn.v_proj.weight'], state[f'{p}.self_attn.v_proj.bias'])

        # Append to cache
        if self_kv_cache[li] is None:
            K_all = K_new
            V_all = V_new
        else:
            K_all = torch.cat([self_kv_cache[li][0], K_new], dim=1)
            V_all = torch.cat([self_kv_cache[li][1], V_new], dim=1)
        self_kv_cache[li] = (K_all, V_all)

        # Reshape for multi-head
        kv_len = K_all.shape[1]
        Q_h = Q.view(1, 1, n_heads, head_dim).transpose(1, 2)       # [1, 6, 1, 64]
        K_h = K_all.view(1, kv_len, n_heads, head_dim).transpose(1, 2)  # [1, 6, kv_len, 64]
        V_h = V_all.view(1, kv_len, n_heads, head_dim).transpose(1, 2)

        # Causal mask: for autoregressive, current query can attend to all cached positions
        # Since we only have Q for current pos and K for all past+current, no mask needed
        attn_w = F.softmax(torch.matmul(Q_h, K_h.transpose(-2, -1)) * scale, dim=-1)
        attn_o = torch.matmul(attn_w, V_h)
        attn_o = attn_o.transpose(1, 2).contiguous().view(1, 1, d_model)
        attn_o = F.linear(attn_o, state[f'{p}.self_attn.out_proj.weight'], state[f'{p}.self_attn.out_proj.bias'])
        x = res + attn_o

        # ── Cross-attention ──
        res = x
        xn = F.layer_norm(x, [d_model],
                          state[f'{p}.encoder_attn_layer_norm.weight'],
                          state[f'{p}.encoder_attn_layer_norm.bias'])

        cQ = F.linear(xn, state[f'{p}.encoder_attn.q_proj.weight'], state[f'{p}.encoder_attn.q_proj.bias'])
        cQ_h = cQ.view(1, 1, n_heads, head_dim).transpose(1, 2)  # [1, 6, 1, 64]

        cK_h, cV_h = cross_kv[li]  # [1, 6, 1500, 64]

        cattn_w = F.softmax(torch.matmul(cQ_h, cK_h.transpose(-2, -1)) * scale, dim=-1)
        cattn_o = torch.matmul(cattn_w, cV_h)
        cattn_o = cattn_o.transpose(1, 2).contiguous().view(1, 1, d_model)
        cattn_o = F.linear(cattn_o, state[f'{p}.encoder_attn.out_proj.weight'], state[f'{p}.encoder_attn.out_proj.bias'])
        x = res + cattn_o

        # ── MLP ──
        res = x
        xn = F.layer_norm(x, [d_model],
                          state[f'{p}.final_layer_norm.weight'],
                          state[f'{p}.final_layer_norm.bias'])
        h = F.gelu(F.linear(xn, state[f'{p}.fc1.weight'], state[f'{p}.fc1.bias']))
        h = F.linear(h, state[f'{p}.fc2.weight'], state[f'{p}.fc2.bias'])
        x = res + h

        if li == 0 and step == len(tokens) - 1:
            print(f"  After decoder layer 0: first 5 = {x[0, 0, :5].tolist()}")

    # After all layers
    if step == len(tokens) - 1:
        print(f"  After all decoder layers: first 5 = {x[0, 0, :5].tolist()}")

    # Final layer norm
    hidden = F.layer_norm(x, [d_model],
                          state['model.decoder.layer_norm.weight'],
                          state['model.decoder.layer_norm.bias'])

    print(f"  Hidden (after final LN) first 5: {hidden[0, 0, :5].tolist()}")
    print(f"  Hidden min={hidden.min().item():.6f}, max={hidden.max().item():.6f}, mean={hidden.mean().item():.6f}")

    # Logits = hidden @ token_embedding.T
    logits = F.linear(hidden, token_emb)  # [1, 1, 51864]
    logits = logits.squeeze()  # [51864]

    print(f"  Logits first 5: {logits[:5].tolist()}")
    print(f"  Logits min={logits.min().item():.6f}, max={logits.max().item():.6f}")
    print(f"  Logits argmax={logits.argmax().item()}, value={logits.max().item():.6f}")

    # Only print detailed info for the last prompt token
    if step == len(tokens) - 1:
        print(f"\n{'=' * 70}")
        print(f"DETAILED OUTPUT AFTER LAST PROMPT TOKEN (50363 = notimestamps)")
        print(f"{'=' * 70}")

        print(f"\nHidden state (after final LN, before logit projection):")
        print(f"  first 5: {hidden[0, 0, :5].tolist()}")
        print(f"  min={hidden.min().item():.6f}, max={hidden.max().item():.6f}, mean={hidden.mean().item():.6f}")

        print(f"\nFinal logits:")
        print(f"  first 5: {logits[:5].tolist()}")
        print(f"  min={logits.min().item():.6f}, max={logits.max().item():.6f}")
        print(f"  argmax={logits.argmax().item()}, value at argmax={logits[logits.argmax()].item():.6f}")

        print(f"\nTop 10 tokens:")
        topk = torch.topk(logits, 10)
        for i in range(10):
            tid = topk.indices[i].item()
            val = topk.values[i].item()
            print(f"  rank {i}: token {tid}, logit {val:.6f}")

        print(f"\nLogit values at specific tokens:")
        for tid in [50256, 50257, 50258, 50363, 50613]:
            print(f"  token {tid}: {logits[tid].item():.6f}")

        print(f"\nText token logit stats (tokens 0-50256):")
        text_logits = logits[:50257]
        print(f"  min={text_logits.min().item():.6f}, max={text_logits.max().item():.6f}, mean={text_logits.mean().item():.6f}")
        text_positive = (text_logits > 0).sum().item()
        print(f"  # positive: {text_positive} / {len(text_logits)}")
        if text_positive > 0:
            top_text = torch.topk(text_logits, min(10, text_positive))
            print(f"  Top text tokens with positive logits:")
            for i in range(len(top_text.indices)):
                tid = top_text.indices[i].item()
                val = top_text.values[i].item()
                print(f"    token {tid}: {val:.6f}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
