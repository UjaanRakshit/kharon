#!/usr/bin/env python3
# PyTorch baseline for the Kharon single-GPU step, SAME proxy config (d512 x 8L, seq256,
# batch32, bf16). Reports tokens/sec for eager and (if available) torch.compile, so the
# README can state "hand-written C is Nx PyTorch eager / Mx compiled" on identical shapes.
import time, math, argparse
import torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--layers", type=int, default=8); p.add_argument("--d", type=int, default=512)
p.add_argument("--heads", type=int, default=8); p.add_argument("--vocab", type=int, default=256)
p.add_argument("--seq", type=int, default=256); p.add_argument("--batch", type=int, default=32)
p.add_argument("--steps", type=int, default=100); p.add_argument("--compile", type=int, default=1)
a = p.parse_args()
dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = True

def gelu(x): return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h
        s.ln1 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3*d); s.proj = nn.Linear(d, d)
        s.ln2 = nn.LayerNorm(d); s.fc = nn.Linear(d, 4*d); s.fcproj = nn.Linear(4*d, d)
    def forward(s, x):
        B, T, C = x.shape; hd = C // s.h
        q, k, v = s.qkv(s.ln1(x)).split(C, 2)
        q = q.view(B, T, s.h, hd).transpose(1, 2); k = k.view(B, T, s.h, hd).transpose(1, 2)
        v = v.view(B, T, s.h, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # official fused FA
        x = x + s.proj(y.transpose(1, 2).reshape(B, T, C))
        x = x + s.fcproj(gelu(s.fc(s.ln2(x))))
        return x

class GPT(nn.Module):
    def __init__(s, L, d, h, V, T):
        super().__init__()
        s.wte = nn.Embedding(V, d); s.wpe = nn.Embedding(T, d)
        s.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        s.lnf = nn.LayerNorm(d); s.head = nn.Linear(d, V, bias=False); s.head.weight = s.wte.weight
    def forward(s, idx, tgt):
        B, T = idx.shape
        x = s.wte(idx) + s.wpe(torch.arange(T, device=idx.device))[None]
        for b in s.blocks: x = b(x)
        logits = s.head(s.lnf(x))
        return F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))

def run(model, tag, steps):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    idx = torch.randint(0, a.vocab, (a.batch, a.seq), device=dev)
    tgt = torch.randint(0, a.vocab, (a.batch, a.seq), device=dev)
    for _ in range(10):  # warmup
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16): loss = model(idx, tgt)
        loss.backward(); opt.step()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16): loss = model(idx, tgt)
        loss.backward(); opt.step()
    torch.cuda.synchronize(); dt = (time.time() - t0) / steps
    toks = a.batch * a.seq / dt
    print(f"  {tag:18s} {dt*1e3:7.2f} ms/step   {toks:9.0f} tok/s")

print(f"PyTorch {torch.__version__} on {torch.cuda.get_device_name()}  (d{a.d} x {a.layers}L, seq{a.seq}, batch{a.batch}, bf16)")
m = GPT(a.layers, a.d, a.heads, a.vocab, a.seq).to(dev)
run(m, "eager", a.steps)
if a.compile:
    try:
        mc = torch.compile(m)
        run(mc, "compile", a.steps)
    except Exception as ex:
        print("  compile failed:", ex)
