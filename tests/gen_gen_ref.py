import struct, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reference for the M8 inference oracle. Same GPT as gen_reference.py (FP32, tied
# head), but instead of one forward we dump several prompts and their GREEDY
# continuations. The C engine loads these weights, runs paged-KV continuous-batch
# decode, and must reproduce every continuation token-for-token. Independent greedy
# per prompt is the ground truth the continuous-batch scheduler is checked against.

SEED = 1337
CFG = dict(n_layer=2, d_model=128, n_head=4, vocab=256, seq=64, batch=8)
MAGIC = b"CHRN"; VERSION = 1

class Writer:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.f.write(MAGIC); self.f.write(struct.pack("<i", VERSION))
        self.f.write(struct.pack("<6i", *[CFG[k] for k in
            ("n_layer","d_model","n_head","vocab","seq","batch")]))
        self.f.write(struct.pack("<5f", 3e-4, 0.9, 0.95, 1e-8, 0.1))
    def put(self, name, t):
        if isinstance(t, torch.Tensor):
            t = t.detach().contiguous().cpu()
            if t.dtype in (torch.int64, torch.int32): arr, dt = t.to(torch.int32), 1
            else: arr, dt = t.to(torch.float32), 0
            dims = list(arr.shape); raw = arr.numpy().tobytes()
        else:
            arr, dt, dims = None, 0, []; raw = struct.pack("<f", float(t))
        nb = name.encode()
        self.f.write(struct.pack("<i", len(nb))); self.f.write(nb)
        self.f.write(struct.pack("<i", len(dims)))
        for d in dims: self.f.write(struct.pack("<i", d))
        self.f.write(struct.pack("<i", dt)); self.f.write(raw)
    def close(self):
        self.f.write(struct.pack("<i", -1)); self.f.close()

def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x**3)))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, h = c["d_model"], c["n_head"]; self.h = h
        self.ln1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3*d); self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d); self.fc = nn.Linear(d, 4*d); self.fcproj = nn.Linear(4*d, d)
    def forward(self, x):
        B, T, C = x.shape; hd = C // self.h
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, self.h, hd).transpose(1, 2)
        k = k.view(B, T, self.h, hd).transpose(1, 2)
        v = v.view(B, T, self.h, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        x = x + self.fcproj(gelu(self.fc(self.ln2(x))))
        return x

class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["d_model"]
        self.wte = nn.Embedding(c["vocab"], d); self.wpe = nn.Embedding(c["seq"], d)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c["n_layer"])])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, c["vocab"], bias=False)
        self.head.weight = self.wte.weight
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)
    def logits(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None]
        for b in self.blocks: x = b(x)
        return self.head(self.lnf(x))

@torch.no_grad()
def greedy(m, prompt, n_new):
    ids = list(prompt)
    for _ in range(n_new):
        idx = torch.tensor([ids], dtype=torch.long)
        nxt = int(m.logits(idx)[0, -1].argmax())
        ids.append(nxt)
    return ids

def main():
    torch.manual_seed(SEED); torch.use_deterministic_algorithms(True)
    m = GPT(CFG)
    w = Writer("tests/gen_ref.bin")
    w.put("wte", m.wte.weight); w.put("wpe", m.wpe.weight)
    for i, b in enumerate(m.blocks):
        p = f"blk{i}."
        w.put(p+"ln1.w", b.ln1.weight); w.put(p+"ln1.b", b.ln1.bias)
        w.put(p+"qkv.w", b.qkv.weight); w.put(p+"qkv.b", b.qkv.bias)
        w.put(p+"proj.w", b.proj.weight); w.put(p+"proj.b", b.proj.bias)
        w.put(p+"ln2.w", b.ln2.weight); w.put(p+"ln2.b", b.ln2.bias)
        w.put(p+"fc.w", b.fc.weight); w.put(p+"fc.b", b.fc.bias)
        w.put(p+"fcproj.w", b.fcproj.weight); w.put(p+"fcproj.b", b.fcproj.bias)
    w.put("lnf.w", m.lnf.weight); w.put("lnf.b", m.lnf.bias)

    # Prompts of staggered length; one shares a prefix with another (prefix-sharing test).
    g = torch.Generator().manual_seed(7)
    base = torch.randint(0, CFG["vocab"], (12,), generator=g).tolist()
    prompts = [
        base[:6],                                   # seq 0
        base[:6] + torch.randint(0,256,(4,),generator=g).tolist(),  # seq 1: shares 6-tok prefix with 0
        torch.randint(0,256,(10,),generator=g).tolist(),            # seq 2
        torch.randint(0,256,(3,),generator=g).tolist(),             # seq 3 (short)
    ]
    n_new = [20, 16, 24, 28]
    w.put("n_seq", torch.tensor([len(prompts)], dtype=torch.int32))
    for i, (pr, nn_) in enumerate(zip(prompts, n_new)):
        full = greedy(m, pr, nn_)
        w.put(f"prompt{i}", torch.tensor(pr, dtype=torch.int32))
        w.put(f"gen{i}", torch.tensor(full, dtype=torch.int32))   # prompt + continuation
    w.close()
    print("wrote tests/gen_ref.bin  prompts=%d" % len(prompts))

if __name__ == "__main__":
    main()
