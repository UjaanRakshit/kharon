import struct, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reference GPT for the M1 oracle. Pre-norm GPT-2 style, FP32, dropout off,
# weight-tied head. We dump the actual initialized weights plus a fixed batch,
# and the expected logits / loss / grads / params-after-one-AdamW-step. The C
# stack loads the same weights and must reproduce all of it to FP32 tolerance.

SEED = 1337
CFG = dict(n_layer=2, d_model=128, n_head=4, vocab=256, seq=64, batch=8)
OPT = dict(lr=3e-4, beta1=0.9, beta2=0.95, eps=1e-8, wd=0.1)

# --- binary container ---------------------------------------------------------
# magic "CHRN", version, 6 int32 config, 5 fp32 opt hparams, then records.
# record: name_len i32, name, ndim i32, dims i32[ndim], dtype i32 (0=f32,1=i32), raw data
MAGIC = b"CHRN"
VERSION = 1

class Writer:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.f.write(MAGIC)
        self.f.write(struct.pack("<i", VERSION))
        self.f.write(struct.pack("<6i", *[CFG[k] for k in
            ("n_layer","d_model","n_head","vocab","seq","batch")]))
        self.f.write(struct.pack("<5f", OPT["lr"], OPT["beta1"], OPT["beta2"],
            OPT["eps"], OPT["wd"]))

    def put(self, name, t):
        if isinstance(t, torch.Tensor):
            t = t.detach().contiguous().cpu()
            if t.dtype == torch.int64 or t.dtype == torch.int32:
                arr, dt = t.to(torch.int32), 1
            else:
                arr, dt = t.to(torch.float32), 0
            dims = list(arr.shape)
            raw = arr.numpy().tobytes()
        else:  # python float scalar
            arr, dt, dims = None, 0, []
            raw = struct.pack("<f", float(t))
        nb = name.encode()
        self.f.write(struct.pack("<i", len(nb))); self.f.write(nb)
        self.f.write(struct.pack("<i", len(dims)))
        for d in dims: self.f.write(struct.pack("<i", d))
        self.f.write(struct.pack("<i", dt))
        self.f.write(raw)

    def close(self):
        self.f.write(struct.pack("<i", -1))  # end marker (name_len = -1)
        self.f.close()


def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x**3)))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, h = c["d_model"], c["n_head"]
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3*d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4*d)
        self.fcproj = nn.Linear(4*d, d)

    def forward(self, x):
        B, T, C = x.shape
        hd = C // self.h
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
        self.wte = nn.Embedding(c["vocab"], d)
        self.wpe = nn.Embedding(c["seq"], d)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c["n_layer"])])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, c["vocab"], bias=False)
        self.head.weight = self.wte.weight  # tie
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None]
        for b in self.blocks:
            x = b(x)
        x = self.lnf(x)
        logits = self.head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def main():
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    m = GPT(CFG)
    idx = torch.randint(0, CFG["vocab"], (CFG["batch"], CFG["seq"]))
    tgt = torch.randint(0, CFG["vocab"], (CFG["batch"], CFG["seq"]))

    w = Writer("tests/m1_ref.bin")
    # input
    w.put("input_ids", idx)
    w.put("targets", tgt)
    # weights, in a fixed, C-friendly naming scheme
    w.put("wte", m.wte.weight)
    w.put("wpe", m.wpe.weight)
    for i, b in enumerate(m.blocks):
        p = f"blk{i}."
        w.put(p+"ln1.w", b.ln1.weight); w.put(p+"ln1.b", b.ln1.bias)
        w.put(p+"qkv.w", b.qkv.weight); w.put(p+"qkv.b", b.qkv.bias)
        w.put(p+"proj.w", b.proj.weight); w.put(p+"proj.b", b.proj.bias)
        w.put(p+"ln2.w", b.ln2.weight); w.put(p+"ln2.b", b.ln2.bias)
        w.put(p+"fc.w", b.fc.weight); w.put(p+"fc.b", b.fc.bias)
        w.put(p+"fcproj.w", b.fcproj.weight); w.put(p+"fcproj.b", b.fcproj.bias)
    w.put("lnf.w", m.lnf.weight); w.put("lnf.b", m.lnf.bias)

    # forward + backward
    logits, loss = m(idx, tgt)
    loss.backward()
    w.put("logits", logits)
    w.put("loss", loss.item())

    named = dict(m.named_parameters())
    # grads (skip the tied head.weight alias; its grad lives on wte.weight)
    grad_map = {
        "wte": m.wte.weight, "wpe": m.wpe.weight, "lnf.w": m.lnf.weight, "lnf.b": m.lnf.bias,
    }
    for i, b in enumerate(m.blocks):
        p = f"blk{i}."
        grad_map.update({
            p+"ln1.w": b.ln1.weight, p+"ln1.b": b.ln1.bias,
            p+"qkv.w": b.qkv.weight, p+"qkv.b": b.qkv.bias,
            p+"proj.w": b.proj.weight, p+"proj.b": b.proj.bias,
            p+"ln2.w": b.ln2.weight, p+"ln2.b": b.ln2.bias,
            p+"fc.w": b.fc.weight, p+"fc.b": b.fc.bias,
            p+"fcproj.w": b.fcproj.weight, p+"fcproj.b": b.fcproj.bias,
        })
    for name, t in grad_map.items():
        w.put(name+".grad", t.grad)

    # one AdamW step, uniform wd, then dump updated params
    opt = torch.optim.AdamW(m.parameters(), lr=OPT["lr"], betas=(OPT["beta1"], OPT["beta2"]),
                            eps=OPT["eps"], weight_decay=OPT["wd"])
    opt.step()
    for name, t in grad_map.items():
        w.put(name+".step", t.detach())

    w.close()
    print("wrote tests/m1_ref.bin  loss=%.6f  params=%d" %
          (loss.item(), sum(p.numel() for p in m.parameters())))


if __name__ == "__main__":
    main()
