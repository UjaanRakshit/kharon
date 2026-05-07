import struct, math
import torch
import torch.nn.functional as F

# FlashAttention oracle: causal scaled-dot-product attention reference. Config is
# chosen so T spans several key tiles (stresses the online-softmax rescaling) and
# hd=64 is a realistic head dim. Output + dQ/dK/dV come from PyTorch SDPA.

SEED = 2024
B, H, T, HD = 2, 4, 256, 64

MAGIC = b"CHRN"; VERSION = 1

class Writer:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.f.write(MAGIC); self.f.write(struct.pack("<i", VERSION))
        # reuse the 6-int config slots: n_layer, d_model, n_head, vocab, seq, batch
        self.f.write(struct.pack("<6i", 0, H * HD, H, 0, T, B))
        self.f.write(struct.pack("<5f", 0, 0, 0, 0, 0))
    def put(self, name, t):
        t = t.detach().contiguous().cpu().to(torch.float32)
        raw = t.numpy().tobytes()
        nb = name.encode()
        self.f.write(struct.pack("<i", len(nb))); self.f.write(nb)
        self.f.write(struct.pack("<i", t.dim()))
        for d in t.shape: self.f.write(struct.pack("<i", d))
        self.f.write(struct.pack("<i", 0)); self.f.write(raw)
    def close(self):
        self.f.write(struct.pack("<i", -1)); self.f.close()

def main():
    torch.manual_seed(SEED)
    q = torch.randn(B, H, T, HD, requires_grad=True)
    k = torch.randn(B, H, T, HD, requires_grad=True)
    v = torch.randn(B, H, T, HD, requires_grad=True)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    do = torch.randn_like(o)
    o.backward(do)

    w = Writer("tests/attn_ref.bin")
    w.put("q", q); w.put("k", k); w.put("v", v)
    w.put("o", o); w.put("do", do)
    w.put("dq", q.grad); w.put("dk", k.grad); w.put("dv", v.grad)
    w.close()
    print("wrote tests/attn_ref.bin  B=%d H=%d T=%d hd=%d" % (B, H, T, HD))

if __name__ == "__main__":
    main()
