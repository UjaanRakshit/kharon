import os, sys, urllib.request

# Byte-level corpus for training. Tries tinyshakespeare; falls back to a repeatable
# synthetic corpus if there's no network (e.g. on a compute node).
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT = sys.argv[1] if len(sys.argv) > 1 else "data/input.bin"
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

data = None
try:
    with urllib.request.urlopen(URL, timeout=10) as r:
        data = r.read()
    print("downloaded tinyshakespeare:", len(data), "bytes")
except Exception as e:
    print("download failed (%s); synthesizing corpus" % e)
    import io
    lines = []
    for i in range(20000):
        lines.append("to be or not to be that is the question %d\n" % (i % 97))
    data = ("".join(lines)).encode()
    print("synthesized:", len(data), "bytes")

with open(OUT, "wb") as f:
    f.write(data)
print("wrote", OUT)
