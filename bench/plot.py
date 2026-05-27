#!/usr/bin/env python3
# Regenerate every plot in bench/plots/ from bench/results.json (the measured numbers).
# One entrypoint, no manual steps: python bench/plot.py
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
R = json.load(open(os.path.join(HERE, "results.json")))
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name), dpi=110); plt.close(fig)
    print("wrote plots/" + name)

# 1. Parallel scaling (tok/s) — TP vs PP vs composed, on PCIe.
s = R["parallel_scaling_tok_s"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(s["labels"], [t / 1e3 for t in s["tok_s"]], color="#4C72B0")
for i, sp in enumerate(s["speedup"]):
    ax.text(i, s["tok_s"][i] / 1e3, f"{sp:.2f}x", ha="center", va="bottom")
ax.set_ylabel("tok/s (thousands)"); ax.set_title("Parallel scaling (L40S PCIe, d512x8L proxy)")
save(fig, "scaling.png")

# 2. Comms breakdown per step (1.2B on 8 GPU) — where the time goes.
b = R["comms_breakdown_ms"]["TP2xPP2xDP2_1B"]
fig, ax = plt.subplots(figsize=(6, 4))
parts = ["compute", "tp_allreduce", "pp_bubble", "dp_optcomm"]
vals = [b[p] for p in parts]
ax.bar(["compute", "TP all-reduce", "PP bubble", "DP opt-comm"], vals,
       color=["#55A868", "#C44E52", "#8172B2", "#CCB974"])
tot = sum(vals)
for i, v in enumerate(vals): ax.text(i, v, f"{v}ms\n{100*v/tot:.0f}%", ha="center", va="bottom")
ax.set_ylabel("ms / step"); ax.set_title("8-GPU step breakdown, 1.2B (PCIe-comms-bound)")
save(fig, "comms_breakdown.png")

# 3. Bubble fraction vs microbatch count (measured vs theory).
bb = R["bubble_fraction_pct"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(bb["M"], bb["measured"], "o-", label="measured")
ax.plot(bb["M"], bb["theory"], "s--", label="theory (P-1)/(M+P-1)")
ax.set_xlabel("microbatches M"); ax.set_ylabel("bubble %"); ax.set_title("1F1B bubble (P=2, L40S)")
ax.legend(); save(fig, "bubble.png")

# 4. Inference throughput vs group size (continuous batching scaling).
inf = R["inference_tok_s"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(inf["G"], inf["tok_s_350M"], "o-", label="350M (d1024x24L)")
ax.plot(inf["G"], inf["tok_s_1p2B"], "s-", label="1.2B (d2048x24L)")
ax.set_xlabel("group size G"); ax.set_ylabel("tok/s"); ax.set_title("Rollout throughput (L40S, paged-KV)")
ax.legend(); save(fig, "inference.png")

# 5. GRPO reward curve: SFT warm-start then RL refinement.
g = R["grpo"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(g["sft_step"], [a * 100 for a in g["sft_acc"]], "o-", color="#937860", label="SFT (CE)")
gx = [g["sft_step"][-1] + s for s in g["grpo_step"]]
ax.plot(gx, [a * 100 for a in g["grpo_acc"]], "s-", color="#C44E52", label="GRPO (RL)")
ax.axvline(g["sft_step"][-1], ls=":", color="gray")
ax.set_xlabel("step (SFT | GRPO)"); ax.set_ylabel("task accuracy %")
ax.set_title("GRPO refines a pretrained policy (single-digit add, L40S)")
ax.legend(); save(fig, "grpo.png")

print("done.")
