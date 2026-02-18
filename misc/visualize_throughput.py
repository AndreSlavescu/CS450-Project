#!/usr/bin/env python3
"""Grouped bar graph comparing vLLM, SgLang, and Megakernels throughput (tokens/s) across GPUs."""

import matplotlib
matplotlib.use("Agg")  # headless; no display required
import matplotlib.pyplot as plt
import numpy as np

# Throughput data (tokens/s)
data = {
    "H100": {
        "vLLM": [528.3286219400202, 543.9213663676006, 595.994130425175],
        "SgLang": [652.4172194758338, 651.894325088411, 617.707075474433],
        "Megakernel": [1016.25, 1004.35, 1004.60],
    },
    "B200": {
        "vLLM": [763.1435042131611, 765.1417427353231, 630.0018816914028],
        "SgLang": [777.8798011215572, 582.5545180470235, 708.0349989612283],
        "Megakernel": [1581.22, 1576.93, 1578.34],
    },
}

gpus = list(data.keys())
methods = ["vLLM", "SgLang", "Megakernel"]
colors = {"vLLM": "#8e63b6", "SgLang": "#f28e2b", "Megakernel": "#4fb6b6"}

means = np.array([[np.mean(data[g][m]) for m in methods] for g in gpus])
stds = np.array([[np.std(data[g][m], ddof=1) for m in methods] for g in gpus])  # sample std (n=3)

x = np.arange(len(gpus))
bar_w = 0.25
offsets = np.linspace(-bar_w, bar_w, num=len(methods))

fig, ax = plt.subplots(figsize=(8, 5))

bars_by_method = []
for i, method in enumerate(methods):
    bars = ax.bar(
        x + offsets[i],
        means[:, i],
        width=bar_w,
        yerr=stds[:, i],
        capsize=6,
        label=method,
        color=colors[method],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )
    bars_by_method.append(bars)

ax.set_ylabel("Tokens/s", fontsize=14, fontweight="bold")
ax.set_xlabel("GPU", fontsize=16, fontweight="bold")
ax.set_title("Llama-1B (BF16) Batch-Size 1, Decoding Throughput", fontsize=18, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(gpus, fontsize=14)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)

ymax = float(np.max(means + stds))
ax.set_ylim(0, ymax * 1.15)

ax.legend(loc="upper left", fontsize=13, frameon=True)

# Annotate bars with mean values
for bars in bars_by_method:
    for b in bars:
        ax.annotate(
            f"{b.get_height():.0f}",
            xy=(b.get_x() + b.get_width() / 2, b.get_height()),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

plt.tight_layout()
plt.savefig("throughput_comparison.png", dpi=150, bbox_inches="tight")
print("Saved throughput_comparison.png")
plt.close()
