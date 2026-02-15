#!/usr/bin/env python3
"""Bar graph comparing vLLM, SgLang, and Megakernels throughput (tokens/s)."""

import matplotlib
matplotlib.use("Agg")  # headless; no display required
import matplotlib.pyplot as plt
import numpy as np

# Throughput data (tokens/s) from 5 iterations each
vllm_throughput = [
    528.3286219400202,
    543.9213663676006,
    595.994130425175,
    625.9173205285103,
    543.755351407247,
]
sglang_throughput = [
    652.4172194758338,
    651.894325088411,
    617.707075474433,
    639.1022085328391,
    638.9435988462286,
]
megakernels_throughput = [
    1016.25,
    1004.35,
    1004.60,
    1004.48,
    1003.43,
]

labels = ["vLLM", "SgLang", "Megakernels"]
means = [
    np.mean(vllm_throughput),
    np.mean(sglang_throughput),
    np.mean(megakernels_throughput),
]
stds = [
    np.std(vllm_throughput),
    np.std(sglang_throughput),
    np.std(megakernels_throughput),
]

x = np.arange(len(labels))
width = 1  # wider bars = smaller gaps between them

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(x, means, width, yerr=stds, capsize=8, color=["#4A90D9", "#50C878", "#E07C3C"], edgecolor="black", linewidth=0.8)

ax.set_ylabel("Throughput (tokens/s)", fontsize=12)
ax.set_xlabel("Approach", fontsize=12)
ax.set_title("Baseline Throughput Comparison (5 iterations)", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, max(means) * 1.25)

# Annotate bars with mean value
for bar, mean in zip(bars, means):
    ax.annotate(
        f"{mean:.0f}",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig("throughput_comparison.png", dpi=150, bbox_inches="tight")
print("Saved throughput_comparison.png")
plt.close()
