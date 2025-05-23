import matplotlib.pyplot as plt
import numpy as np
import os

# === COMMON DATA ===
models = ['Qwen2.5-7B-Instruct-1M', 'Llama-3.1-8B-Instruct', 'Gemini-2.0-flash']
# prompts = ['Zero-shot', 'Few-shot-1', 'Few-shot-2', 'Few-shot-3']
prompts = ['ZS', 'FS-1', 'FS-2', 'FS-3']
# logos = ['logo/qwen.png', 'logo/llama.png', 'logo/gemini.png']  # removed logo part

scores_dict = {
    # With Attack and Feature Description
    # "CICIDS2017": [
    #     [0.1807, 0.0942, 0.1896],
    #     [0.6344, 0.2063, 0.7101],
    #     [0.5426, 0.2145, 0.7359],
    #     [0.6163, 0.2209, 0.7277]
    # ],
    # "CICDDoS2019": [
    #     [0.0923, 0.0555, 0.0886],
    #     [0.3468, 0.2603, 0.6689],
    #     [0.3590, 0.3370, 0.7662],
    #     [0.3455, 0.4083, 0.7703]
    # ]

    # No Attack, No Feature Description
    "CICIDS2017": [
        [0.1672, 0.0848, 0.2054],
        [0.6401, 0.1907, 0.7444],
        [0.5779, 0.2069, 0.7261],
        [0.4827, 0.1784, 0.711]
    ],
    "CICDDoS2019": [
        [0.0797, 0.0709, 0.157],
        [0.3738, 0.2292, 0.7385],
        [0.3499, 0.4238, 0.786],
        [0.4123, 0.3776, 0.7821]
    ]
}

# high-contrast palette
colors = ['#FF6B6B', '#4ECDC4', '#1A535C']

# === IMAGE UTILS ===
# (logo utility functions removed)

# prepare logo arrays
# (logo arrays removed)

# === PLOTTING ===
fig, axes = plt.subplots(1, 2, figsize=(25, 4), sharey=True, dpi=300)

bar_width = 0.3
x = np.linspace(0, 4, len(prompts))
offsets = [-bar_width, 0, bar_width]

for ax, (dataset, all_scores) in zip(axes, scores_dict.items()):
    flat = [s for row in all_scores for s in row]
    ax.set_ylim(0, max(flat) + 0.08)

    for i, (model_name, scores) in enumerate(zip(models, zip(*all_scores))):
        bars = ax.bar(
            x + offsets[i],
            scores,
            width=bar_width,
            label=model_name,
            color=colors[i]
        )

        for bar, score in zip(bars, scores):
            # add the icon
            # (logo handling removed)

            # add the numeric label
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{score:.2f}",
                ha='center',
                va='bottom',
                fontsize=18
            )

    ax.set_title(dataset, fontsize=42)
    ax.set_ylabel("F1 Score", fontsize=42)
    if ax != axes[0]:
        ax.set_ylabel("") 
    ax.set_xticks(x)
    ax.set_xticklabels(prompts, fontsize=42)
    ax.tick_params(axis='y', labelsize=26)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

# single legend at top center
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='upper center',
    fontsize=30,
    frameon=False,
    ncol=3,
    bbox_to_anchor=(0.5, 1.25)
)

os.makedirs("plots", exist_ok=True)
plt.subplots_adjust(wspace=0.1, top=0.88)
plt.savefig("plots/f1_no_attack_no_feature_des.pdf", bbox_inches='tight', dpi=300)
# plt.savefig("plots/f1_with_attack_with_feature_des.pdf", bbox_inches='tight', dpi=300)
