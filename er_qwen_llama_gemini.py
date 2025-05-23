import matplotlib.pyplot as plt
import numpy as np

# Prompt names for X-axis
prompts = ['ZS', 'FS1', 'FS2', 'FS3']
x = np.arange(len(prompts))

# Models and their styles
models = ['Qwen2.5-7B-Instruct-1M', 'Llama-3.1-8B-Instruct', 'Gemini-2.0-flash']
colors = ['#FF6B6B', '#4ECDC4', '#1A535C']
markers = ['D', 's', '*']

# Error rate values: dataset → prompt → model
# Each row = prompt, each col = model
scores_dict = {
    # With Attack and Feature Description
    # "CICIDS2017": [
    #     [0.0297, 0.0466, 0.0],     # ZS
    #     [0.0039, 0.0836, 0.0],     # FS1
    #     [0.0067, 0.0945, 0.0018],  # FS2
    #     [0.0086, 0.1254, 0.0022],  # FS3
    # ],
    # "CICDDoS2019": [
    #     [0.0143, 0.0743, 0.0],     # ZS
    #     [0.0044, 0.0268, 0.0],     # FS1
    #     [0.0061, 0.0385, 0.0],     # FS2
    #     [0.0055, 0.0654, 0.0],     # FS3
    # ]

    # No Attack, No Feature Description
    "CICIDS2017": [
        [0.0138, 0.0661, 0.0],     # ZS
        [0.0145, 0.0845, 0.0],     # FS1
        [0.0695, 0.0797, 0.0002],  # FS2
        [0.3657, 0.0892, 0.0007],  # FS3
    ],
    "CICDDoS2019": [
        [0.0025, 0.0698, 0.0001],     # ZS
        [0.0042, 0.0281, 0.0],     # FS1
        [0.0376, 0.0273, 0.0],     # FS2
        [0.072, 0.0351, 0.0],     # FS3
    ]


}

# Compute global maximum value for dynamic Y-axis scaling
all_values = [v for dataset in scores_dict.values() for row in dataset for v in row]
y_max = max(all_values)

# Create 1x2 subplot grid
fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True, dpi=300)

for i, (dataset, score_rows) in enumerate(scores_dict.items()):
    ax = axes[i]
    
    for m_idx, model in enumerate(models):
        y = [row[m_idx] for row in score_rows]
        ax.plot(
            x, y,
            label=model,
            color=colors[m_idx],
            marker=markers[m_idx],
            linestyle='--',
            linewidth=2,
            markersize=8
        )
    
    ax.set_title(dataset, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(prompts, fontsize=14)
    if i == 0:
        ax.set_ylabel("Error Rate", fontsize=14)

    # Dynamic Y-axis with slight padding
    ax.set_ylim(-0.01, y_max + 0.02)
    # ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')


    # Clean up axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Shared legend above all plots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.08),
    ncol=len(models),
    frameon=False,
    fontsize=10
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(wspace=0.3) 
plt.savefig("plots/error_rate_no_attack_no_feature_des.pdf", bbox_inches='tight')
