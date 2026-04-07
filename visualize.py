"""
visualize.py — Generate plots from RL Outbreak Detective results.
Runs after demo.py, train.py, and evaluate.py.
Missing JSON files are skipped gracefully.
"""
import json
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

PALETTE = {
    "random":          "#e74c3c",
    "degree":          "#3498db",
    "contact_tracing": "#2ecc71",
    "rl_agent":        "#9b59b6",
    "bg":              "#1a1a2e",
    "panel":           "#16213e",
    "text":            "#eaeaea",
    "grid":            "#2a2a4a",
}

def styled_fig(title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])
    ax.set_title(title, color=PALETTE["text"], fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel, color=PALETTE["text"], fontsize=11)
    ax.set_ylabel(ylabel, color=PALETTE["text"], fontsize=11)
    ax.tick_params(colors=PALETTE["text"])
    ax.spines[:].set_color(PALETTE["grid"])
    ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.6, alpha=0.7)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Baseline step-by-step reward curves
# ─────────────────────────────────────────────────────────────────────────────
DEMO_PATH = "results_demo.json"
if os.path.exists(DEMO_PATH):
    with open(DEMO_PATH) as f:
        demo = json.load(f)

    fig, ax = styled_fig(
        "Baseline Strategies — Cumulative Reward per Step",
        "Test Step", "Cumulative Reward"
    )

    strategy_labels = {
        "random":          "Random Testing",
        "degree":          "Degree Heuristic",
        "contact_tracing": "Contact Tracing",
    }

    for key, label in strategy_labels.items():
        if key not in demo:
            continue
        rewards = [s["reward"] for s in demo[key]["steps"]]
        cumulative = np.cumsum(rewards)
        color = PALETTE.get(key, "#ffffff")
        ax.plot(range(1, len(cumulative) + 1), cumulative,
                marker="o", linewidth=2.2, markersize=5,
                color=color, label=label)
        # Mark final guess point differently
        ax.scatter([len(cumulative)], [cumulative[-1]],
                   s=120, color=color, zorder=5,
                   marker="*" if demo[key].get("final_correct") else "X")

    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], loc="lower left")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "baseline_steps.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✅  Saved: {out}")
else:
    print(f"⚠️  Skipping baseline plot — {DEMO_PATH} not found. Run demo.py first.")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — RL Training curve
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_PATH = "results_train.json"
if os.path.exists(TRAIN_PATH):
    with open(TRAIN_PATH) as f:
        train = json.load(f)

    metrics = train.get("training_metrics", [])
    iterations = [m["iteration"] for m in metrics]
    rewards    = [m["mean_reward"] if m["mean_reward"] is not None else 0 for m in metrics]

    fig, ax = styled_fig(
        "PPO Agent — Mean Reward per Training Iteration",
        "Iteration", "Mean Episode Reward"
    )
    ax.plot(iterations, rewards, color=PALETTE["rl_agent"],
            linewidth=2.5, marker="D", markersize=7, label="PPO (RLlib)")
    ax.fill_between(iterations, rewards, alpha=0.15, color=PALETTE["rl_agent"])
    ax.axhline(0, color=PALETTE["grid"], linewidth=1, linestyle="--")
    ax.legend(facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"])
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "training_curve.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✅  Saved: {out}")
else:
    print(f"⚠️  Skipping training curve — {TRAIN_PATH} not found. Run train.py first.")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Final accuracy comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
bar_labels  = []
bar_values  = []   # 1 = correct, 0 = wrong
bar_colors  = []

if os.path.exists(DEMO_PATH):
    with open(DEMO_PATH) as f:
        demo = json.load(f)
    label_map = {
        "random":          ("Random", PALETTE["random"]),
        "degree":          ("Degree", PALETTE["degree"]),
        "contact_tracing": ("Contact\nTracing", PALETTE["contact_tracing"]),
    }
    for key, (lbl, col) in label_map.items():
        if key in demo:
            bar_labels.append(lbl)
            bar_values.append(1 if demo[key].get("final_correct") else 0)
            bar_colors.append(col)

if os.path.exists("results_evaluate.json"):
    with open("results_evaluate.json") as f:
        ev = json.load(f)
    bar_labels.append("RL Agent\n(PPO)")
    bar_values.append(1 if ev.get("final_correct") else 0)
    bar_colors.append(PALETTE["rl_agent"])

if bar_labels:
    fig, ax = styled_fig(
        "Patient-Zero Identification — Accuracy Comparison",
        "Strategy", "Identified Correctly (1 = Yes, 0 = No)"
    )
    x = np.arange(len(bar_labels))
    bars = ax.bar(x, bar_values, color=bar_colors, width=0.5,
                  edgecolor=PALETTE["grid"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, color=PALETTE["text"], fontsize=11)
    ax.set_ylim(0, 1.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Wrong ✗", "Correct ✓"], color=PALETTE["text"])
    for bar, val in zip(bars, bar_values):
        label = "✓ Correct" if val else "✗ Wrong"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                label, ha="center", va="bottom",
                color=PALETTE["text"], fontsize=10, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "accuracy_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"✅  Saved: {out}")
else:
    print("⚠️  No result files found for accuracy chart. Run demo.py or evaluate.py first.")

print("\nDone! Check the 'plots/' folder for all generated charts.")
