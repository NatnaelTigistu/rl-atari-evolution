from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ── Directories ──────────────────────────────────────────────────────────────
_PLOTS_DIR = Path("benchmarks/plots")
_LOGS_DIR  = Path("benchmarks/logs")

# ── Style constants ──────────────────────────────────────────────────────────
_ALGO_COLORS = {
    "REINFORCE":   "#4878D0",   # blue
    "Vanilla A2C": "#EE854A",   # orange
    "A2C + GAE":   "#6ACC65",   # green
    "PPO":         "#D65F5F",   # red
}
_ROLLING_WINDOW = 50   # same window used in experiment.ipynb


def _load_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a training CSV written by src/train.py CSVLogger."""
    df = pd.read_csv(csv_path)
    # Normalize column names regardless of minor naming differences
    df.columns = [c.strip().lower() for c in df.columns]
    # Accept both 'reward' (train.py) and 'score' (spec name)
    if "reward" in df.columns and "score" not in df.columns:
        df = df.rename(columns={"reward": "score"})
    if "elapsed_s" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"elapsed_s": "time"})
    return df


def plot_single_algorithm(
    csv_path: str | Path,
    title: str,
    output_png_path: str | Path,
    window: int = _ROLLING_WINDOW,
) -> Path:
    """
    Plot episode score, rolling average, and loss for one algorithm.

    Saves a two-panel figure (score | loss) to output_png_path and returns it.
    """
    df = _load_csv(csv_path)
    df["rolling_avg"] = df["score"].rolling(window, min_periods=1).mean()

    output_png_path = Path(output_png_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Left panel: score + rolling average
    ax = axes[0]
    ax.plot(df["episode"], df["score"],
            alpha=0.25, color="steelblue", linewidth=0.8, label="Episode score")
    ax.plot(df["episode"], df["rolling_avg"],
            color="navy", linewidth=2.0, label=f"{window}-ep avg")
    ax.axhline(0.0,   color="gray",  linewidth=0.8, linestyle="--", label="Score = 0")
    ax.axhline(-21.0, color="black", linewidth=0.8, linestyle=":",  label="Score = −21 (worst)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("Training Scores")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right panel: loss
    ax = axes[1]
    ax.plot(df["episode"], df["loss"], alpha=0.5, color="tomato", linewidth=0.8, label="Loss")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluator] Saved single-algo plot → '{output_png_path}'")
    return output_png_path


def plot_master_comparison(
    log_dict: dict[str, str | Path] | None = None,
    output_png_path: str | Path = _PLOTS_DIR / "evolution_comparison.png",
    window: int = _ROLLING_WINDOW,
) -> Path:
    """
    Plot rolling-average scores for all algorithms on one graph.

    log_dict maps display name → CSV path. Defaults to the standard layout:
        {
            "REINFORCE":   "benchmarks/logs/reinforce.csv",
            "Vanilla A2C": "benchmarks/logs/a2c.csv",
            "A2C + GAE":   "benchmarks/logs/a2c_gae.csv",
            "PPO":         "benchmarks/logs/ppo.csv",
        }

    Only CSVs that actually exist are plotted (others are silently skipped).
    """
    if log_dict is None:
        log_dict = {
            "REINFORCE":   _LOGS_DIR / "reinforce.csv",
            "Vanilla A2C": _LOGS_DIR / "a2c.csv",
            "A2C + GAE":   _LOGS_DIR / "a2c_gae.csv",
            "PPO":         _LOGS_DIR / "ppo.csv",
        }

    output_png_path = Path(output_png_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))

    plotted = 0
    for algo_name, csv_path in log_dict.items():
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"[evaluator] Skipping '{algo_name}' — '{csv_path}' not found.")
            continue
        df = _load_csv(csv_path)
        df["rolling_avg"] = df["score"].rolling(window, min_periods=1).mean()
        color = _ALGO_COLORS.get(algo_name, None)
        ax.plot(
            df["episode"],
            df["rolling_avg"],
            label=f"{algo_name} ({window}-ep avg)",
            color=color,
            linewidth=2.0,
        )
        plotted += 1

    if plotted == 0:
        print("[evaluator] No CSV files found — nothing to plot.")
        plt.close(fig)
        return output_png_path

    # Baselines
    ax.axhline(0.0,   color="gray",  linewidth=1.0, linestyle="--", alpha=0.7,
               label="Score = 0 (random-ish)")
    ax.axhline(-21.0, color="black", linewidth=1.0, linestyle=":",  alpha=0.7,
               label="Score = −21 (worst possible)")

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel(f"Score ({window}-ep rolling avg)", fontsize=11)
    ax.set_title("Deep RL Evolution: REINFORCE → A2C → PPO on ALE/Pong-v5",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="minor", alpha=0.1)

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluator] Saved comparison plot → '{output_png_path}'")
    return output_png_path


# ── CLI convenience ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # No args: generate the master comparison from default log paths
        plot_master_comparison()
    elif len(sys.argv) == 4:
        # positional: csv_path title output_png
        plot_single_algorithm(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage:")
        print("  python -m benchmarks.evaluator                        # master comparison")
        print("  python -m benchmarks.evaluator <csv> <title> <out>   # single-algo plot")
