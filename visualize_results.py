from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def load_all_runs(log_dir):
    log_dir = Path(log_dir)
    runs = []

    for run_dir in log_dir.rglob("*"):
        if not run_dir.is_dir():
            continue

        event_files = list(run_dir.glob("events.out.tfevents.*"))
        if not event_files:
            continue

        parts = run_dir.relative_to(log_dir).parts
        if len(parts) < 2:
            continue

        config = parts[0]
        seed = parts[1]

        ea = event_accumulator.EventAccumulator(str(run_dir))
        ea.Reload()

        if "reward/mean" in ea.Tags()["scalars"]:
            data = ea.Scalars("reward/mean")
            df = pd.DataFrame([(r.step, r.value) for r in data], columns=["step", "reward"])
            df["config"] = config
            df["seed"] = seed
            df["run"] = f"{config}_{seed}"
            runs.append(df)

    return pd.concat(runs, ignore_index=True) if runs else pd.DataFrame()


def plot_with_ci(df, output_path):
    fig, ax = plt.subplots(figsize=(14, 7))

    configs = sorted(df["config"].unique())
    colors = plt.cm.Set2(range(len(configs)))

    # Find common step range
    min_step = df["step"].min()
    max_step = df["step"].max()
    common_steps = np.arange(min_step, max_step + 1, 1000)

    for i, config in enumerate(configs):
        config_data = df[df["config"] == config]
        seeds = config_data["seed"].unique()

        # Interpolate each seed to common steps
        interpolated_runs = []
        for seed in seeds:
            seed_data = config_data[config_data["seed"] == seed].sort_values("step")
            interpolated = np.interp(common_steps, seed_data["step"], seed_data["reward"])
            interpolated_runs.append(interpolated)

        # Compute mean and CI across seeds
        interpolated_runs = np.array(interpolated_runs)
        mean = interpolated_runs.mean(axis=0)
        std = interpolated_runs.std(axis=0)
        stderr = std / np.sqrt(len(seeds))
        ci = 1.96 * stderr

        ax.plot(common_steps, mean, label=config, color=colors[i], linewidth=2.5)
        ax.fill_between(common_steps, mean - ci, mean + ci, color=colors[i], alpha=0.2)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("reward/mean", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    log_dir = Path(__file__).parent / "outputs" / "tblogs"
    df = load_all_runs(log_dir)

    if df.empty:
        print("No data found")
    else:
        num_runs = df["run"].nunique()
        num_configs = df["config"].nunique()
        print(f"Loaded {len(df)} points from {num_runs} runs across {num_configs} configs")
        output_path = Path(__file__).parent / "experiment_results" / "learning_curves.png"
        output_path.parent.mkdir(exist_ok=True)
        plot_with_ci(df, output_path)
        print(f"Plot saved to {output_path}")
