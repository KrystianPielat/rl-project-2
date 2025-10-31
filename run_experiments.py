import copy
import os
import warnings
from pathlib import Path

import bbrl_utils
import torch
from omegaconf import OmegaConf

from lander import TD3, run_td3
from src import (
    ActionTimeExtensionWrapper,
    FeatureFilterWrapper,
    ObsTimeExtensionWrapper,
    launch_tensorboard,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning)

bbrl_utils.setup()


def run_single_experiment(config_name, wrappers, params, seed, results_dir):
    """Run a single experiment."""
    params = copy.deepcopy(params)
    params["algorithm"]["seed"] = seed
    params["base_dir"] = f"{config_name}/seed_{seed}"

    cfg = OmegaConf.create(params)
    td3 = TD3(cfg, wrappers)

    print(f"\n{'=' * 60}")
    print(f"Running: {config_name} | Seed: {seed}")
    print(f"{'=' * 60}")

    try:
        run_td3(td3)
    except KeyboardInterrupt:
        print(f"\nInterrupted: {config_name} seed {seed}")
        raise

    model_path = results_dir / f"{config_name}_seed{seed}.pth"
    torch.save(td3.actor.state_dict(), model_path)


def run_experiments(configurations, base_params, seeds, results_dir):
    """Run multiple experiments across configurations and seeds."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        for config_name, wrappers in configurations.items():
            run_single_experiment(config_name, wrappers, base_params, seed, results_dir)

    print("\nAll experiments complete. Run visualize_results.py to see plots.")


if __name__ == "__main__":
    outputs_dir = Path(__file__).parent / "outputs"
    launch_tensorboard(outputs_dir)

    base_params = {
        "save_best": True,
        "collect_stats": False,
        "plot_agents": False,
        "algorithm": {
            "max_grad_norm": 0.5,
            "n_envs": 20,
            "n_steps": 50,
            "nb_evals": 10,
            "discount_factor": 0.98,
            "buffer_size": 2e5,
            "batch_size": 264,
            "tau_target": 0.05,
            "eval_interval": 2000,
            "max_epochs": 1000,
            "learning_starts": 1000,
            "action_noise": 0.1,
            "policy_delay": 2,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "architecture": {
                "actor_hidden_size": [128, 64],
                "critic_hidden_size": [128, 64],
            },
        },
        "gym_env": {"env_name": "LunarLander-v3", "env_args": {"continuous": True}},
        "actor_optimizer": {
            "classname": "torch.optim.Adam",
            "lr": 1e-4,
        },
        "critic_optimizer": {
            "classname": "torch.optim.Adam",
            "lr": 1e-3,
        },
    }

    configurations = {
        "full_observability": [],
        "partial_observability": [
            (FeatureFilterWrapper, [3], {}),
            (FeatureFilterWrapper, [2], {}),
        ],
        "memory": [
            (ObsTimeExtensionWrapper, [], {}),
        ],
        "chunked_actions": [
            (ActionTimeExtensionWrapper, [], {"num_actions": 3}),
        ],
    }

    seeds = [1, 2, 3, 4, 5]

    results_dir = Path(__file__).parent / "experiment_results"

    try:
        run_experiments(configurations, base_params, seeds, results_dir)
    except KeyboardInterrupt:
        print("\n\nExperiments stopped by user.")
