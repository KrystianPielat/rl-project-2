import copy
import os
import time
import warnings
from pathlib import Path

import bbrl_utils
import gymnasium as gym
import torch
import torch.nn as nn
from bbrl.agents import Agents
from bbrl_utils.algorithms import EpochBasedAlgo
from bbrl_utils.nn import setup_optimizer
from omegaconf import OmegaConf

from src import (
    AddOUNoise,
    ContinuousDeterministicActor,
    ContinuousQAgent,
    launch_tensorboard,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning)

bbrl_utils.setup()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mse = nn.MSELoss()

print(f"Using device: {device}")


def apply_wrappers(env, wrappers):
    for wrapper_class, args, kwargs in wrappers:
        env = wrapper_class(env, *args, **kwargs)
    return env


def compute_critic_loss(cfg, reward, must_bootstrap, q_values, target_q_values_min):
    gamma = cfg.algorithm.discount_factor
    y = reward + must_bootstrap * (gamma * target_q_values_min.detach())
    return mse(q_values, y)


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1.0 - tau) * tp.data + tau * sp.data)


class TD3(EpochBasedAlgo):
    def __init__(self, cfg, wrappers):
        super().__init__(cfg)

        for i in range(len(self.train_env.envs)):
            self.train_env.envs[i] = apply_wrappers(self.train_env.envs[i], wrappers)

        for i in range(len(self.eval_env.envs)):
            self.eval_env.envs[i] = apply_wrappers(self.eval_env.envs[i], wrappers)

        self.train_env.observation_space = self.train_env.envs[0].observation_space
        self.eval_env.observation_space = self.eval_env.envs[0].observation_space
        self.train_env.action_space = self.train_env.envs[0].action_space
        self.eval_env.action_space = self.eval_env.envs[0].action_space

        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()

        wrapper_names = [w[0].__name__ for w in wrappers] if wrappers else ["None"]
        print(f"Wrappers: {' -> '.join(wrapper_names)}")
        print(f"obs_size={obs_size}, act_size={act_size}")

        self.device = device

        self.critic_1 = ContinuousQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size, device
        ).with_prefix("critic-1/")
        self.target_critic_1 = copy.deepcopy(self.critic_1).with_prefix("target-critic-1/")

        self.critic_2 = ContinuousQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size, device
        ).with_prefix("critic-2/")
        self.target_critic_2 = copy.deepcopy(self.critic_2).with_prefix("target-critic-2/")

        self.actor = ContinuousDeterministicActor(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size, device
        )
        self.target_actor = copy.deepcopy(self.actor)

        self.actor.to(device)
        self.target_actor.to(device)
        self.critic_1.to(device)
        self.critic_2.to(device)
        self.target_critic_1.to(device)
        self.target_critic_2.to(device)

        noise_agent = AddOUNoise(cfg.algorithm.action_noise, device)
        self.train_policy = Agents(self.actor, noise_agent)
        self.eval_policy = self.actor

        self.actor_optimizer = setup_optimizer(cfg.actor_optimizer, self.actor)
        self.critic_1_optimizer = setup_optimizer(cfg.critic_optimizer, self.critic_1)
        self.critic_2_optimizer = setup_optimizer(cfg.critic_optimizer, self.critic_2)


def run_td3(td3: TD3):
    tau = td3.cfg.algorithm.tau_target
    batch_size = int(td3.cfg.algorithm.batch_size)
    max_grad_norm = td3.cfg.algorithm.max_grad_norm
    policy_delay = td3.cfg.algorithm.policy_delay
    update_step = 0
    policy_noise = td3.cfg.algorithm.policy_noise
    noise_clip = td3.cfg.algorithm.noise_clip

    for rb in td3.iter_replay_buffers():
        for _ in range(3):
            rb_workspace = rb.get_shuffled(batch_size).to(device)

            with torch.no_grad():
                td3.target_actor(rb_workspace, t=1)
                next_actions = rb_workspace["action"][1].to(device)

                noise = torch.normal(0, policy_noise, size=next_actions.shape, device=device)
                noise = noise.clamp(-noise_clip, noise_clip)
                next_actions = (next_actions + noise).clamp(-1.0, 1.0)
                rb_workspace.set("action", 1, next_actions)

                td3.target_critic_1(rb_workspace, t=1)
                td3.target_critic_2(rb_workspace, t=1)

                q1_target = rb_workspace["target-critic-1/q_value"][1].view(-1, 1)
                q2_target = rb_workspace["target-critic-2/q_value"][1].view(-1, 1)
                q_target_min = torch.min(q1_target, q2_target)

                rewards = rb_workspace["env/reward"][0].view(-1, 1)
                dones = rb_workspace["env/done"][1].float().unsqueeze(-1)
                must_bootstrap = 1.0 - dones

            td3.critic_1(rb_workspace, t=0)
            td3.critic_2(rb_workspace, t=0)

            q1 = rb_workspace["critic-1/q_value"][0].view(-1, 1)
            q2 = rb_workspace["critic-2/q_value"][0].view(-1, 1)

            critic_1_loss = compute_critic_loss(td3.cfg, rewards, must_bootstrap, q1, q_target_min)
            critic_2_loss = compute_critic_loss(td3.cfg, rewards, must_bootstrap, q2, q_target_min)

            td3.critic_1_optimizer.zero_grad()
            critic_1_loss.backward()
            torch.nn.utils.clip_grad_norm_(td3.critic_1.parameters(), max_grad_norm)
            td3.critic_1_optimizer.step()

            td3.critic_2_optimizer.zero_grad()
            critic_2_loss.backward()
            torch.nn.utils.clip_grad_norm_(td3.critic_2.parameters(), max_grad_norm)
            td3.critic_2_optimizer.step()

            soft_update(td3.target_critic_1, td3.critic_1, tau)
            soft_update(td3.target_critic_2, td3.critic_2, tau)

            td3.logger.add_log("critic_loss_1", critic_1_loss, td3.nb_steps)
            td3.logger.add_log("critic_loss_2", critic_2_loss, td3.nb_steps)

            if update_step % policy_delay == 0:
                obs = rb_workspace["env/env_obs"][0]
                actor_action = td3.actor.model(obs)

                obs_act = torch.cat((obs, actor_action), dim=1)
                q_for_actor = td3.critic_1.model(obs_act).squeeze(-1)
                actor_loss = -torch.mean(q_for_actor)

                td3.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(td3.actor.parameters(), max_grad_norm)
                td3.actor_optimizer.step()

                td3.logger.add_log("actor_loss", actor_loss, td3.nb_steps)
                soft_update(td3.target_actor, td3.actor, tau)

            update_step += 1

        td3.evaluate()


if __name__ == "__main__":
    outputs_dir = Path(__file__).parent / "outputs"
    launch_tensorboard(outputs_dir)

    wrappers = [
        # (ActionTimeExtensionWrapper, [], {"num_actions": 3}),
    ]

    params = {
        "save_best": True,
        "base_dir": "${gym_env.env_name}/td3-S${algorithm.seed}_${current_time:}",
        "collect_stats": False,
        "plot_agents": False,
        "algorithm": {
            "seed": 2,
            "max_grad_norm": 0.5,
            "n_envs": 10,
            "n_steps": 20,
            "nb_evals": 10,
            "discount_factor": 0.98,
            "buffer_size": 2e5,
            "batch_size": 264,
            "tau_target": 0.05,
            "eval_interval": 2000,
            "max_epochs": 20000,
            "learning_starts": 1000,
            "action_noise": 0.1,
            "policy_delay": 2,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "architecture": {
                "actor_hidden_size": [32, 32],
                "critic_hidden_size": [32, 32],
            },
        },
        "gym_env": {
            "env_name": "CartPoleContinuous-v1",
        },
        "actor_optimizer": {
            "classname": "torch.optim.Adam",
            "lr": 1e-4,
        },
        "critic_optimizer": {
            "classname": "torch.optim.Adam",
            "lr": 1e-3,
        },
    }

    cfg = OmegaConf.create(params)

    td3 = TD3(cfg, wrappers)
    try:
        run_td3(td3)
    except KeyboardInterrupt:
        pass

    wrapper_names = (
        [w[0].__name__.replace("Wrapper", "").lower() for w in wrappers] if wrappers else ["full"]
    )
    model_name = f"td3_actor_{'_'.join(wrapper_names)}.pth"
    torch.save(td3.actor.state_dict(), model_name)

    actor = td3.actor
    actor.eval()

    env = gym.make("LunarLander-v3", continuous=True, render_mode="human")
    env = apply_wrappers(env, wrappers)

    obs, info = env.reset()
    done = False

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = actor.model(obs_tensor).cpu().numpy()[0]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        time.sleep(0.01)

    env.close()
