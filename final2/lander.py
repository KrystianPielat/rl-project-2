# TD3 on CartPoleContinuous-v1 with configurable wrappers

import os
import warnings
import subprocess
import sys
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning, module='pygame')
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import copy
import numpy as np
import gymnasium as gym
import math
import bbrl_gymnasium  # noqa: F401
import torch
import torch.nn as nn
from bbrl.agents import Agent, Agents, TemporalAgent
from bbrl_utils.algorithms import EpochBasedAlgo
from bbrl_utils.nn import build_mlp, setup_optimizer
from bbrl.visu.plot_policies import plot_policy
from omegaconf import OmegaConf

from bbrl_algos.models.exploration_agents import AddGaussianNoise
import time

import bbrl_utils

bbrl_utils.setup()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mse = nn.MSELoss()

print(f"Using device: {device}")

# =============================================================================
# CONFIGURABLE WRAPPERS
# =============================================================================

class FeatureFilterWrapper(gym.Wrapper):
    """Remove a single feature at index `feature_index` from the observation vector."""
    def __init__(self, env, feature_index, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.feature_index = int(feature_index)

        obs_space = env.observation_space
        prev_shape = obs_space.shape
        prev_low = np.array(obs_space.low, copy=True)
        prev_high = np.array(obs_space.high, copy=True)
        prev_dtype = obs_space.dtype

        idx = self.feature_index % prev_shape[0]

        low = np.delete(prev_low, idx)
        high = np.delete(prev_high, idx)
        shape = (prev_shape[0] - 1,)

        self._remove_index = idx
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape, dtype=prev_dtype)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs_filt = np.delete(np.asarray(obs), self._remove_index)
        return obs_filt, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_filt = np.delete(np.asarray(obs), self._remove_index)
        return obs_filt, reward, terminated, truncated, info


class ObsTimeExtensionWrapper(gym.Wrapper):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        low = env.observation_space.low
        high = env.observation_space.high
        shape = env.observation_space.shape
        low_extended = np.concatenate([low, low])
        high_extended = np.concatenate([high, high])
        self.observation_space = gym.spaces.Box(
            low=low_extended,
            high=high_extended,
            dtype=env.observation_space.dtype
        )
        self.obs_shape = shape[0]

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        zeros = np.zeros(self.obs_shape)
        extended_obs = np.concatenate([zeros, obs])
        self.previous_obs = obs
        return (extended_obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        extended_obs = np.concatenate([self.previous_obs, obs])
        self.previous_obs = obs
        return (extended_obs, reward, terminated, truncated, info)


class ActionTimeExtensionWrapper(gym.Wrapper):
    def __init__(self, env, num_actions, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        low = env.action_space.low
        high = env.action_space.high
        shape = env.action_space.shape
        low_action = np.tile(low, num_actions)
        high_action = np.tile(high, num_actions)
        if len(shape) == 0:
            shape_action = (num_actions,)
        else:
            shape_action = (shape[0] * num_actions,)
        self.action_space = gym.spaces.Box(
            low=low_action,
            high=high_action,
            dtype=env.action_space.dtype
        )
        self.num_actions = num_actions
        self.inner_action_size = shape[0] if len(shape) > 0 else 1

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return (obs, info)

    def step(self, action):
        if len(self.env.action_space.shape) == 0:
            first_action = action[0]
        else:
            n = self.env.action_space.shape[0]
            first_action = action[:n]
        obs, reward, terminated, truncated, info = self.env.step(first_action)
        return (obs, reward, terminated, truncated, info)


def apply_wrappers(env, wrappers):
    """Apply list of (wrapper_class, args, kwargs) to env."""
    for wrapper_class, args, kwargs in wrappers:
        env = wrapper_class(env, *args, **kwargs)
    return env


# -----------------------------------------
# TD3 components
# -----------------------------------------

class ContinuousQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.is_q_function = True
        self.model = build_mlp(
            [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.Tanh()
        )

    def forward(self, t):
        obs = self.get(("env/env_obs", t)).to(device)
        action = self.get(("action", t)).to(device)
        obs_act = torch.cat((obs, action), dim=1)
        q_value = self.model(obs_act).squeeze(-1)
        self.set((f"{self.prefix}q_value", t), q_value)


class ContinuousDeterministicActor(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_mlp(layers, activation=nn.Tanh(), output_activation=nn.Tanh())

    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs", t)).to(device)
        action = self.model(obs)
        self.set(("action", t), action)


class AddOUNoise(Agent):
    """Ornstein-Uhlenbeck process noise for actions."""
    def __init__(self, std_dev, theta=0.15, dt=1e-2):
        super().__init__()
        self.theta = theta
        self.std_dev = std_dev
        self.dt = dt
        self.x_prev = None

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        if act is None:
            return
        act = act.to(device)
        if self.x_prev is None:
            self.x_prev = torch.zeros_like(act, device=device)
        x = (
            self.x_prev
            + self.theta * (act - self.x_prev) * self.dt
            + self.std_dev * math.sqrt(self.dt) * torch.randn(act.shape, device=device)
        )
        self.x_prev = x
        self.set(("action", t), x)


def compute_critic_loss(cfg, reward, must_bootstrap, q_values, target_q_values_min):
    """TD target: r + gamma * (1 - done) * min(target_Q1, target_Q2)"""
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
        
        # Refresh spaces from wrapped environments
        self.train_env.observation_space = self.train_env.envs[0].observation_space
        self.eval_env.observation_space = self.eval_env.envs[0].observation_space
        self.train_env.action_space = self.train_env.envs[0].action_space
        self.eval_env.action_space = self.eval_env.envs[0].action_space

        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()
        
        # Print configuration
        wrapper_names = [w[0].__name__ for w in wrappers] if wrappers else ["None"]
        print(f"Wrappers: {' -> '.join(wrapper_names)}")
        print(f"  obs_size={obs_size}, act_size={act_size}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic_1 = ContinuousQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
        ).with_prefix("critic-1/")
        self.target_critic_1 = copy.deepcopy(self.critic_1).with_prefix("target-critic-1/").to(device)

        self.critic_2 = ContinuousQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
        ).with_prefix("critic-2/")
        self.target_critic_2 = copy.deepcopy(self.critic_2).with_prefix("target-critic-2/").to(device)

        self.actor = ContinuousDeterministicActor(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
        )
        self.target_actor = copy.deepcopy(self.actor).to(device)

        self.actor.to(self.device)
        self.target_actor.to(self.device)
        self.critic_1.to(self.device)
        self.critic_2.to(self.device)
        self.target_critic_1.to(self.device)
        self.target_critic_2.to(self.device)

        noise_agent = AddOUNoise(cfg.algorithm.action_noise)
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
            rb_workspace = rb.get_shuffled(batch_size).to(td3.device)

            with torch.no_grad():
                td3.target_actor(rb_workspace, t=1)
                next_actions = rb_workspace["action"][1].to(device)

                noise = torch.normal(0, policy_noise, size=next_actions.shape, device=next_actions.device)
                noise = noise.clamp(-noise_clip, noise_clip)
                next_actions = (next_actions + noise).clamp(-1.0, 1.0)
                rb_workspace.set("action", 1, next_actions)

                td3.target_critic_1(rb_workspace, t=1)
                td3.target_critic_2(rb_workspace, t=1)

                q1_target = rb_workspace["target-critic-1/q_value"][1].view(-1, 1).to(device)
                q2_target = rb_workspace["target-critic-2/q_value"][1].view(-1, 1).to(device)
                q_target_min = torch.min(q1_target, q2_target)

                rewards = rb_workspace["env/reward"][0].view(-1, 1).to(device)
                dones = rb_workspace["env/done"][1].float().unsqueeze(-1)
                must_bootstrap = (1.0 - dones)

            td3.critic_1(rb_workspace, t=0)
            td3.critic_2(rb_workspace, t=0)

            q1 = rb_workspace["critic-1/q_value"][0].view(-1, 1).to(device)
            q2 = rb_workspace["critic-2/q_value"][0].view(-1, 1).to(device)

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
                obs = rb_workspace["env/env_obs"][0].to(device)
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

        if td3.evaluate():
            if td3.cfg.plot_agents:
                plot_policy(
                    td3.actor, td3.eval_policy, td3.best_reward,
                    str(td3.base_dir / "plots"), td3.cfg.gym_env.env_name,
                    stochastic=False
                )

# -----------------------------------------
# Launch
# -----------------------------------------

outputs_dir = Path(__file__).parent / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)
try:
    subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", "--logdir", str(outputs_dir.absolute())],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
    )
except Exception:
    pass

# =============================================================================
# SPECIFY WRAPPERS HERE
# =============================================================================
wrappers = [
    # (FeatureFilterWrapper, [3], {}),  # Remove y velocity
    # (FeatureFilterWrapper, [2], {}),  # Remove x velocity (index shifts after first removal)
    # Features for LunarLander-v3 (continuous):
    # 0: x position         (horizontal position)
    # 1: y position         (vertical position)
    # 2: x velocity         (horizontal velocity)
    # 3: y velocity         (vertical velocity)
    # 4: angle              (lander angle)
    # 5: angular velocity   (lander angular velocity)
    # 6: left leg contact   (boolean, in contact: 1 or 0)
    # 7: right leg contact  (boolean, in contact: 1 or 0)
    # (ObsTimeExtensionWrapper, [], {}),
    (ActionTimeExtensionWrapper, [], {"num_actions": 3}),
]


params = {
    "save_best": True,
    "base_dir": "${gym_env.env_name}/td3-S${algorithm.seed}_${current_time:}",
    "collect_stats": False,
    "plot_agents": False,
    "algorithm": {
        "seed": 2,
        "max_grad_norm": 0.5,
        # "max_grad_norm": 1,
        "n_envs": 20,
        "n_steps": 50,  # Collect 100 steps per epoch (like teacher)
        "nb_evals": 10,
        "discount_factor": 0.98,
        "buffer_size": 2e5,
        "batch_size": 264,
        "tau_target": 0.05,
        "eval_interval": 2000,
        "max_epochs": 20000,
        "learning_starts": 1000,
        "action_noise": 0.1,  # OU noise during exploration
        "policy_delay": 2,  # TD3: Update actor every 2 critic updates
        "policy_noise": 0.2,  # TD3: Target policy smoothing noise std
        "noise_clip": 0.5,  # TD3: Clip target policy noise
        "architecture": {
            "actor_hidden_size": [128, 64],
            "critic_hidden_size": [128, 64],
        },
    },
    "gym_env": {
        "env_name": "LunarLander-v3",
        "env_args": {"continuous": True}
        # "env_name": "CartPoleContinuous-v1",
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

# Save model with descriptive name
wrapper_names = [w[0].__name__.replace("Wrapper", "").lower() for w in wrappers] if wrappers else ["full"]
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

