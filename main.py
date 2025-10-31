# Outlook

# In this notebook, using BBRL, you will study some effects of partial observability
# on the continuous action version of the LunarLander-v3 environment, using the TD3 algorithm.

# To emulate partial observability, you will design dedicated wrappers. Then you will study
# whether extending the input of the agent policy and critic with a memory of previous states
# and can help solve the partial observability issue. Yu will also study whether using action chunks
# instead of single actions as ouptput has an effect of the learning performance.
# This will also be achieved by designing other temporal extension wrappers.

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

# -----------------------------------------
# Temporal modification wrappers
# -----------------------------------------

class FeatureFilterWrapper(gym.Wrapper):
    """
    Remove a single feature at index `feature_index` from the observation vector.
    Updates observation_space accordingly.
    """
    def __init__(self, env, feature_index, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.feature_index = int(feature_index)

        obs_space = env.observation_space
        prev_shape = obs_space.shape
        prev_low = np.array(obs_space.low, copy=True)
        prev_high = np.array(obs_space.high, copy=True)
        prev_dtype = obs_space.dtype

        # Normalize negative indices
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
    """
    Concatenate previous observation (memory size 1) with current observation.
    On reset, previous observation is a zero vector.
    Updates observation_space accordingly.
    """
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        obs_space = env.observation_space

        low = np.array(obs_space.low, copy=True)
        high = np.array(obs_space.high, copy=True)
        shape = obs_space.shape
        dtype = obs_space.dtype

        low_extended = np.concatenate([low, low], axis=0)
        high_extended = np.concatenate([high, high], axis=0)
        shape_extended = (shape[0] * 2,)

        self.observation_space = gym.spaces.Box(low=low_extended, high=high_extended, shape=shape_extended, dtype=dtype)
        self._obs_dim = shape[0]
        self._dtype = dtype
        self._prev = np.zeros(self._obs_dim, dtype=self._dtype)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = np.asarray(obs, dtype=self._dtype).reshape(self._obs_dim)
        extended_obs = np.concatenate([self._prev, obs], axis=0)
        self._prev = obs.copy()
        return extended_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.asarray(obs, dtype=self._dtype).reshape(self._obs_dim)
        extended_obs = np.concatenate([self._prev, obs], axis=0)
        self._prev = obs.copy()
        return extended_obs, reward, terminated, truncated, info


class ActionTimeExtensionWrapper(gym.Wrapper):
    """
    Expand the action space by a factor M, receiving an action-chunk but executing only the first action.
    Handles scalar (shape == ()) and vector actions.
    """
    def __init__(self, env, M: int, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.M = int(M)

        act_space = env.action_space

        self._orig_action_shape = act_space.shape  # could be (), (1,), (k,)
        self._base_action_dim = int(np.prod(self._orig_action_shape)) if self._orig_action_shape != () else 1
        self._dtype = act_space.dtype

        # Flatten base low/high to 1-D
        if self._orig_action_shape == ():  # scalar Box
            base_low = np.array([act_space.low], dtype=self._dtype).reshape(1,)
            base_high = np.array([act_space.high], dtype=self._dtype).reshape(1,)
        else:
            base_low = np.asarray(act_space.low, dtype=self._dtype).reshape(self._base_action_dim)
            base_high = np.asarray(act_space.high, dtype=self._dtype).reshape(self._base_action_dim)

        new_low = np.tile(base_low, self.M)
        new_high = np.tile(base_high, self.M)
        new_shape = (self._base_action_dim * self.M,)  # always 1-D expanded

        self.action_space = gym.spaces.Box(low=new_low, high=new_high, shape=new_shape, dtype=self._dtype)

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def _extract_first_action(self, action):
        flat = np.asarray(action, dtype=self._dtype).reshape(-1)
        first_flat = flat[:self._base_action_dim]
        if self._orig_action_shape == ():  # scalar action expected
            return np.array(first_flat[0], dtype=self._dtype).item()
        else:
            return first_flat.reshape(self._orig_action_shape)

    def step(self, action):
        first_action = self._extract_first_action(action)
        obs, reward, terminated, truncated, info = self.env.step(first_action)
        return obs, reward, terminated, truncated, info


# -----------------------------------------
# TD3 components
# -----------------------------------------

class ContinuousQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.is_q_function = True
        self.model = build_mlp(
            [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t):
        obs = self.get(("env/env_obs", t)).to(device)      # B x D_obs
        action = self.get(("action", t)).to(device)         # B x D_act
        obs_act = torch.cat((obs, action), dim=1)           # B x (D_obs + D_act)
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
    """
    Ornstein-Uhlenbeck process noise for actions as suggested by DDPG paper.
    """
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

    def sample(self, act):
        act = act.to(device)
        if self.x_prev is None:
            self.x_prev = torch.zeros_like(act, device=device)
        x = (
            self.x_prev
            + self.theta * (0 - self.x_prev) * self.dt
            + self.std_dev * math.sqrt(self.dt) * torch.randn_like(act, device=device)
        )
        self.x_prev = x
        return act + x


def compute_critic_loss(
    cfg,
    reward: torch.Tensor,
    must_bootstrap: torch.Tensor,
    q_values: torch.Tensor,
    target_q_values_min: torch.Tensor,
):
    """
    TD target: r + gamma * (1 - done) * min(target_Q1, target_Q2)
    """
    gamma = cfg.algorithm.discount_factor
    y = reward + must_bootstrap * (gamma * target_q_values_min.detach())
    return mse(q_values, y)


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1.0 - tau) * tp.data + tau * sp.data)


class TD3(EpochBasedAlgo):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Define the agents and optimizers for TD3
        obs_size, act_size = self.train_env.get_obs_and_actions_sizes()
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

        # exploration noise
        noise_agent = AddOUNoise(cfg.algorithm.action_noise)

        self.train_policy = Agents(self.actor, noise_agent)
        self.eval_policy = self.actor

        # optimizers
        self.actor_optimizer = setup_optimizer(cfg.actor_optimizer, self.actor)
        self.critic_1_optimizer = setup_optimizer(cfg.critic_optimizer, self.critic_1)
        self.critic_2_optimizer = setup_optimizer(cfg.critic_optimizer, self.critic_2)


def run_td3(td3: TD3):
    gamma = td3.cfg.algorithm.discount_factor
    tau = td3.cfg.algorithm.tau_target
    batch_size = int(td3.cfg.algorithm.batch_size)
    max_grad_norm = td3.cfg.algorithm.max_grad_norm
    policy_delay = getattr(td3.cfg.algorithm, "policy_delay", 2)
    update_step = 0
    policy_noise = getattr(td3.cfg.algorithm, "policy_noise", 0.0)
    noise_clip = getattr(td3.cfg.algorithm, "noise_clip", 0.5)

    for rb in td3.iter_replay_buffers():
        for _ in range(3):  # Multiple updates per environment step
            rb_workspace = rb.get_shuffled(batch_size).to(td3.device)

            # --- build TD target components ---
            with torch.no_grad():
                # next actions using CURRENT actor at t=1 (like teacher's code)
                td3.actor(rb_workspace, t=1)
                next_actions = rb_workspace["action"][1].to(device)

                # add clipped noise to next_actions (TD3 policy smoothing)
                if policy_noise and policy_noise > 0:
                    noise = torch.normal(0, policy_noise, size=next_actions.shape, device=next_actions.device)
                    noise = noise.clamp(-noise_clip, noise_clip)
                    next_actions = (next_actions + noise).clamp(-1.0, 1.0)

                rb_workspace.set("action", 1, next_actions)

                # evaluate target critics at t=1 and take min
                td3.target_critic_1(rb_workspace, t=1)
                td3.target_critic_2(rb_workspace, t=1)

                q1_target = rb_workspace["target-critic-1/q_value"][1].view(-1, 1).to(device)  # (B,1)
                q2_target = rb_workspace["target-critic-2/q_value"][1].view(-1, 1).to(device)  # (B,1)
                q_target_min = torch.min(q1_target, q2_target)

                rewards = rb_workspace["env/reward"][0].view(-1, 1).to(device)  # r_t at t=0
                # Use env/done which includes both terminated and truncated
                dones = rb_workspace["env/done"][1].float().unsqueeze(-1)
                must_bootstrap = (1.0 - dones)

            # --- critics update using current Q at t=0 ---
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

            # soft-update critics' targets
            soft_update(td3.target_critic_1, td3.critic_1, tau)
            soft_update(td3.target_critic_2, td3.critic_2, tau)

            td3.logger.add_log("critic_loss_1", critic_1_loss, td3.nb_steps)
            td3.logger.add_log("critic_loss_2", critic_2_loss, td3.nb_steps)

            # --- actor update (no delay for CartPole) ---
            # Compute action and Q-value directly to preserve gradients
            obs = rb_workspace["env/env_obs"][0].to(device)
            actor_action = td3.actor.model(obs)  # Action with gradients from actor
            
            # Evaluate critic directly with action that has gradients
            obs_act = torch.cat((obs, actor_action), dim=1)
            q_for_actor = td3.critic_1.model(obs_act).squeeze(-1)
            actor_loss = -torch.mean(q_for_actor)

            td3.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(td3.actor.parameters(), max_grad_norm)
            td3.actor_optimizer.step()

            td3.logger.add_log("actor_loss", actor_loss, td3.nb_steps)

            # soft-update actor target
            soft_update(td3.target_actor, td3.actor, tau)

        update_step += 1

        if td3.evaluate():
            if td3.cfg.plot_agents:
                plot_policy(
                    td3.actor,
                    td3.eval_policy,
                    td3.best_reward,
                    str(td3.base_dir / "plots"),
                    td3.cfg.gym_env.env_name,
                    stochastic=False,
                )

# -----------------------------------------
# Launch
# -----------------------------------------

outputs_dir = Path(__file__).parent / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)
try:
    subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", "--logdir", str(outputs_dir.absolute())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
except Exception:
    pass

params = {
    "save_best": True,
    "base_dir": "${gym_env.env_name}/td3-S${algorithm.seed}_${current_time:}",
    "collect_stats": False,
    "plot_agents": False,
    "algorithm": {
        "seed": 2,
        "max_grad_norm": 0.5,
        "n_envs": 1,
        "n_steps": 100,  # CRITICAL: Match teacher's collection frequency
        "nb_evals": 10,
        "discount_factor": 0.98,
        "buffer_size": 2e5,
        "batch_size": 64,
        "tau_target": 0.05,
        "eval_interval": 2000,
        "max_epochs": 10000,  # More epochs since we collect less per epoch
        "learning_starts": 10000,
        "action_noise": 0.1,
        "architecture": {
            # "actor_hidden_size": [256, 256],
            # "critic_hidden_size": [256, 256],
            "actor_hidden_size": [1000, 1000],
            "critic_hidden_size": [1000, 1000],
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

td3 = TD3(cfg)
run_td3(td3)

# Save actor only
torch.save(td3.actor.state_dict(), "td3_actor.pth")

# Evaluation rollout with classic env render
actor = td3.actor
actor.eval()

env = gym.make("LunarLander-v3", continuous=True, render_mode="human")
# env = gym.make("CartPoleContinuous-v1", render_mode="human")

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
