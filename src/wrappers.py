import gymnasium as gym
import numpy as np


class FeatureFilterWrapper(gym.Wrapper):
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
            low=low_extended, high=high_extended, dtype=env.observation_space.dtype
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
            pass
        else:
            (shape[0] * num_actions,)
        self.action_space = gym.spaces.Box(
            low=low_action, high=high_action, dtype=env.action_space.dtype
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
