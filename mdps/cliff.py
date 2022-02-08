from abc import ABC
from typing import Dict

import gym
import numpy as np
from scipy import stats

from cmdp import MDP, cMDP
from utils.distributions import ConstantDistribution


class CliffMDP(MDP, ABC):
    def __init__(self, context: np.ndarray = None, config: Dict = None, reinit=False):
        super().__init__(context, config, reinit)
        self.left_bound, self.right_bound, self.pow, self.step_size, self.noise, self.drift = context
        self.context = context
        self.T = 100
        self._mid = 0.5 * (self.left_bound + self.right_bound)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(np.full((7,), -np.inf),
                                                np.full((7,), np.inf), dtype=np.float32)
        self.x = np.array([self._mid])
        self.steps = 0
        self.done = False

    def reset_context(self, context):
        self.left_bound, self.right_bound, self.pow, self.step_size, self.noise, self.drift = context
        self.context = np.array(context)
        self._mid = 0.5 * (self.left_bound + self.right_bound)
        self.reset()

    def reset(self):
        self.x = np.array([self._mid])
        self.steps = 0
        self.done = False
        return np.concatenate((self.x, self.context))

    def step(self, action: int):
        """next_x = x + N(0, noise) - drift * x + 2*(a-0.5) * step"""
        self.x = self.x + np.random.normal(loc=0, scale=self.noise, size=1) \
                 - self.drift * self.x \
                 + 2 * (action - 0.5) * self.step_size
        if self.x > self.right_bound or self.x < self.left_bound:
            r = -1
            self.done = True
        else:
            r = np.abs(self.x - self._mid) ** 2
        if self.steps >= self.T-1:
            self.done = True
        r = float(r)
        self.steps += 1
        return np.concatenate((self.x, self.context)), r, self.done, {}


class ContextualCliff(cMDP, ABC):
    """
    ### Description
    A contextual Cliff cMDP

    ### Context Variables:
    left_bound: the left end of the cliff
    right_bound: the right end of the cliff
    pow: the reward r = (x - mid)**pow
    step_size: size of each moving step
    noise: noise N(0, noise) added to each move
    drift: drift term that pulls x back to mid point
    context = (left_bound, right_bound, pow, step_size, noise, drift)
    """

    def __init__(self, config=None):
        if config is None:
            config = {}
        config.setdefault('mdp_type', CliffMDP)
        config.setdefault('context_distribution',
                          ConstantDistribution(dim=6,
                                               constant_vector=np.array([-1, 1, 2, 0.1, 0.1, 0.03])))
        config.setdefault('env_config', {})
        super().__init__(config=config)

    def unpack_env_config(self, config):
        return {}

    def likelihood(self, obs_prev, action_prev, obs_curr):
        left_bound, right_bound, pow, step_size, noise, drift = self.context
        x_prev = obs_prev[0]
        x_curr = obs_curr[0]
        diff_term = x_curr - x_prev + drift * x_prev - 2*(action_prev-0.5) * step_size
        return stats.norm.pdf(diff_term, loc=0, scale=noise)


