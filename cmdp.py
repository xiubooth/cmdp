from abc import ABC, abstractmethod
from typing import Type, Dict, Any, Tuple, Union

import gym
import numpy as np
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

from utils.distributions import Distribution

gym.logger.set_level(40)


class MDP(gym.Env, ABC):
    """
    ### Description
    A Markov decision process
    """
    def __init__(self, context: np.ndarray = None,
                 config: Dict = None,
                 reinit=True,
                 ):
        """Initialize Gym env from context and config"""
        self.context = context
        self.config = config
        if reinit:
            assert all(k in config for k in ['action_space', 'observation_space'])
            self.action_space: gym.Space = config['action_space']
            self.observation_space: gym.Space = config['observation_space']
            self.state: np.ndarray = None
            self.observation: np.ndarray = None
            self.steps = 0
            self.done = False

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment like in gym"""

    @abstractmethod
    def step(self, action: np.ndarray) -> (np.ndarray, float, bool, dict):
        """Step using action like in gym"""


class cMDP(ABC, TaskSettableEnv):
    """
    ### Description
    A cMDP is a contextual Markov decision process [https://arxiv.org/pdf/1502.02259.pdf].
    Given a context vector c, one can obtain a unique MDP M(c) from cMDPs.

    ### Context sampling
    > self.sample_context()
    Each cMDP can have a context distribution P(c) from which c (and therefore MDPs) can be sampled from
    self.context_distribution.

    ### cMDPs are independent of MDPs
    cMDPs are independent of MDPs. It simply provides a context sampling interface.
    """

    def __init__(self, config: Dict = None):
        """Initialize cMDP from config dict
        config must contain: context_distribution, mdp_type"""
        if config is None:
            config = {}

        assert all(k in config for k in ['context_distribution', 'mdp_type', 'env_config'])
        self.context_distribution: Distribution = config['context_distribution']  # to sample context from
        self.mdp_type: Type[MDP] = config['mdp_type']  # the class of the MDPs to generate
        self.context: np.ndarray = self.sample_context()  # cache of the current context
        self.resample = True  # default allow resample

        self.env_config: Dict = self.unpack_env_config(config)
        self.mdp: Union[MDP, gym.Env] = self.mdp_type(context=self.context, config=self.env_config)
        self.observation_space = self.mdp.observation_space
        self.action_space = self.mdp.action_space

    @abstractmethod
    def unpack_env_config(self, config) -> Dict:
        """Unpack env_config from config; set default values
        Returns the dict env_config"""

    def sample_context(self) -> np.ndarray:
        """Sample a context array from the context distribution"""
        context_array = self.context_distribution.sample()
        self.context = context_array
        return context_array

    def to_resample(self, flag: bool):
        """Change resample flag"""
        self.resample = flag

    def get_task(self):
        return self.context

    def set_task(self, context_distribution):
        self.context_distribution = context_distribution

    def update_context(self, data: Dict = None):
        """Update the context distribution of cMDP given data"""
        self.context_distribution.update(data=data)

    def get_mdp(self, resample=False) -> MDP:
        """Return an MDP object"""
        if resample:
            self.sample_context()
        return self.mdp_type(context=self.context, config=self.env_config)

    def reset(self) -> np.ndarray:
        """Meta method to reset the current context MDP
        Default resample context for every cMDP"""
        if self.mdp is None or self.resample:
            self.mdp = self.get_mdp(resample=self.resample)
        observation = self.mdp.reset()
        return observation

    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, dict]:
        """Meta method to step the current context MDP"""
        obs, rew, done, info = self.mdp.step(action)
        return obs, rew, done, info
