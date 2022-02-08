from abc import ABC
from typing import Dict, Any, Union

import gym
import numpy as np

from cmdp import cMDP, MDP
from utils.distributions import ConstantDistribution
from utils.math_fn import log_ar1


class MDPWhited(MDP, ABC):

    def reset(self) -> np.ndarray:
        k_init, z_init = self.config['initial_state']
        self.state = np.array([k_init, z_init])
        self.observation = np.concatenate((np.array([k_init, z_init], dtype=np.float32),
                                           self.context), axis=0).flatten()
        self.steps = 0
        self.done = False
        return self.observation

    def step(self, action: Union[float, np.ndarray]):  # action is investment I_t over capital k_
        # Unpack context
        gamma, delta, theta, rho, sigma = self.context
        k_curr, z_curr = self.state
        if isinstance(action, np.ndarray):
            assert action.shape[0] == 1
            action = int(action[0])  # Discrete action denoting fraction of capital to invest
        i_curr = action * k_curr / (self.action_space.n - 1.)
        # update z
        z_new = self.config['shock_process_fn'](z_curr, rho, sigma)
        # update k
        k_new = (1 - delta) * k_curr + i_curr
        # reward function
        reward = self.config['pi_fn'](k_curr, z_curr, theta) - self.config['psi_fn'](i_curr, k_curr) - i_curr
        assert isinstance(reward, float)
        # new state and observations
        self.observation = np.concatenate((np.array([k_new, z_new], dtype=np.float32), self.context),
                                          axis=0).flatten()
        self.state = np.array([k_new, z_new])
        # add step
        self.steps += 1
        # decide to end or not
        if self.steps == self.config['max_steps']:
            self.done = True
        return self.observation, reward, self.done, {}


class cMDPWhited(cMDP, ABC):
    """
    ### Description
    Basic Model in Strebulaev and Whited (2012), *Dynamic Models and Structural Estimation*

        Define $e(k_t,I_t,z_t) = \pi(k_t,z_t)-\psi(I_t,k_t)-I_t$

        where $k_t$ is beginning of period capital stock, $I_t$ is investment in capital,
        $\pi(k_t,z_t)$ is a profit function, and $z_t$ is the exogenous shock that is Markov,
        and $\psi(I_t,k_t)$ Is an investment adjustment cost term.

        Firm solves
        $$\max_{k_{t+j}} E_t[\sum_{j=0}^{\infty} \gamma^j e(k,I,z)]$$ where $\gamma=(1/1+r)$
        s.t. $k_{t+1} = (1-\delta)k_t+I_t$.

        **Assume**
        - $\ln z_t$ is AR(1) with autocorrelation $\rho$ and variance $\sigma_{\epsilon}$
        - $\pi = zk^{\theta}$ where $0<\theta<1$
        - adjustment costs either 0 or fixed or convex

        **RL setup**
        - State $s_t = (k_t,z_t)$ (if Markov)
        - Control/action $I_t$
        - State transition: $k_{t+1} = (1-\delta)k_t+I_t$ and AR(1) for $z_t$
        - Reward: $e(s_t,a_t)$


    ### Context Variables (5 in total; all float)
    gamma: the discount rate
    delta: the capital depreciation rate
    theta: the concavity of production function (pi = z * (k**theta) )
    rho: the persistence of the shock process
    sigma: the standard deviation or noise of the shock process
    The context vector is an (ordered) array of (gamma, delta, theta, rho, sigma)
    Default: (0.98, 0.15, 0.7, 0.7, 0.15)

    ### MDP Variables
    States: k (capital), z (shock)
    Action: i (investment)
    Reward: e (cash flow)
    """

    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
        config.setdefault('mdp_type', MDPWhited)
        config.setdefault('context_distribution',
                          ConstantDistribution(dim=5, constant_vector=np.array([0.98, 0.15, 0.7, 0.7, 0.15])))
        config['env_config'] = self.unpack_env_config(config)

        super().__init__(config=config)

    def unpack_env_config(self, config):
        env_config = config.get('env_config', {})
        env_config.setdefault('action_space', gym.spaces.Discrete(20))
        env_config.setdefault('observation_space', gym.spaces.Box(np.zeros((7,)),
                                                                   np.full((7,), np.inf),
                                                                   dtype=np.float32))
        env_config.setdefault('max_steps', 100)
        env_config.setdefault('initial_state', (1., 1.))  # (k_0, z_0) time 0 capital and shock
        env_config.setdefault('psi_fn', lambda i, k: 0.)  # investment adjustment cost function psi(k, z)
        env_config.setdefault('pi_fn', lambda k, z, theta: z * (
                k ** theta))  # profit function pi(k, z) = z*k**theta
        env_config.setdefault('shock_process_fn', log_ar1)  # z process: z(z0, rho, sigma)
        return env_config
