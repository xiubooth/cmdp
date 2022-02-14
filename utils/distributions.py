from abc import abstractmethod

import numpy as np

__all__ = ['Distribution', 'ConstantDistribution', 'ParticleDistribution', 'UniformDistribution']


class Distribution:
    """
    An implicit distribution over a vector of dimension 'dim'
    Only requires a sampling method to sample from the distribution
    And an update method to update the distribution from a 'data' dict
    """

    @abstractmethod
    def __init__(self, dim: int = 1):
        assert dim > 0
        self.dim = dim

    @abstractmethod
    def sample(self) -> np.ndarray:
        """Sample a np array of shape (dim,) from the distribution"""
        raise NotImplementedError

    @abstractmethod
    def update(self, data: dict = None) -> None:
        """Update distribution from data (which is a generic dict)"""
        raise NotImplementedError


class ConstantDistribution(Distribution):
    """Always returns a constant vector"""

    def __init__(self, dim: int = 1, constant_vector: np.ndarray = None):
        super().__init__(dim)
        if constant_vector is None:
            self.constant = np.ones((dim,)).flatten()
        else:
            self.constant = constant_vector

    def sample(self):
        return self.constant

    def update(self, data: dict = None) -> None:
        try:
            self.constant = data['constant_vector']
        except KeyError:
            print("Error: reset constant failed as key 'constant_vector' not found in data!")
            raise


class UniformDistribution(Distribution):
    """Returns a uniformly sampled vector between the upper and lower bounds"""

    def __init__(self, dim: int = 1, lower_bound_vector: np.ndarray = None, upper_bound_vector: np.ndarray = None):
        super().__init__(dim)
        self.lower_bound_vector = np.zeros((dim,)).flatten() if lower_bound_vector is None else lower_bound_vector
        self.upper_bound_vector = np.zeros((dim,)).flatten() if upper_bound_vector is None else upper_bound_vector
        assert lower_bound_vector.shape == upper_bound_vector.shape

    def sample(self):
        return np.random.uniform(self.lower_bound_vector, self.upper_bound_vector)

    def update(self, data: dict = None) -> None:
        try:
            self.lower_bound_vector = data['lower_bound_vector']
            self.upper_bound_vector = data['upper_bound_vector']
        except KeyError:
            print("Error: reset constant failed as key 'upper_bound_vector' or 'lower_bound_vector' not found in data!")
            raise


class ParticleDistribution(Distribution):
    """A distribution that keeps track of an size (n_particles, dim) array of particles
    See https://www.sas.upenn.edu/~jesusfv/ejemplo.pdf for an introduction"""

    def __init__(self, dim: int = 1, particles: np.ndarray = None, n_particles: int = 1000):
        super().__init__(dim=dim)
        self.particles = particles
        self.n_particles = n_particles
        if particles is None:
            self.particles = np.ones((n_particles, dim)).flatten()
        else:
            self.particles = particles

    def sample(self):
        return self.particles[np.random.randint(self.n_particles, size=1)]

    def update(self, data: dict = None) -> None:
        if 'p' in data:
            p = data['p']
            self.resample_particles_from_probability(p)
        elif 'resample_index' in data:
            resample_index = data['resample_index']
            self.resample_particles_from_index(resample_index)
        else:
            print("KeyError: reset constant failed as key 'p' or 'resample_index' not found in data!")
            raise

    def resample_particles_from_probability(self, p: np.ndarray) -> np.ndarray:
        """Resample particles and return resample index"""
        assert p.shape == (self.n_particles,), "probability shape does not match!"
        resample_index = np.random.choice(np.arange(self.n_particles), self.n_particles, p=p)
        self.particles = self.particles[resample_index]
        return resample_index

    def resample_particles_from_index(self, resample_index: np.ndarray) -> None:
        """Resample particles from index directly"""
        self.particles = self.particles[resample_index]
