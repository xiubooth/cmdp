import numpy as np

__all__ = ['log_ar1']


def log_ar1(z0: float, rho: float, sigma: float):
    """Model log(z) as an AR(1) with autocorrelation rho and std sigma"""
    lnz = np.log(z0)
    lnz_new = rho * lnz + np.random.normal(0, sigma, 1)[0]
    return np.exp(lnz_new)
