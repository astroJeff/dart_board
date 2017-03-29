# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import maxwell

from . import constants as c


def ln_prior_M1(M1):
    """
    Return the prior probability on M1: P(M1).

    """

    if M1 < c.min_mass or M1 > c.max_mass: return -np.inf
    norm_const = (c.alpha+1.0) / (np.power(c.max_mass, c.alpha+1.0) - np.power(c.min_mass, c.alpha+1.0))
    return np.log( norm_const * np.power(M1, c.alpha) )


def ln_prior_M2(M2, M1):
    """
    Return the prior probability on M2: P(M2 | M1).

    """

    if M2 < c.min_mass or M2 > M1: return -np.inf
    return np.log(1.0 / M1)


def ln_prior_a(a, ecc):
    """
    Return the prior probability on a: P(a).

    """

    if a*(1.0-ecc) < c.min_a or a*(1.0+ecc) > c.max_a: return -np.inf
    norm_const = np.log(c.max_a) - np.log(c.min_a)

    return np.log( norm_const / a )


def ln_prior_ecc(ecc):
    """
    Return the prior probability on ecc: P(ecc).

    """

    if ecc < 0.0 or ecc > 1.0: return -np.inf
    return np.log(2.0 * ecc)


def ln_prior_v_kick(v_kick):
    """
    Return the prior probability on v_kick: P(v_kick).

    """

    if v_kick < 0.0: return -np.inf
    return np.log(maxwell.pdf(v_kick, scale=c.v_k_sigma))


def ln_prior_theta_kick(theta_kick):
    """
    Return the prior probability on the SN kick theta: P(theta).

    """

    if theta_kick <= 0.0 or theta_kick >= np.pi: return -np.inf
    return np.log(np.sin(theta_kick) / 2.0)


def ln_prior_phi_kick(phi_kick):
    """
    Return the prior probability on the SN kick phi: P(phi).

    """

    if phi_kick < 0.0 or phi_kick > np.pi: return -np.inf
    return -np.log(np.pi)


def ln_prior_t(t_b):
    """
    Return the prior probability on the binary's birth time (age).

    """

    if t_b < c.min_t or t_b > c.max_t: return -np.inf
    return 0.0
