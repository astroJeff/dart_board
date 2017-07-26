# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import maxwell

from . import constants as c


def ln_prior(x, dart):
    """
    Calculate all the prior probabilities.

    """


    if dart.second_SN:
        if dart.prior_pos is None:
            ln_M1, ln_M2, ln_a, ecc, v_kick1, theta_kick1, phi_kick1, v_kick2, theta_kick2, phi_kick2, ln_t_b = x
        else:
            ln_M1, ln_M2, ln_a, ecc, v_kick1, theta_kick1, phi_kick1, v_kick2, theta_kick2, phi_kick2, ra_b, dec_b, ln_t_b = x
    else:
        if dart.prior_pos is None:
            ln_M1, ln_M2, ln_a, ecc, v_kick1, theta_kick1, phi_kick1, ln_t_b = x
        else:
            ln_M1, ln_M2, ln_a, ecc, v_kick1, theta_kick1, phi_kick1, ra_b, dec_b, ln_t_b = x

    # Set defaults
    kick_sigma = c.v_kick_sigma
    M1_alpha = c.alpha
    mass_function = 'Salpeter'
    M1_min = c.min_mass_M1
    M1_max = c.max_mass_M1
    M2_min = c.min_mass_M2
    a_min = c.min_a
    a_max = c.max_a
    t_min = c.min_t
    t_max = c.max_t
    # End defaults

    # Set values according to inputs
    for key, value in dart.system_kwargs.items():
        if key == 'kick_sigma': kick_sigma = value
        if key == 'M1_alpha': M1_alpha = value
        if key == 'M1_min': M1_min = value
        if key == 'M1_max': M1_min = value
        if key == 'M2_min': M2_min = value
        if key == 'a_min': a_min = value
        if key == 'a_max': a_max = value
        if key == 't_min': t_min = value
        if key == 't_max': t_max = value
        if key == 'mass_function': mass_function = value


    # Calculate prior probabilities
    lp = 0.0
    lp += dart.prior_M1(ln_M1, mass_function=mass_function, alpha=M1_alpha, M1_min=M1_min, M1_max=M1_max)
    lp += dart.prior_M2(ln_M2, ln_M1, M2_min=M2_min)
    lp += dart.prior_ecc(ecc)
    lp += dart.prior_a(ln_a, ecc, a_min=a_min, a_max=a_max)
    lp += dart.prior_v_kick1(v_kick1, sigma=kick_sigma)
    lp += dart.prior_theta_kick1(theta_kick1)
    lp += dart.prior_phi_kick1(phi_kick1)


    if dart.second_SN:
        lp += dart.prior_v_kick2(v_kick2, sigma=kick_sigma)
        lp += dart.prior_theta_kick2(theta_kick2)
        lp += dart.prior_phi_kick2(phi_kick2)

    if dart.prior_pos is None:
        lp += dart.prior_t(ln_t_b, t_min=t_min, t_max=t_max)
    else:
        lp += dart.prior_pos(ra_b, dec_b, ln_t_b)

    return lp



def ln_prior_M1(M1, mass_function='Salpeter', alpha=c.alpha, M1_min=c.min_mass_M1, M1_max=c.max_mass_M1):
    """
    Return the prior probability on M1: P(M1).

    """

    if M1 < M1_min or M1 > M1_max: return -np.inf
    norm_const = (alpha+1.0) / (np.power(M1_max, alpha+1.0) - np.power(M1_min, alpha+1.0))
    return np.log( norm_const * np.power(M1, alpha) )

def ln_prior_ln_M1(ln_M1, mass_function='Salpeter', alpha=c.alpha, M1_min=c.min_mass_M1, M1_max=c.max_mass_M1):
    """
    Return the prior probability on the natural log of M1: P(ln_M1).

    """

    M1 = np.exp(ln_M1)

    if M1 < M1_min or M1 > M1_max: return -np.inf
    norm_const = (alpha+1.0) / (np.power(M1_max, alpha+1.0) - np.power(M1_min, alpha+1.0))
    return np.log( norm_const * np.power(M1, alpha+1.0) )

def ln_prior_M2(M2, M1, M2_min=c.min_mass_M2):
    """
    Return the prior probability on M2: P(M2 | M1).

    """

    if M2 < M2_min or M2 > M1: return -np.inf
    return np.log(1.0 / M1)

def ln_prior_ln_M2(ln_M2, ln_M1, M2_min=c.min_mass_M2):
    """
    Return the prior probability on the natural log of M2: P(ln_M2 | M1).

    """

    M1 = np.exp(ln_M1)
    M2 = np.exp(ln_M2)

    if M2 < M2_min or M2 > M1: return -np.inf
    return np.log(M2 / M1)

def ln_prior_a(a, ecc, a_min=c.min_a, a_max=c.max_a):
    """
    Return the prior probability on a: P(a).

    """

    if a*(1.0-ecc) < a_min or a*(1.0+ecc) > a_max: return -np.inf
    norm_const = 1.0 / (np.log(a_max / (1.0+ecc)) - np.log(a_min / (1.0-ecc)))

    return np.log( norm_const / a )

def ln_prior_ln_a(ln_a, ecc, a_min=c.min_a, a_max=c.max_a):
    """
    Return the prior probability on the natural log of a: P(ln_a).

    """

    a = np.exp(ln_a)

    if a*(1.0-ecc) < a_min or a*(1.0+ecc) > a_max: return -np.inf
    norm_const = 1.0 / (np.log(a_max / (1.0+ecc)) - np.log(a_min / (1.0-ecc)))

    return np.log( norm_const )

def ln_prior_ecc(ecc):
    """
    Return the prior probability on ecc: P(ecc).

    """

    if ecc < 0.0 or ecc > 1.0: return -np.inf
    return np.log(2.0 * ecc)


def ln_prior_v_kick(v_kick, sigma=c.v_kick_sigma):
    """
    Return the prior probability on v_kick: P(v_kick).

    """

    if v_kick < 0.0: return -np.inf
    return np.log(maxwell.pdf(v_kick, scale=sigma))


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


def ln_prior_t(t_b, t_min=c.min_t, t_max=c.max_t):
    """
    Return the prior probability on the binary's birth time (age).

    """

    if t_b < t_min or t_b > t_max: return -np.inf
    norm_const = 1.0 / (t_max - t_min)

    return np.log(norm_const)

def ln_prior_ln_t(ln_t_b, t_min=c.min_t, t_max=c.max_t):
    """
    Return the prior probability on the natural log of the binary's birth time (age).

    """

    t_b = np.exp(ln_t_b)

    if t_b < t_min or t_b > t_max: return -np.inf
    norm_const = 1.0 / (t_max - t_min)
    return np.log(norm_const * t_b)
