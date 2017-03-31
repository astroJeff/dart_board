# -*- coding: utf-8 -*-

import sys
import numpy as np
from scipy.stats import norm

from . import constants as c
from . import darts
from . import priors


def P_to_A(M1, M2, P):
    """
    Orbital period (days) to separation (Rsun).

    """

    mu = c.G * (M1 + M2) * c.Msun_to_g
    n = 2.0*np.pi / P / c.day_to_sec
    a = np.power(mu/(n*n), 1.0/3.0) / c.Rsun_to_cm

    return a


def A_to_P(M1, M2, a):
    """
    Orbital separation (Rsun) to period (days).

    """

    mu = c.G * (M1 + M2) * c.Msun_to_g
    n = np.sqrt(mu/(a**3 * c.Rsun_to_cm**3))
    P = 2.0*np.pi / n

    return P / c.day_to_sec


def ln_posterior(x, dart):
    """ Calculate the natural log of the posterior probability

    Parameters
    ----------
    x : floats
        Model parameters

    dart : DartBoard
        Positions of the dart

    Returns
    -------
    lp : float
        Natural log of the posterior probability

    """


    if dart.second_SN:
        if dart.prior_pos is None:
            M1, M2, a, ecc, v_kick1, theta_kick1, phi_kick1, v_kick2, theta_kick2, phi_kick2, t_b = x
        else:
            M1, M2, a, ecc, v_kick1, theta_kick1, phi_kick1, v_kick2, theta_kick2, phi_kick2, ra_b, dec_b, t_b = x
    else:
        if dart.prior_pos is None:
            M1, M2, a, ecc, v_kick1, theta_kick1, phi_kick1, t_b = x
        else:
            M1, M2, a, ecc, v_kick1, theta_kick1, phi_kick1, ra_b, dec_b, t_b = x


    # Empty array for emcee blobs
    empty_arr = np.zeros(9)


    # Calculate the prior probability
    lp = priors.prior_probability(x, dart)
    if np.isinf(lp) or np.isnan(lp): return -np.inf, empty_arr


    # Get initial orbital period
    orbital_period = A_to_P(M1, M2, a)


    # Proxy values if binary_type does not include second SN
    if not dart.second_SN:
        v_kick2 = v_kick1
        theta_kick2 = theta_kick1
        phi_kick2 = phi_kick1


    # Run rapid binary evolution code
    output = dart.evolve_binary(1, M1, M2, orbital_period, ecc,
                                v_kick1, theta_kick1, phi_kick1,
                                v_kick2, theta_kick2, phi_kick2,
                                t_b, dart.metallicity, False)
    # m1_out, m2_out, a_out, ecc_out, v_sys, mdot, t_SN1, k1, k2 = output


    # Return posterior probability and blobs
    if not check_output(output, dart.binary_type): return -np.inf, empty_arr


    # Check for kwargs arguments
    ll = 0
    if not dart.kwargs == {}: ll = posterior_properties(output, **dart.kwargs)


    return ll+lp, np.array([output])




def posterior_properties(output, **kwargs):
    """
    Calculate the (log of the) posterior probability given specific observables.

    """

    M1, M2, a, ecc, v_sys, mdot, t_SN1, k1, k2 = output
    P_orb = A_to_P(M1, M2, a)


    # Calculate an X-ray luminosity
    L_x = calculate_L_x(M1, mdot, k1)


    def get_error_from_kwargs(param, **kwargs):

        # Search kwargs for parameter uncertainty
        for key, error in kwargs.items():
            if key == param + "_err":
                return error

        # Parameter uncertainty was not provided
        print("You must provide an uncertainty for the observables. Example:")
        print("M2 : 8.0, M2_err : 1.0")
        sys.exit(-1)


    # Possible observables
    observables = ["M1","M2","P_orb","a","ecc","L_x","v_sys"]
    model_vals = [M1,M2,P_orb,a,ecc,L_x,v_sys]


    # Add log probabilities for each observable
    ll = 0
    for key, value in kwargs.items():
        for i,param in enumerate(observables):
            if key == param:
                error = get_error_from_kwargs(param, **kwargs)
                ll += np.log(norm.pdf(model_vals[i], loc=value, scale=error))


    return ll



def calculate_L_x(M1, mdot, k1):
    """
    Calculate the X-ray luminosity of an accreting binary.

    """

    if k1 == 13:
        R_acc = c.R_NS * 1.0e5  # NS radius in cm
        epsilon = 1.0 # Surface accretion
        eta = 0.15 # Wind accretion
    elif k1 == 14:
        R_acc = 3.0 * 2.0 * c.G * (M1*c.Msun_to_g) / (c.c_light * c.c_light) # BH has accretion radius of 3 Schwarzchild radii
        epsilon = 0.5 # Disk accretion
        eta = 0.8 # All BH accretion
    else:
        return -1

    L_bol = epsilon * c.G * (M1*c.Msun_to_g) * mdot / R_acc

    L_x = eta * L_bol

    return L_x


def check_output(output, binary_type):
    """ Determine if the resulting binary from binary population synthesis
    is of the type desired.

    Parameters
    ----------
    M1_out, M2_out : float
        Masses of each object returned

    a_out, ecc_out : float
        Orbital separation and eccentricity

    v_sys : float
        Systemic velocity of the system

    L_x : float
        X-ray luminosity of the system

    k1, k2 : int
        K-types for each star

    Returns
    -------
    binary_type : bool
        Is the binary of the requested type?

    """

    m1_out, m2_out, a_out, ecc_out, v_sys, mdot, t_SN1, k1, k2 = output

    type_options = np.array(["HMXB", "BHBH", "NSNS", "BHNS"])

    if not np.any(binary_type == type_options):
        print("The code is not set up to detect the type of binary you are interested in")
        sys.exit(-1)

    if binary_type == "HMXB":
        if k1 != 13 and k1 != 14: return False
        if k2 > 9: return False
        if a_out <= 0.0: return False
        if ecc_out < 0.0 or ecc_out >= 1.0: return False
        if mdot <= 0.0: return False
        if m2_out < 4.0: return False

    elif binary_type == "BHBH":
        if k1 != 14 or k2 != 14: return False
        if a_out <= 0.0: return False
        if ecc_out < 0.0 or ecc_out >= 1.0: return False

    elif binary_type == "NSNS":
        if k1 != 13 or k2 != 13: return False
        if a_out <= 0.0: return False
        if ecc_out < 0.0 or ecc_out >= 1.0: return False

    elif binary_type == "BHNS":
        if (k1 != 14 or k2 != 13) and (k1 != 13 or k2 != 14): return False
        if a_out <= 0.0: return False
        if ecc_out < 0.0 or ecc_out >= 1.0: return False

    return True
