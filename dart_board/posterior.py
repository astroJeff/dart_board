# -*- coding: utf-8 -*-

import sys
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

import time as tm  # For testing

from .utils import P_to_A, A_to_P
from . import constants as c
# from . import darts
from . import priors


def ln_posterior(x, dart):
    """
    Calculate the natural log of the posterior probability. This function
    calls `ln_prior` to calculate the prior probability of model parameters and
    then `ln_likelihood` to evaluate the likelihood.

    Args:
        x : tuple, set of model parameters
        dart : DartBoard instance, positions of the dart

    Returns:
        lp : float, natural log of the posterior probability
    """

    # Empty array for emcee blobs
    empty_arr = np.zeros(17)


    # Calculate the prior probability
    lp = priors.ln_prior(x, dart)

    if dart.ntemps is None:

        if np.isinf(lp) or np.isnan(lp): return -np.inf, empty_arr

        ll, output = ln_likelihood(x, dart)

        # Convert from numpy structured array to a regular ndarray
        if len(output.shape) == 0:
            output = np.column_stack(output[name] for name in output.dtype.names)[0]


        return lp+ll, np.array([output])

    else:

        if np.isinf(lp) or np.isnan(lp): return -np.inf

        ll = ln_likelihood(x, dart)
        return lp+ll


def ln_likelihood(x, dart):
    """
    Calculate the natural log of the likelihood. This function first calls the
    binary evolution code given the set of model parameters, then checks to make
    sure the resulting binary is of the type specified by the user. Then, it
    calls the `posterior_properties` function to account for observations.

    Args:
        x : tuple, set of model parameters
        dart : DartBoard instance

    Returns:
        lp : float, natural log of the likelihood
    """

    # For use later in posterior properties
    x_in = x

    # Save model parameters to variables
    ln_M1, ln_M2, ln_a, ecc = x[0:4]
    x = x[4:]
    if dart.first_SN:
        v_kick1, theta_kick1, phi_kick1 = x[0:3]
        x = x[3:]
    else:
        v_kick1, theta_kick1, phi_kick1 = 0.0, 0.0, 0.0
    if dart.second_SN:
        v_kick2, theta_kick2, phi_kick2 = x[0:3]
        x = x[3:]
    if dart.prior_pos is not None:
        ra_b, dec_b = x[0:2]
        x = x[2:]
    if dart.model_metallicity:
        ln_z = x[0]
        z = np.exp(ln_z)
        x = x[1:]
    else:
        z = dart.metallicity
    ln_t_b = x[0]


    # Move from log vars to linear
    M1 = np.exp(ln_M1)
    M2 = np.exp(ln_M2)
    a = np.exp(ln_a)
    t_b = np.exp(ln_t_b)


    # Empty array for emcee blobs
    empty_arr = np.zeros(17)


    # Get initial orbital period
    orbital_period = A_to_P(M1, M2, a)


    # Proxy values if binary_type does not include second SN
    if not dart.second_SN:
        v_kick2 = v_kick1
        theta_kick2 = theta_kick1
        phi_kick2 = phi_kick1


    # Run rapid binary evolution code
    output = dart.evolve_binary(M1, M2, orbital_period, ecc,
                                v_kick1, theta_kick1, phi_kick1,
                                v_kick2, theta_kick2, phi_kick2,
                                t_b, z, False, **dart.model_kwargs)

    # Return posterior probability and blobs
    if not check_output(output, dart.binary_type):
        if dart.ntemps is None:
            return -np.inf, empty_arr
        else:
            return -np.inf


    # Check for kwargs arguments
    ll = 0
    if not dart.system_kwargs == {}: ll = posterior_properties(x_in, output, dart)

    if dart.ntemps is None:
        return ll, output
    else:
        return ll




def posterior_properties(x, output, dart):
    """
    Calculate the (log of the) likelihood given specific observables.
    This function evaluates the likelihood of the model parameters. This function
    calculates the likelihood for the model parameters to account for the
    observational constraints provided.

    Args:
        x : tuple, set of model parameters
        output : ndarray, binary evolution output
        dart : DartBoard class, provides the necessary functions

    Returns:
        likelihood : Likelihood of the set of model parameters.

    """

    P_orb_out = A_to_P(output['M1'], output['M2'], output['a'])


    # Save model parameters to variables
    ln_M1, ln_M2, ln_a, ecc = x[0:4]
    x = x[4:]
    if dart.first_SN:
        v_kick1, theta_kick1, phi_kick1 = x[0:3]
        x = x[3:]
    if dart.second_SN:
        v_kick2, theta_kick2, phi_kick2 = x[0:3]
        x = x[3:]
    if dart.prior_pos is not None:
        ra_b, dec_b = x[0:2]
        x = x[2:]
    if dart.model_metallicity:
        ln_z = x[0]
        z = np.exp(ln_z)
        x = x[1:]
    else:
        z = dart.metallicity
    ln_t_b = x[0]


    M1 = np.exp(ln_M1)
    M2 = np.exp(ln_M2)
    a = np.exp(ln_a)
    t_b = np.exp(ln_t_b)

    # Calculate an X-ray luminosity
    L_x_out = calculate_L_x(output['M1'], output['mdot1'], output['k1'])


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
    observables = ["M1", "M2", "M_tot", "P_orb", "a", "ecc", "L_x", "v_sys",
                   "R1", "R2", "Teff1", "Teff2", "L1", "L2",'Mdot1', 'Mdot2']
    model_vals = [output['M1'], output['M2'], output['M1']+output['M2'],
                  P_orb_out, output['a'], output['ecc'], L_x_out,
                  output['v_sys'], output['R1'], output['R2'], output['Teff1'],
                  output['Teff2'], output['L1'], output['L2'], output['mdot1'],
                  output['mdot2'] ]


    # Add log probabilities for each observable
    ll = 0
    for key, value in dart.system_kwargs.items():
        for i,param in enumerate(observables):
            if key == param:
                error = get_error_from_kwargs(param, **dart.system_kwargs)
                likelihood = norm.pdf(model_vals[i], loc=value, scale=error)
                if likelihood == 0.0: return -np.inf
                ll += np.log(likelihood)

        # Constraints on k-type
        if key == 'k1':
            if not output['k1'] in value: return -np.inf
        if key == 'k2':
            if not output['k2'] in value: return -np.inf

        # Mass function must be treated specially
        if key == "m_f_1":
            error = get_error_from_kwargs("m_f_1", **dart.system_kwargs)
            likelihood = calc_prob_from_mass_function(output['M1'], output['M2'], value, error)
            if likelihood == 0.0: return -np.inf
            ll += np.log(likelihood)

        if key == "m_f_2":
            error = get_error_from_kwargs("m_f_2", **dart.system_kwargs)
            likelihood = calc_prob_from_mass_function(output['M2'], output['M1'], value, error)
            if likelihood == 0.0: return -np.inf
            ll += np.log(likelihood)

        if key == 'L_x_min':
            if L_x_out < value: return -np.inf
        if key == 'L_x_max':
            if L_x_out > value: return -np.inf

        if key == "ecc_max":
            if output['ecc'] > value: return -np.inf

        # Age of the compact object, based on e.g., modeling the SN remnant
        # as in e.g., Circ X-1
        if key == 'NS_age_max':
            if t_b - output['t_SN1'] > value: return -np.inf


    # Add log probabilities if position is provided
    if dart.ra_obs is not None and dart.dec_obs is not None:

        # Projected travel angle
        theta_proj = get_theta_proj(c.deg_to_rad*dart.ra_obs, c.deg_to_rad*dart.dec_obs, \
                                    c.deg_to_rad*ra_b, c.deg_to_rad*dec_b)

        # Travel time in seconds
        t_sn = (t_b - output['t_SN1']) * 1.0e6 * c.yr_to_sec

        # Maximum angle in radians
        angle_max = (output['v_sys'] * t_sn) / c.distance

        # Define conditional
        conds = [theta_proj>=angle_max, theta_proj<angle_max]
        funcs = [lambda theta_proj: -np.inf,
                 lambda theta_proj: np.log(np.tan(np.arcsin(theta_proj/angle_max))/angle_max)]
                #  lambda theta_proj: np.log( theta_proj / angle_max**2 / np.sqrt(1.0 - (theta_proj/angle_max)**2) )]

        # Jacobian for coordinate change - ra, dec in radians
        J_coor = np.abs(get_J_coor(c.deg_to_rad*dart.ra_obs, c.deg_to_rad*dart.dec_obs, \
                        c.deg_to_rad*ra_b, c.deg_to_rad*dec_b))
        J_coor *= c.deg_to_rad * c.deg_to_rad  # Change jacobian from radians to degrees

        P_omega = 1.0 / (2.0 * np.pi)

        # Likelihood
        ll += np.piecewise(theta_proj, conds, funcs) + np.log(P_omega) + np.log(J_coor)

    return ll



def calculate_L_x(M1, mdot, k1):
    """
    Calculate the X-ray luminosity of an accreting binary.

    Args:
        M1 : float, primary mass (Msun)
        mdot : float, accretion rate onto the primary (Msun/yr)
        k1 : int, stellar type of primary (13=NS, 14=BH)

    Returns:
        L_x : float, X-ray luminosity (erg/s)
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

    L_bol = epsilon * c.G * (M1*c.Msun_to_g) * (mdot*c.Msun_to_g/c.yr_to_sec) / R_acc

    L_x = eta * L_bol

    return L_x


def calc_prob_from_mass_function(M1, M2, f_obs, f_err):
    """
    Calculate the contribution to the posterior probability for an observation
    of the mass function.

    Args:
        M1 : float, primary mass (Msun)
        M2 : float, secondary mass (Msun)
        f_obs : float, mass function (Msun)
        f_err : float, uncertainty on mass function (Msun)

    Returns:
        P_ln_f : float, likelihood of mass function observation.
    """

    # Function to calculate h from Equation 15 from Andrews et al. 2014, ApJ, 797, 32
    def model_h(M1, M2, f):

        Mtot = M1 + M2

        h_out = np.zeros(len(f))

        # idx = np.where(f > 0.0)[0]
        # idx = idx[M2*M2 > (f[idx]*Mtot*Mtot)**(2.0/3.0)]
        idx = M2*M2 > (f*Mtot*Mtot)**(2.0/3.0)

        numerator = Mtot**(4.0/3.0)
        denominator = 3.0 * f[idx]**(1.0/3.0) * M2 * np.sqrt(M2*M2 - (f[idx]*Mtot*Mtot)**(2.0/3.0))

        h_out[idx] = numerator / denominator
        #
        # print(len(h_out))
        # print(len(h_out[M2*M2 > (f*Mtot*Mtot)**(2.0/3.0)]))
        # h_out[M2*M2 > (f*Mtot*Mtot)**(2.0/3.0)] = Mtot**(4.0/3.0) / (3.0 * f**(1.0/3.0) * M2 * np.sqrt(M2*M2 - (f*Mtot*Mtot)**(2.0/3.0)))

        # if M2*M2 < (f*Mtot*Mtot)**(2.0/3.0): return 0.0
        #
        # numerator = Mtot**(4.0/3.0)
        # denominator = 3.0 * f**(1.0/3.0) * M2 * np.sqrt(M2*M2 - (f*Mtot*Mtot)**(2.0/3.0))
        #
        # return numerator / denominator

        return h_out


    # Integrand in which h is multiplied by the error on the mass function
    def func_integrand(f, f_obs, f_err, M1, M2):

        h = model_h(M1, M2, f)

        obs = norm.pdf(f, loc=f_obs, scale=f_err)

        return obs * h

    # Wrapper for integration
    def calc_f(f_obs, f_err, M1, M2):

        # Calculate using scipy.integrate.quad
        # # Limit integral to 5-sigma and/or 0
        # f_min = max(0.0, f_obs - 5.0*f_err)
        # f_max = f_obs + 5.0*f_err
        #
        # args = f_obs, f_err, M1, M2
        #
        # result = quad(func_integrand, f_min, f_max, args=args, epsabs=1.0e-04)
        # val = result[0]

        # Calculate using Monte Carlo
        N = 100000
        ran_f = norm.rvs(size=N, loc=f_obs, scale=f_err)
        val = np.mean(model_h(M1, M2, ran_f))

        return val


    likelihood = calc_f(f_obs, f_err, M1, M2)

    return likelihood


def check_output(output, binary_type):
    """
    Determine if the resulting binary from binary population synthesis is of
    the type desired.

    Args:
        M1_out, M2_out : float, masses of each object returned (Msun)
        a_out, ecc_out : float, orbital separation (arbitrary - scaled but not
            translated - units) and eccentricity
        v_sys : float, systemic velocity of the system (km/s)
        L_x : float, X-ray luminosity of the system (erg/s)
        k1, k2 : int, K-types for each star

    Returns:
        binary_type : bool, is the binary of the requested type?
    """

    type_options = np.array(["BH", "NS",
                             "BHHMXB", "NSHMXB", "HMXB", "LMXB", "BHBH", "NSNS",
                             "BHNS", "WDWD", "ELMWD", "ELMWD_WD", "nondegenerate"])

    if not np.any(binary_type == type_options):
        print("The code is not set up to detect the type of binary you are interested in")
        sys.exit(-1)

    if binary_type == "BH":
        if output['k1'] != 14: return False
        if output['a'] <= 0.0: return False
        if output['ecc'] < 0.0 or output['ecc'] >= 1.0: return False

    if binary_type == "NS":
        if output['k1'] != 13: return False
        if output['a'] <= 0.0: return False
        if output['ecc'] < 0.0 or output['ecc'] >= 1.0: return False

    if binary_type == "HMXB":
        if output['k1'] != 13 and output['k1'] != 14: return False
        if output['k2'] > 9: return False
        if output['a'] <= 0.0: return False
        if output['ecc'] < 0.0 or output['ecc'] >= 1.0: return False
        if output['mdot1'] <= 0.0: return False
        if output['M2'] < 6.0: return False

    if binary_type == "BHHMXB":
        if output['k1'] != 14: return False
        if output['k2'] > 9: return False
        if output['a'] <= 0.0: return False
        if output['ecc'] < 0.0 or output['ecc'] >= 1.0: return False
        if output['mdot1'] <= 0.0: return False
        if output['M2'] < 6.0: return False

    if binary_type == "NSHMXB":
        if output['k1'] != 13: return False
        if output['k2'] > 9: return False
        if output['a'] <= 0.0: return False
        if output['ecc'] < 0.0 or output['ecc'] >= 1.0: return False
        if output['mdot1'] <= 0.0: return False
        if output['M2'] < 6.0: return False

    elif binary_type == "LMXB":
        if output['k1'] != 13 and output['k1'] != 14: return False
        if output['k2'] > 9: return False
        if output['a'] <= 0.0: return False
        if output['ecc'] < 0.0 or output['ecc'] >= 1.0: return False
        if output['mdot1'] <= 0.0: return False
        if output['M2'] > 6.0: return False

    elif binary_type == "BHBH":
        if output['k1'] != 14 or output['k2'] != 14: return False
        if output['a'] <= 0.0: return False
        if output['ecc'] < 0.0 or output['ecc'] >= 1.0: return False

    elif binary_type == "NSNS":
        if output['k1'] != 13 or output['k2'] != 13: return False
        if output['a'] <= 0.0: return False
        if output['ecc'] < 0.0 or output['ecc'] >= 1.0: return False

    elif binary_type == "BHNS":
        if (output['k1'] != 14 or output['k2'] != 13) and (output['k1'] != 13 or output['k2'] != 14): return False
        if output['a'] <= 0.0: return False
        if output['ecc'] < 0.0 or output['ecc'] >= 1.0: return False

    elif binary_type == "WDWD":
        if output['k1'] < 10 or output['k1'] > 12 or output['k2'] < 10 or output['k2'] > 12: return False
        if output['a'] <= 0.0: return False
        if output['ecc'] < 0.0 or output['ecc'] >= 1.0: return False

    elif binary_type == "ELMWD":
        if output['k1'] != 10 and output['k2'] != 10: return False
        if output['a'] <= 0.0: return False
        if output['ecc'] < 0.0 or output['ecc'] >= 1.0: return False

    elif binary_type == "ELMWD_WD":
        if output['k1'] < 10 or output['k1'] > 12: return False
        if output['k2'] < 10 or output['k2'] > 12: return False
        if output['k1'] != 10 and output['k2'] != 10: return False
        if output['a'] <= 0.0: return False
        if output['ecc'] < 0.0 or output['ecc'] >= 1.0: return False

    elif binary_type == "nondegenerate":
        if output['k1'] > 9 or output['k2'] > 9: return False
        if output['a'] <= 0.0: return False
        if output['ecc'] < 0.0 or output['ecc'] >= 1.0: return False

    return True


def get_theta_proj(ra, dec, ra_b, dec_b):
    """
    Get the angular distance between two coordinates.

    Args:
        ra : float, right ascension of coordinate 1 (radians).
        dec : float, declination of coordinate 1 (radians).
        ra_b : float, right ascension of coordinate 2 (radians).
        dec_b : float, declination of coordinate 2 (radians).

    Returns:
        theta : float, angular separation of the two coordinates (radians).
    """

    return np.sqrt((ra-ra_b)**2 * np.cos(dec)*np.cos(dec_b) + (dec-dec_b)**2)


# Functions for coordinate jacobian transformation
def get_dtheta_dalpha(alpha, delta, alpha_b, delta_b):
    """
    Calculate the coordinate transformation derivative dtheta/dalpha.

    Args:
        alpha : float, right ascension of coordinate 1 (radians).
        delta : float, declination of coordinate 1 (radians).
        alpha_b : float, right ascension of coordinate 2 (radians).
        delta_b : float, declination of coordinate 2 (radians).

    Returns:
        dtheta_dalpha : float, coordinate transformation derivative dtheta/dalpha.
    """

    theta_proj = get_theta_proj(alpha, delta, alpha_b, delta_b)
    return (alpha_b-alpha) * np.cos(delta) * np.cos(delta_b) / theta_proj

def get_dtheta_ddelta(alpha, delta, alpha_b, delta_b):
    """
    Calculate the coordinate transformation derivative dtheta/ddelta.

    Args:
        alpha : float, right ascension of coordinate 1 (radians).
        delta : float, declination of coordinate 1 (radians).
        alpha_b : float, right ascension of coordinate 2 (radians).
        delta_b : float, declination of coordinate 2 (radians).

    Returns:
        dtheta_ddelta : float, coordinate transformation derivative dtheta/ddelta.
    """

    theta_proj = get_theta_proj(alpha, delta, alpha_b, delta_b)
    # return 1.0/(2.0*theta_proj) * (-np.cos(delta_b)*np.sin(delta)*(alpha_b-alpha)**2 + 2.0*(delta_b-delta))
    return (delta_b-delta) / theta_proj

def get_domega_dalpha(alpha, delta, alpha_b, delta_b):
    """
    Calculate the coordinate transformation derivative domega/dalpha.

    Args:
        alpha : float, right ascension of coordinate 1 (radians).
        delta : float, declination of coordinate 1 (radians).
        alpha_b : float, right ascension of coordinate 2 (radians).
        delta_b : float, declination of coordinate 2 (radians).

    Returns:
        domega/dalpha : float, coordinate transformation derivative domega/dalpha.
    """

    z = (delta_b-delta) / ((alpha_b-alpha) * np.cos(delta_b))
    return 1.0 / (1.0 + z*z) * z / (alpha_b - alpha)

def get_domega_ddelta(alpha, delta, alpha_b, delta_b):
    """
    Calculate the coordinate transformation derivative domega/ddelta.

    Args:
        alpha : float, right ascension of coordinate 1 (radians).
        delta : float, declination of coordinate 1 (radians).
        alpha_b : float, right ascension of coordinate 2 (radians).
        delta_b : float, declination of coordinate 2 (radians).

    Returns:
        domega/ddelta : float, coordinate transformation derivative domega/ddelta.
    """

    z = (delta_b-delta) / ((alpha_b-alpha) * np.cos(delta_b))
    return - 1.0 / (1.0 + z*z) / ((alpha_b-alpha) * np.cos(delta_b))

def get_J_coor(alpha, delta, alpha_b, delta_b):
    """
    Calculate the Jacobian (determinant of the jacobian matrix) of the
    coordinate transformation from right ascension and declination to
    angular separation and position angle.

    Args:
        alpha : float, right ascension of coordinate 1 (radians).
        delta : float, declination of coordinate 1 (radians).
        alpha_b : float, right ascension of coordinate 2 (radians).
        delta_b : float, declination of coordinate 2 (radians).

    Returns:
        J_coor : float, Jacobian coordinate transformation.
    """

    dt_da = get_dtheta_dalpha(alpha, delta, alpha_b, delta_b)
    dt_dd = get_dtheta_ddelta(alpha, delta, alpha_b, delta_b)
    do_da = get_domega_dalpha(alpha, delta, alpha_b, delta_b)
    do_dd = get_domega_ddelta(alpha, delta, alpha_b, delta_b)

    return dt_da*do_dd - dt_dd*do_da
