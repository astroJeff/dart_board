import numpy as np
from scipy.stats import maxwell, uniform

from . import constants as c
#from . import darts
from .utils import A_to_P

def generate_population(dart, N, ra_in=None, dec_in=None):
    """
    Generate initial conditions for a set of binaries.

    Args:
        dart : DartBoard instance
        N : int, number of binary initial conditions to generate
        ra_in : float (optional, default: None), observed system right ascension
        dec_in : float (optional, default: None), observed system declination

    Returns:
        tuple of initial binary conditions
    """

    # Binary masses
    M1 = dart.generate_M1(N)
    M2 = dart.generate_M2(M1, N)

    # Kick vector
    v_kick1 = dart.generate_v_kick1(N)
    theta_kick1 = dart.generate_theta_kick1(N)
    phi_kick1 = dart.generate_phi_kick1(N)

    v_kick2 = dart.generate_v_kick2(N)
    theta_kick2 = dart.generate_theta_kick2(N)
    phi_kick2 = dart.generate_phi_kick2(N)

    # Orbit parameters
    a = np.zeros(N)
    ecc = np.zeros(N)
    for i in range(N):
        ecc[i] = dart.generate_ecc(1)
        a[i] = dart.generate_a(1)
        while (a[i]*(1.0-ecc[i]) < c.min_a or a[i]*(1.0+ecc[i]) > c.max_a):
            ecc[i] = dart.generate_ecc(1)
            a[i] = dart.generate_a(1)

    # Calculate the orbital period
    orbital_period = A_to_P(M1, M2, a)


    # Generate (spatially-resolved?) star formation history
    if dart.generate_pos is None:
        ra = np.zeros(N)
        dec = np.zeros(N)
        if dart.binary_type=='HMXB' or dart.binary_type=='NSHMXB' or dart.binary_type=='BHHMXB':
            t_b = dart.generate_t(N, max_time=70.0)
        else:
            t_b = dart.generate_t(N)
    else:
        ra = np.zeros(N)
        dec = np.zeros(N)
        if dart.binary_type=='HMXB' or dart.binary_type=='NSHMXB' or dart.binary_type=='BHHMXB':
            t_b = dart.generate_t(N, max_time=70.0)
        else:
            t_b = dart.generate_t(N)

        for i in range(N):
            ra[i], dec[i], N_stars = dart.generate_pos(1, t_b[i], ra_in=ra_in, dec_in=dec_in)


    return M1, M2, orbital_period, ecc, v_kick1, theta_kick1, phi_kick1, \
            v_kick2, theta_kick2, phi_kick2, ra, dec, t_b



# Define random deviate functions
def get_v_kick(N, sigma=c.v_kick_sigma):
    """
    Generate random kick velocities from a Maxwellian distribution.

    Args:
        N : int, number of random samples to generate
        sigma : float (default: constants.v_kick_sigma), Maxwellian dispersion
            velocity (km/s)

    Returns:
        v_kick : ndarray, array of random kick velocities (km/s)
    """

    return maxwell.rvs(scale=sigma, size=N)

def get_theta(N):
    """
    Generate N random polar angles. It is assumed to be isotropic.

    Args:
        N : int, number of random samples to generate

    Returns:
        theta : ndarray, array of random kick polar angles (radians)
    """

    return np.arccos(1.0-2.0*uniform.rvs(size = N))

def get_phi(N):
    """
    Generate N random azimuthal angles. It is assumed to be isotropic.

    Args:
        N : int, number of random samples to generate

    Returns:
        phi : ndarray, array of random kick azimuthal angles (radians)
    """

    return 2.0*np.pi*uniform.rvs(size = N)

def get_M1(N, alpha=c.alpha, M1_min=c.min_mass_M1, M1_max=c.max_mass_M1):
    """ Generate random primary masses. The inversion method is used.

    Args:
        N : int, number of random samples to generate
        alpha : float (default: constants.alpha), IMF power law index
        M1_min : float (default: constants.min_mass_M1), minimum primary star mass (Msun)
        M1_max : float (default: constants.max_mass_M1), maximum primary star mass (Msun)

    Returns:
        M1 : ndarray, array of random primary masses (Msun)
    """

    A = (alpha+1.0) / (np.power(M1_max, alpha+1.0) - np.power(M1_min, alpha+1.0))
    x = uniform.rvs(size = N)

    return np.power(x*(alpha+1.0)/A + np.power(M1_min, alpha+1.0), 1.0/(alpha+1.0))

def get_M2(M1, N, M2_min=c.min_mass_M2, M2_max=c.max_mass_M2):
    """ Generate random secondary masses. The inversion method is used.

    Args:
        M1 : ndarray, array of primary masses (Msun)
        N : int, number of random samples to generate
        M2_min : float (default: constants.min_mass_M2), minimum secondary star mass (Msun)
        M2_max : float (default: constants.max_mass_M2), maximum secondary star mass (Msun)

    Returns:
        M2 : ndarray, array of random secondary masses (Msun)
    """

    # Limits on M2
    M2_max = np.min((np.ones(N)*M2_max, M1), axis=0)

    # Flat in mass ratio
    return (M2_max - M2_min) * uniform.rvs(size = N) + M2_min

def get_a(N, a_min=c.min_a, a_max=c.max_a):
    """ Generate random orbital separations. The inversion method is used.

    Args:
        N : int, number of random samples to generate
        a_min : float (default: constants.min_a), minimum orbital separation (Rsun)
        a_max : float (default: constants.max_a), maximum orbital separation (Rsun)

    Returns:
        a : ndarray, array of random orbital separations (Rsun)
    """


    C = 1.0 / (np.log(a_max) - np.log(a_min))
    x = uniform.rvs(size=N)

    return a_min * np.exp(x / C)

def get_ecc(N):
    """
    Generate N random eccentricities.

    Args:
        N : int, number of random samples to generate

    Returns:
        ecc : ndarray, array of random eccentricities
    """

    return np.sqrt(uniform.rvs(size=N))

def get_t(N, min_time=c.min_t, max_time=c.max_t):
    """
    Generate N random birth times.

    Args:
        N : int, number of random samples to generate
        min_time : float (default: constants.min_t), minimum birth time (Myr)
        max_time : float (default: constants.max_t), maximum birth time (Myr)

    Returns:
        t_b : ndarray, array of random birth times (Myr)
    """

    return (max_time - min_time) * uniform.rvs(size=N) + min_time

def get_z(t_b, N, min_z=c.min_z, max_z=c.max_z):
    """
    Generate N random metallicities.

    Args:
        t_b : ndarray, array of birth times (Myr)
        N : int, number of random samples to generate
        min_z : float (default: constants.min_z), minimum metallicity
        max_z : float (default: constants.max_z), maximum metallicity

    Returns:
        z : ndarray, array of random metallicities
    """

    C = 1.0 / (np.log(max_z) - np.log(min_z))
    x = uniform.rvs(size=N)

    return min_z * np.exp(x / C)

def get_new_ra_dec(ra, dec, theta_proj, pos_ang):
    """
    Find the new ra, dec from an initial ra, dec and how it moved in angular
    distance and position angle.

    Args:
        ra : float, right ascension birth place (degrees)
        dec : float, declination birth place (degrees)
        theta_proj : float, projected distance traveled (radians)
        pos_angle : float, position angle moved (radians)

    Returns:
        ra_out : float, new right ascension (degrees)
        dec_out : float, new declination (degrees)
    """

    delta_dec = theta_proj * np.cos(pos_ang)
    delta_ra = theta_proj * np.sin(pos_ang) / np.cos(c.deg_to_rad * dec)

    ra_out = ra + c.rad_to_deg * delta_ra
    dec_out = dec + c.rad_to_deg * delta_dec

    return ra_out, dec_out
