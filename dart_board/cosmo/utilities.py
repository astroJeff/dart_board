import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d


########### Module Constants ############
# From Planck 2015 results
h = 0.68
Omega_m = 0.31
Omega_l = 0.69
Omega_b_h_2 = 0.0223
H0 = 100.0*h   # km/s/Mpc
H0_inverse = 9.78e9/h

# From Madau & Dickinson (2014)
rho_b = 2.77e11*Omega_b_h_2   # Msun/Mpc^3
y = 0.023   # from Salpeter IMF 10-60 Msun
R = 0.29   # from Salpeter IMF 10-60 Msun
Z_sun = 0.02   # Solar metallicity
########### Module Constants ############



z_from_t_interp = None


def H_z(z):
    return H0 * np.sqrt(Omega_m * (1.0+z)**3 + Omega_l)

def calc_rho_z(z):

    def integrand(z):
        return calc_SFR(z) / (H_z(z) * (1.0+z)) * (H0_inverse*H0)

    rho, rho_err = quad(integrand, z, 200)

    return (1.0-R) * rho

def calc_SFR(z):
    return 0.015 * (1.0+z)**2.7 / (1.0 + ((1.0+z)/2.9)**5.6)


def calc_z(t):
    """

    """

    def integrand(z):
        return 1.0 / (H_z(z) * (1.0+z)) * (H0_inverse*H0)

    z, z_err = quad(integrand, z, 200)


def calc_lookback_time(z):
    """
    Calculate the lookback time (yrs) from the redshift
    """

    def integrand(z):
        return 1.0 / (H_z(z) * (1.0 + z)) * (H0_inverse*H0)

    t, t_err = quad(integrand, 0, z)

    return t

def initialize():
    """
    Initialize the interpolation function to provide the redshift from the lookback time

    """

    global z_from_t_interp

    # Logarithmic spacing
    log_z_set = np.linspace(0.0, 3.0, 300)
    z_set = 10**(log_z_set) - 1.0

    t_set = np.zeros(len(z_set))
    for i, z in enumerate(z_set):
        t_set[i] = calc_lookback_time(z) / 1.0e6  # in Myr

    z_from_t_interp = interp1d(t_set, z_set, bounds_error=False, fill_value=100.0)



def get_z_from_t(t):
    """
    Provide the redshift from the lookback time.

    Parameters
    ----------
    t : float
        lookback time (Myr)

    Returns
    -------
    z : float
        redshift

    """


    global z_from_t_interp

    if z_from_t_interp is None: initialize()

    return z_from_t_interp(t)
