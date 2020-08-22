import numpy as np
from . import constants as c

# For calculating GR merger times
H_e = None

def P_to_A(M1, M2, P):
    """
    Calculate the orbital separation from the stellar masses and orbital period.

    Args:
        M1, M2 : floats, masses of the two stars (Msun)
        P : float, orbital period (days)

    Returns:
        a : float, orbital separation (Rsun)
    """

    mu = c.G * (M1 + M2) * c.Msun_to_g
    n = 2.0*np.pi / P / c.day_to_sec
    a = np.power(mu/(n*n), 1.0/3.0) / c.Rsun_to_cm

    return a


def A_to_P(M1, M2, a):
    """
    Calculate the orbital period from the stellar masses and orbital separation.

    Args:
        M1, M2 : floats, masses of the two stars (Msun)
        a : float, orbital separation (Rsun)

    Returns:
        P : float, orbital period (days)
    """

    mu = c.G * (M1 + M2) * c.Msun_to_g
    n = np.sqrt(mu/(a**3 * c.Rsun_to_cm**3))
    P = 2.0*np.pi / n

    return P / c.day_to_sec


def roche_radius(M1, M2, a):
    """ Calculate the Roche lobe radius using formulat from Eggleton (1983)

    Parameters
    ----------
    M1 : float
        Mass of the star for calculating the Roche radius
    M2 : float
        Mass of the companion star
    a : float
        Orbital separation

    Returns
    -------
    r_L : float
        Roche radius (same units as input parameter, a)
    """


    q = M1 / M2
    return a * 0.49*q**(2.0/3.0) / (0.6*q**(2.0/3.0) + np.log(1.0 + q**(1.0/3.0)))

def bondi_radius(M1, velocity):
    """ Calculate the Bondi radius

    Parameters
    ----------
    M1 : float
        Mass of the accreting object (Msun)
    velocity : float
        Velocity of the object through the material (km/s)

    Returns
    -------
    r_b : float
        Bondi radius

    """

    r_b = 2.0 * c.G * (M1 * c.Msun_to_g) / (velocity*1.0e5)**2 / c.Rsun_to_cm

    return r_b


def orbital_velocity(M1, M2, a):
    """ Calculate the orbital velocity of a binary

    Parameters
    ----------
    M1, M2 : float
        Masses of the primary and secondary stars in the binary (Msun)
    a : float
        Orbital separation of the binary (Rsun)

    Returns
    -------
    v_orb : float
        Orbital velocity (km/s)

    """

    v_orb = np.sqrt(c.G * (M1+M2) * c.Msun_to_g / (a * c.Rsun_to_cm)) / 1.0e5

    return v_orb


def flatten_chains(chains, temp=0, thin=None):
    """ A function to flatten (and possibly thin) an emcee or PTemcee chain (or blob)

    Parameters
    ----------
    chains : ndarray
        Chain output from emcee in 3 dimensions (or 4 dimensions if PTemcee)
    temp : int (optional)
        Array index cooresponding to the temperature of the data you want flattened
    thin : int (optional)
        Factor by which to thin the chain

    Returns
    -------
    chains : ndarray
        Flattened, 2 dimensional array of chains

    """


    if chains.ndim == 4:
        try:
            chains = chains[temp]
        except ValueError:
            print("You must provide a valid temperature for PT array")

    # Thin the chain, if specified
    if not thin is None: chains = thin_chains(chains, thin)

    n_chains, length, n_var = chains.shape
    chains = chains.reshape((n_chains*length, n_var))

    return chains



def thin_chains(chains, thin_factor):
    """ A function to thin an emcee or PTemcee chain (or blob)

    Parameters
    ----------
    chains : ndarray
        Chain output from emcee in 3 dimensions (or 4 dimensions if PTemcee)
    thin_factor : int
        Factor to reduce the array size by

    Returns
    -------
    chains : ndarray
        Thinned, 3 dimensional array of chains

    """

    if chains.ndim == 4:
        return chains[:,:,::thin_factor,:]
    else:
        return chains[:,::thin_factor,:]


##############################################################################
##################                                          ##################
################## Gravitational Wave Merger Time Functions ##################
##################                                          ##################
##############################################################################

def integrand_ecc(e):
    return e**(29./19.) * (1.0 + 121./304.*e*e)**(1181./2299.) / (1.0-e*e)**(1.5)

def H_e_full(e):

    from scipy.integrate import quad

    term_1 = ((1.0 + 121./304.*e*e)**(-870./2299.))**4.
    term_2 = ((1.0-e*e) * e**(-12./19.))**4.
    term_3 = quad(integrand_ecc, 0.0, e, epsabs=1.0e-60, epsrel=1.0e-60)[0]

    return term_1 * term_2 * term_3


def calc_merger_time(M1, M2, A_f=None, P_orb=None, ecc=0.0):
    """ Calculate the merger time due to gravitational waves (Myr) """

    global H_e
    if H_e is None:
        merger_time_set_up()

    if A_f is None and P_orb is None:
        raise ValueError("You must provide either an orbital separation, A_f, or an orbital period, P_orb.")

    if A_f is None:
        A_f = P_to_A(M1, M2, P_orb)

    beta = 64./5. * c.G**3 * M1 * M2 * (M1+M2) * c.Msun_to_g**3 / (c.c_light)**5

    return 12./19. * (A_f * c.Rsun_to_cm)**4 / beta * H_e(ecc) / (1.0e6 * c.yr_to_sec)



def merger_time_set_up():

    from scipy.interpolate import interp1d

    global H_e

    # Calculate the interpolation function of H(e)
    ecc_set = np.linspace(1.0e-4, 0.9, 50)
    ecc_set = np.append(ecc_set, 1.0 - 10**np.linspace(-1.1, -10, 50))
    H = np.zeros(100)
    for i, ecc in enumerate(ecc_set): H[i] = H_e_full(ecc)

    ecc_set = np.append(0.0, ecc_set)
    H = np.append(H[0], H)
    ecc_set = np.append(ecc_set, 1.0)
    H = np.append(H, 0.0)

    H[ecc_set>0.999] = 304./425. * (1.0-ecc_set[ecc_set>0.999]**2)**3.5

	# Make sure no negative values exist
    H = np.clip(H,a_min=0.0, a_max=1.0e10)

    H_e = interp1d(ecc_set, H, bounds_error=False, fill_value=0.0, kind='cubic')
