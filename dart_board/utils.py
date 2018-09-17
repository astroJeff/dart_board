import numpy as np
from . import constants as c

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
