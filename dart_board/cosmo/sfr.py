import numpy as np
from . import utilities
from .. import constants as c


def ln_prior_t_cosmo(ln_t_b, t_min=c.min_t, t_max=c.max_t):
    """
    Calculate the prior on the birth time based on the cosmological star formation rate evolution

    """

    t_b = np.exp(ln_t_b)

    if t_b < t_min or t_b > t_max: return -np.inf

    z = utilities.get_z_from_t(t_b)

    # From Madau & Dickinson (2014)
    sfr = 0.015 * (1.0 + z)**2.7 / (1.0 + ((1.0+z)/2.9)**5.6)


    return np.log( sfr*t_b )
