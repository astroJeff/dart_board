import numpy as np
from scipy.stats import truncnorm
from . import utilities
from .. import constants as c



def calc_Z(z):
    return utilities.y*utilities.calc_rho_z(z)/utilities.rho_b


def ln_prior_z(ln_z_b, ln_t_b, z_min=c.min_z, z_max=c.max_z):
    """
    Return the prior probability on the log of the metallicity.

    """

    Z = np.exp(ln_z_b)
    t = np.exp(ln_t_b)

    if Z < z_min or Z > z_max: return -np.inf

    # Get redshift corresponding to age
    z_ref = utilities.get_z_from_t(t)

    # Get metallicity of the universe at that time
    Z_ref = calc_Z(z_ref)

    a, b = (np.log10(z_min) - np.log10(Z_ref)) / 0.5, (np.log10(z_max) - np.log10(Z_ref)) / 0.5
    return np.log(truncnorm.pdf(np.log10(Z), a, b, loc=np.log10(Z_ref), scale=0.5))
