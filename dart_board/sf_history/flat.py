import numpy as np

from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d

from dart_board import constants as c
from . import lmc


lmc_dist = 5.0e4 * c.pc_to_km


def load_lmc_sf_history(z=0.008):
    """ Load star formation history data for the LMC

    Parameters
    ----------
    z : float
        Metallicity of star formation history
        Default = 0.008
    """

    coor = lmc.load_lmc_coor()
    pad = 0.2

    # Set the coordinate bounds for the LMC
    c.ra_min = min(coor['ra'])-pad
    c.ra_max = max(coor['ra'])+pad
    c.dec_min = min(coor['dec'])-pad
    c.dec_max = max(coor['dec'])+pad

    # Set distance to the LMC
    c.distance = lmc_dist


def prior_lmc(ra, dec, t_b):
    """ This function returns a flat star formation history for the lmc """

    if c.ra_min is None or c.ra_max is None or c.dec_min is None or c.dec_max is None:
        load_lmc_sf_history()

    # Positional boundaries
    if ra < c.ra_min or ra > c.ra_max or dec < c.dec_min or dec > c.dec_max:
        return -np.inf
    else:
        return np.log(1.0)
