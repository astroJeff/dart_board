import os
import numpy as np

from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d

from dart_board import constants as c
from .sf_plotting import get_plot_polar


sfh = None
coor = None
lmc_dist = 5.0e4 * c.pc_to_km



def load_lmc_data():
    """ Return LMC star formation history per unit steradian

    Returns
    -------
    lmc_sfh : np structured array
        LMC star formation history
        dtype: [('region','<S10'),
                ('log_age','<f8'),
                ('sfh_z008','<f8'),
                ('sfh_z004','<f8'),
                ('sfh_z0025','<f8'),
                ('sfh_z001','<f8')]
    """

    # Create an empty array to start with
    dtypes = [('region','<S10'), \
            ('log_age','<f8'), \
            ('sfh_z008','<f8'), \
            ('sfh_z004','<f8'), \
            ('sfh_z0025','<f8'), \
            ('sfh_z001','<f8')]
    lmc_data = np.recarray(0, dtype=dtypes)
    out_line = np.recarray(1, dtype=dtypes)

    # Test to load data
    this_dir, this_filename = os.path.split(__file__)
    file_path = os.path.join(this_dir, "lmc_sfh_reduced.dat")

    with open(file_path) as f:
#    with open("./lmc_sfh_reduced.dat") as f:
        line_num = 0

        for line in f:
            line_num += 1

            if line_num < 17: continue
            if "Region" in line:
                region = np.array(line.split()[2]).astype(np.str)
            elif "(" in line:
                1 == 1
            else:
                line_data = line.split()
                line_data = np.array(line_data).astype(np.float64)

                if "_" in str(region):
                    area = 1.218e-5  # radian^2
                else:
                    area = 4.874e-5  # radian^2

                out_line[0][0] = region
                out_line[0][1] = line_data[0]
                out_line[0][2] = line_data[1] / area
                out_line[0][3] = line_data[4] / area
                out_line[0][4] = line_data[7] / area
                out_line[0][5] = line_data[10] / area

                lmc_data = np.append(lmc_data, out_line[0])

    return lmc_data


def load_lmc_coor():
    """ Load coordinates to LMC regions

    Returns
    -------
    lmc_coor: np structured array
        Coordinates of LMC regions in degrees
        dtype: [('region','<S10'),
                ('ra','float64'),
                ('dec','float64')]
    """


    # Load data
    this_dir, this_filename = os.path.split(__file__)
    data_file = os.path.join(this_dir, "lmc_coordinates.dat")

    lmc_coor_2 = np.genfromtxt(data_file, dtype="S10,S2,S2,S3,S2")

    lmc_coor = np.recarray(0, dtype=[('region','<S10'),('ra','float64'),('dec','float64')])
    tmp = np.recarray(1, dtype=[('region','<S10'),('ra','float64'),('dec','float64')])


    for coor in lmc_coor_2:
        ra = str(coor[1].decode("utf-8"))+"h"+str(coor[2].decode("utf-8"))+"m"
        dec = str(coor[3].decode("utf-8"))+"d"+str(coor[4].decode("utf-8"))+"m"

        region = coor[0]

        coor = SkyCoord(ra, dec)

        tmp["region"] = region
        tmp["ra"] = coor.ra.degree
        tmp["dec"] = coor.dec.degree

        lmc_coor = np.append(lmc_coor, tmp)

    return lmc_coor




def load_lmc_sfh(z=0.008):
    """ Create array of 1D interpolations in time of the
    star formation histories for each region in the LMC.

    Parameters
    ----------
    z : float (0.001, 0.0025, 0.004, 0.008)
        Metallicity for which to return star formation history
        Default = 0.008

    Returns
    -------
    SF_history : ndarray
        Array of star formation histories for each region
    """


    # Load the LMC coordinates and SFH data
    lmc_data = load_lmc_data()

    regions = np.unique(lmc_data["region"])

    lmc_sfh = np.array([])
    age = np.array([])
    sfr = np.array([])
    for r in regions:

        age = lmc_data["log_age"][np.where(lmc_data["region"] == r)]

        if z == 0.008:
            sfr = lmc_data["sfh_z008"][np.where(lmc_data["region"] == r)]
        elif z == 0.004:
            sfr = lmc_data["sfh_z004"][np.where(lmc_data["region"] == r)]
        elif z == 0.0025:
            sfr = lmc_data["sfh_z0025"][np.where(lmc_data["region"] == r)]
        elif z == 0.001:
            sfr = lmc_data["sfh_z001"][np.where(lmc_data["region"] == r)]
        else:
            print("ERROR: You must choose an appropriate metallicity input")
            print("Possible options are 0.001, 0.0025, 0.004, 0.008")
            return -1

        lmc_sfh = np.append(lmc_sfh, interp1d(age[::-1], sfr[::-1], bounds_error=False, fill_value=0.0))

    return lmc_sfh





def load_sf_history(z=0.008):
    """ Load star formation history data for the LMC

    Parameters
    ----------
    z : float
        Metallicity of star formation history
        Default = 0.008
    """

    global coor
    global sfh

    coor = load_lmc_coor()
    sfh = load_lmc_sfh(z)
    pad = 0.2

    c.ra_min = min(coor['ra'])-pad
    c.ra_max = max(coor['ra'])+pad
    c.dec_min = min(coor['dec'])-pad
    c.dec_max = max(coor['dec'])+pad


def get_SFH(ra, dec, t_b):
    """ Returns the star formation rate in Msun/Myr for a sky position and age

    Parameters
    ----------
    ra : float64 or ndarray
        (Individual or ndarray of) right ascensions (degrees)
    dec : float64 or ndarray
        (Individual or ndarray of) declinations (degrees)
    t_b : float64 or ndarray
        (Individual or ndarray of) times (Myr)

    Returns
    -------
    SFH : float64 or ndarray
        Star formation history (Msun/Myr)
    """

    global coor
    global sfh


    if (coor is None) or (sfh is None): load_sf_history()

    if isinstance(ra, np.ndarray):

        ra1, ra2 = np.meshgrid(c.deg_to_rad * ra, c.deg_to_rad * coor["ra"])
        dec1, dec2 = np.meshgrid(c.deg_to_rad * dec, c.deg_to_rad * coor["dec"])

        dist = np.sqrt((ra1-ra2)**2*np.cos(dec1)*np.cos(dec2) + (dec1-dec2)**2)
        indices = dist.argmin(axis=0)

        SFR = np.zeros(len(ra))

        for i in np.arange(len(indices)):

            if ra[i]>c.ra_min and ra[i]<c.ra_max and dec[i]>c.dec_min and dec[i]<c.dec_max:
                SFR[i] = sfh[indices[i]](np.log10(t_b[i]*1.0e6))

        return SFR

    else:
        ra1 = c.deg_to_rad * ra
        dec1 = c.deg_to_rad * dec
        ra2 = c.deg_to_rad * coor["ra"]
        dec2 = c.deg_to_rad * coor["dec"]

        dist = np.sqrt((ra1-ra2)**2*np.cos(dec1)*np.cos(dec2) + (dec1-dec2)**2)

        # If outside the LMC, set to zero
        if ra<c.ra_min or ra>c.ra_max or dec<c.dec_min or dec>c.dec_max:
            return 0.0
        else:
            index = np.argmin(dist)
            return sfh[index](np.log10(t_b*1.0e6))


def prior_lmc(ra, dec, t_b):
    """
    Prior on position and time based on the spatially resolved star formation
    history maps of the LMC from Harris & Zaritsky (2004).

    """

    if c.ra_min is None or c.ra_max is None or c.dec_min is None or c.dec_max is None:
        load_sf_history()

    # Positional boundaries
    if ra < c.ra_min or ra > c.ra_max or dec < c.dec_min or dec > c.dec_max:
        return -np.inf

    # Get star formation history
    lp_pos = get_SFH(ra, dec, t_b)

    if lp_pos == 0:
        return -np.inf
    else:
        return np.log(lp_pos)


def plot_lmc_map(t_b, fig_in=None, ax=None, gs=None):

    xcenter, ycenter = 0.0, 21.0
    xwidth, ywidth = 5.0, 5.0

    rot_angle = 0.2

    sfh_levels = np.linspace(1.0e7, 2.0e8, 10)
    sfh_bins = 30

    get_plot_polar(t_b,
                   sfh_function=get_SFH,
                   fig_in=fig_in,
                   ax=ax,
                   gs=gs,
                   xcenter=xcenter,
                   ycenter=ycenter,
                   xwidth=xwidth,
                   ywidth=ywidth,
                   rot_angle=rot_angle,
                   sfh_bins=sfh_bins,
                   sfh_levels=sfh_levels)
