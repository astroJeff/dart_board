import os
import numpy as np

from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d
from scipy.stats import uniform

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
    # file_path = "lmc_sfh_reduced.dat"

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
    #data_file = "lmc_coordinates.dat"

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
    global lmc_dist


    coor = load_lmc_coor()
    sfh = load_lmc_sfh(z)
    pad = 0.2

    # Set the coordinate bounds for the LMC
    c.ra_min = min(coor['ra'])-pad
    c.ra_max = max(coor['ra'])+pad
    c.dec_min = min(coor['dec'])-pad
    c.dec_max = max(coor['dec'])+pad

    # Set distance to the LMC
    c.distance = lmc_dist


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


def prior_lmc(ra, dec, ln_t_b):
    """
    Prior on position and time based on the spatially resolved star formation
    history maps of the LMC from Harris & Zaritsky (2004).

    """

    # Used only when input variable is ln_t_b
    t_b = np.exp(ln_t_b)

    if c.ra_min is None or c.ra_max is None or c.dec_min is None or c.dec_max is None:
        load_sf_history()

    # Positional boundaries
    if ra < c.ra_min or ra > c.ra_max or dec < c.dec_min or dec > c.dec_max:
        return -np.inf

    # Get star formation history
    lp_pos = get_SFH(ra, dec, t_b)


    # TO DO: This probability is unnormalized. To fix it should be dividied by the number of stars in the LMC.
    if lp_pos == 0:
        return -np.inf
    else:
        return np.log(lp_pos * t_b)





def get_random_positions(N, t_b, ra_in=None, dec_in=None):
    """ Use the star formation history to generate a population of new binaries

    Parameters
    ----------
    N : integer
        Number of positions to calculate
    t_b : float
        Birth time to calculate star formation history (Myr)
    ra_in : float
        RA of system (optional)
    dec_in : float
        Dec of system (optional)

    Returns
    -------
    ra_out : ndarray
        Array of output RA's (degrees)
    dec_out : ndarray
        Array of output Dec's (degrees)
    N_stars : int
        Normalization constant calculated from number of stars formed at time t_b
    """

    global coor
    global sfh

    if sfh is None or coor is None:
        load_sf_history()

    N_regions = len(coor)

    # If provided with an ra and dec, only generate stars within 3 degrees of input position
    SF_regions = np.zeros((2,N_regions))
    for i in np.arange(N_regions):
        SF_regions[0,i] = i

        if ra_in is None or dec_in is None:
            SF_regions[1,i] = sfh[i](np.log10(t_b*1.0e6))
        elif sf_history.get_theta_proj_degree(coor["ra"][i], coor["dec"][i], ra_in, dec_in) < c.deg_to_rad * 3.0:
            SF_regions[1,i] = sfh[i](np.log10(t_b*1.0e6))
        else:
            SF_regions[1,i] = 0.0

    N_stars = np.sum(SF_regions, axis=1)[1]

    # Normalize
    SF_regions[1] = SF_regions[1] / N_stars

    # Sort
    SF_sort = SF_regions[:,SF_regions[1].argsort()]

    # Move from normed PDF to CDF
    SF_sort[1] = np.cumsum(SF_sort[1])

    # Random numbers
    y = uniform.rvs(size=N)

    # Create a 2D grid of CDFs, and random numbers
    SF_out, y_out = np.meshgrid(SF_sort[1], y)

    # Get index of closest region in sorted array
    indices = np.argmin((SF_out - y_out)**2,axis=1)

    # Move to indices of stored LMC SFH data array
    indices = SF_sort[0][indices].astype(int)

    # Get random ra's and dec's of each region
    ra_out = coor["ra"][indices]
    dec_out = coor["dec"][indices]

    # Width is 12 arcmin or 12/60 degrees for outermost regions
    # Width is 6 arcmin or 6/60 degrees for inner regions
#    width = 12.0 / 60.0 * np.ones(len(indices))
    width = 6.0 / 60.0 * np.ones(len(indices))
#    for i in np.arange(len(indices)):
#        if str(smc_coor["region"][indices[i]]).find("_") != -1:
#            width[i] = 6.0 / 60.0

    tmp_delta_ra = width * (2.0 * uniform.rvs(size=len(indices)) - 1.0) / np.cos(c.deg_to_rad * dec_out) * 2.0
    tmp_delta_dec = width * (2.0 * uniform.rvs(size=len(indices)) - 1.0)

    ra_out = ra_out + tmp_delta_ra
    dec_out = dec_out + tmp_delta_dec

    return ra_out, dec_out, N_stars

# def prior_lmc_position(x, dart):
#
#     if dart.second_SN:
#         M1, M2, a, ecc, v_kick1, theta_kick1, phi_kick1, v_kick2, theta_kick2, phi_kick2, ra_b, dec_b, t_b = x
#     else:
#         M1, M2, a, ecc, v_kick1, theta_kick1, phi_kick1, ra_b, dec_b, t_b = x
#
#     for key, value in dart.kwargs.items():
#         if key == "ra": ra_obs = value
#         if key == "dec": dec_obs = value
#
#
#     ############ Get the time limits of the binary ############
#     t_min, t_max = get_time_limits()
#
#     # Limits on time
#     if t_b < t_min or t_b > t_max: return -np.inf
#
#
#     ############ Evolve the binary ############
#     # Get initial orbital period
#     orbital_period = A_to_P(M1, M2, a)
#
#     # Proxy values if binary_type does not include second SN
#     if not dart.second_SN:
#         v_kick2 = v_kick1
#         theta_kick2 = theta_kick1
#         phi_kick2 = phi_kick1
#
#     # Call binary population synthsis code
#     output = dart.evolve_binary(1, M1, M2, orbital_period, ecc,
#                                 v_kick1, theta_kick1, phi_kick1,
#                                 v_kick2, theta_kick2, phi_kick2,
#                                 t_b, dart.metallicity, False)
#
#     M1_out, M2_out, a_out, ecc_out, v_sys, mdot_out, t_SN1, k1_out, k2_out = output
#
#
#     ############ Calculate the prior ############
#     theta_C = (v_sys * (t_max - t_min)) / c.distance
#
#     stars_formed = get_stars_formed(ra_obs, dec_obs, t_min, t_max, v_sys)
#
#     # Outside the region of star formation
#     if stars_formed == 0.0: return -np.inf
#
#     volume_cone = (np.pi/3.0 * theta_C**2 * (t_max - t_min) / c.yr_to_sec / 1.0e6)
#     sfh = get_SFH(ra, dec, t_b)
#
#     ln_pos = np.log(sfh / stars_formed / volume_cone)
#
#     return ln_pos
#
#


def plot_lmc_map(t_b, fig_in=None, ax=None, gs=None,
                 xcenter=0.0, ycenter=21.0,
                 xwidth=5.0, ywidth=5.0,
                 ra=None, dec=None):


    rot_angle = 0.2

    # We want sfh_levels in Msun/yr/deg.^2
    sfh_levels = np.linspace(1.0e7, 2.0e8, 10) / 1.0e6 * (np.pi/180.0)**2
    sfh_bins = 30

    sf_plot, ax1 = get_plot_polar(t_b,
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
                                  sfh_levels=sfh_levels,
                                  ra=ra,
                                  dec=dec)

    return sf_plot



def get_stars_formed(ra, dec, t_min, t_max, v_sys, N_size=512):
    """ Get the normalization constant for stars formed at ra and dec

    Parameters
    ----------
    ra : float
        right ascension input (decimals)
    dec : float
        declination input (decimals)
    t_min : float
        minimum time for a star to have been formed (Myr)
    t_max : float
        maximum time for a star to have been formed (Myr)
    v_sys : float
        Systemic velocity of system (km/s)

    Returns
    -------
    SFR : float
        Star formation normalization constant
    """

    ran_phi = 2.0 * np.pi * uniform.rvs(size = N_size)

    c_1 = 3.0 / np.pi / (t_max - t_min)**3 * (c.distance/v_sys)**2
    ran_x = uniform.rvs(size = N_size)
    ran_t_b = (3.0 * ran_x / (c_1 * np.pi * (v_sys/c.distance)**2))**(1.0/3.0) + t_min

    theta_c = v_sys / c.distance * (ran_t_b - t_min)
    c_2 = 1.0 / (np.pi * theta_c**2)
    ran_y = uniform.rvs(size = N_size)
    ran_theta = np.sqrt(ran_y / (c_2 * np.pi))

    ran_ra = c.rad_to_deg * ran_theta * np.cos(ran_phi) / np.cos(c.deg_to_rad * dec) + ra
    ran_dec = c.rad_to_deg * ran_theta * np.sin(ran_phi) + dec

    # Specific star formation rate (Msun/Myr/steradian)
    SFR = get_SFH(ran_ra, ran_dec, ran_t_b/(c.yr_to_sec*1.0e6))

    return np.mean(SFR)
