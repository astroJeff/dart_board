import os
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d



def load_smc_coor():
    """ Load coordinates to SMC regions

    Returns
    -------
    smc_coor: np structured array
        Coordinates of SMC regions in degrees
        dtype: [('region','<S10'),
                ('ra','float64'),
                ('dec','float64')]
    """


    # Load data
    this_dir, this_filename = os.path.split(__file__)
    data_file = os.path.join(this_dir, "smc_coordinates.dat")

    smc_coor_2 = np.genfromtxt(data_file, dtype="S10,S2,S2,S3,S2")

    smc_coor = np.recarray(0, dtype=[('region','<S10'),('ra','float64'),('dec','float64')])
    tmp = np.recarray(1, dtype=[('region','<S10'),('ra','float64'),('dec','float64')])


    for coor in smc_coor_2:
        ra = str(coor[1].decode("utf-8"))+"h"+str(coor[2].decode("utf-8"))+"m"
        dec = str(coor[3].decode("utf-8"))+"d"+str(coor[4].decode("utf-8"))+"m"

        region = coor[0]

        coor = SkyCoord(ra, dec)

        tmp["region"] = region
        tmp["ra"] = coor.ra.degree
        tmp["dec"] = coor.dec.degree

        smc_coor = np.append(smc_coor, tmp)

    return smc_coor


def load_smc_data():
    """ Return SMC star formation history per unit steradian

    Returns
    -------
    smc_sfh : np structured array
        SMC star formation history
        dtype: [('region','<S10'),
                ('log_age','<f8'),
                ('sfh_z008','<f8'),
                ('sfh_z004','<f8'),
                ('sfh_z001','<f8')]
    """

    # Create an empty array to start with
    dtypes = [('region','<S10'), \
            ('log_age','<f8'), \
            ('sfh_z008','<f8'), \
            ('sfh_z004','<f8'), \
            ('sfh_z001','<f8')]

    smc_data = np.recarray(0, dtype=dtypes)
    out_data = np.recarray(1, dtype=dtypes)

    smc_coor = load_smc_coor()

    # Each region has an area of 12' x 12', or 1.218e-5 steradians
    area = 1.218e-5

    # Star formation history file
    # Test to load data
    this_dir, this_filename = os.path.split(__file__)
    file_path = os.path.join(this_dir, "smc_sfh.dat")

    with open(file_path) as f:
        line_num = 0

        for line in f:

            line_num += 1

            if line_num < 27: continue

            line_data = np.array(line.split()).astype(str)

            out_data[0][0] = line_data[0]
            out_data[0][1] = (line_data[5].astype(np.float64)+line_data[6].astype(np.float64))/2.0
            out_data[0][2] = line_data[7].astype(np.float64) / area
            out_data[0][3] = line_data[10].astype(np.float64) / area

            if len(line_data) < 15:
                out_data[0][4] = 0.0
            else:
                out_data[0][4] = line_data[13].astype(np.float64) / area

            smc_data = np.append(smc_data, out_data[0])

    return smc_data


def load_smc_sfh(z=0.008):
    """ Create array of 1D interpolations in time of the
    star formation histories for each region in the SMC.

    Parameters
    ----------
    z : float (0.001, 0.004, 0.008)
        Metallicity for which to return star formation history
        Default = 0.008

    Returns
    -------
    SF_history : ndarray
        Array of star formation histories for each region
    """


    # Load the LMC coordinates and SFH data
    smc_data = load_smc_data()

    smc_sfh = np.array([])
    age = np.array([])
    sfr = np.array([])

    _, idx = np.unique(smc_data["region"], return_index=True)
    regions = smc_data["region"][np.sort(idx)]

    for i in np.arange(len(regions)):
#    for r in regions:
        r = regions[i]

        age = smc_data["log_age"][np.where(smc_data["region"] == r)]
        if z == 0.008:
            sfr = smc_data["sfh_z008"][np.where(smc_data["region"] == r)]
        elif z == 0.004:
            sfr = smc_data["sfh_z004"][np.where(smc_data["region"] == r)]
        elif z == 0.001:
            sfr = smc_data["sfh_z001"][np.where(smc_data["region"] == r)]
        else:
            print("ERROR: You must choose an appropriate metallicity input")
            print("Possible options are 0.001, 0.004, 0.008")
            return -1

        smc_sfh = np.append(smc_sfh, interp1d(age[::-1], sfr[::-1], bounds_error=False, fill_value=0.0))

    return smc_sfh
