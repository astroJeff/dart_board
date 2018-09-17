import numpy as np
import sys

Z_ref = 0.0142
photometry = None

def load_photometry(filename):
    global photometry
    photometry = np.load("data/"+filename)


def get_photometry(metallicity, t_eff, luminosity, band):

    global photometry
    if photometry is None:
        print("You must first load photometry from tables.")
        sys.exit()

    log_Z_H = np.log10(metallicity/Z_ref)
    log_L = np.log10(luminosity)
    log_Teff = np.log10(t_eff)

    # Get closest metallicity
    Z_options = np.unique(photometry['FeH_init'])
    Z_closest = Z_options[np.argsort(np.abs(Z_options-log_Z_H))[0:2]]

    idx = np.union1d(np.where(photometry['FeH_init'] == Z_closest[0]),
                     np.where(photometry['FeH_init'] == Z_closest[1]))

    photo = photometry[idx]
