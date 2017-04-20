import sys
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

sys.path.append("../pyBSE/")
import pybse
import dart_board
from dart_board import sf_history


# Values for Swift J0513.4-6547 from Coe et al. 2015, MNRAS, 447, 1630
kwargs = {"P_orb" : 27.405, "P_orb_err" : 0.5, "ecc_max" : 0.17, "m_f" : 9.9,
          "m_f_err" : 2.0, "ra" : 78.36775, "dec" : -65.7885278}
pub = dart_board.DartBoard("NSHMXB", evolve_binary=pybse.evolv_wrapper,
                           ln_prior_pos=sf_history.flat.prior_lmc, nwalkers=320,
                           kwargs=kwargs)

pub.aim_darts()


start_time = time.time()
pub.throw_darts(nburn=50000, nsteps=50000)
print("Simulation took",time.time()-start_time,"seconds.")


# Acceptance fraction
print("Acceptance fractions:",pub.sampler.acceptance_fraction)

# Autocorrelation length
try:
    print("Autocorrelation length:", pub.sample.acor)
except:
    print("Acceptance fraction is too low.")


# Pickle results
import pickle
pickle.dump(pub.sampler.chain, open("../data/J0513_flatsfh_chain.obj", "wb"))
pickle.dump(pub.sampler.lnprobability, open("../data/J0513_flatsfh_lnprobability.obj", "wb"))
pickle.dump(pub.binary_data, open("../data/J0513_flatsfh_binary_data.obj", "wb"))
