import sys
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

sys.path.append("../pyBSE/")
import pybse
import dart_board
from dart_board import sf_history


# Values for mock system 3
# Input values: 11.01 7.42 744.19 0.20 167.69 1.79 2.08 83.2559 -69.9377 36.59
# Output values:  1.30 7.43 112.51 0.62 31.23 3.98e-13 25.68 13 1

system_kwargs = {"M2" : 7.50, "M2_err" : 0.25,
                 "P_orb" : 15.0, "P_orb_err" : 1.0,
                 "ecc" : 0.60, "ecc_err" : 0.05,
                 "L_x" : 1.90e33, "L_x_err" : 1.0e32,
                 "ra" : 81.5858 , "dec" : -70.8483}
pub = dart_board.DartBoard("HMXB", evolve_binary=pybse.evolve,
                           ln_prior_pos=sf_history.lmc.prior_lmc, nwalkers=320,
                           threads=20, system_kwargs=system_kwargs)

pub.aim_darts()


start_time = time.time()
pub.throw_darts(nburn=20000, nsteps=100000)
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
pickle.dump(pub.chains, open("../data/mock_3_chain.obj", "wb"))
pickle.dump(pub.lnprobability, open("../data/mock_3_lnprobability.obj", "wb"))
pickle.dump(pub.derived, open("../data/mock_3_derived.obj", "wb"))
