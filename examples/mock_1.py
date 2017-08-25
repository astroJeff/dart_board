import sys
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

sys.path.append("../pyBSE/")
import pybse
import dart_board
from dart_board import sf_history


# Values for mock system 1
# Input values: 11.77 8.07 4850.81 0.83 153.04 2.05 2.33 34.74
# Output values: 1.32 8.05 43.22 0.40 24.66 1.72e-11 21.66 13 1

system_kwargs = {"M2" : 8.3, "M2_err" : 0.5, "ecc" : 0.42, "ecc_err" : 0.05}
pub = dart_board.DartBoard("NSHMXB", evolve_binary=pybse.evolve,
                           nwalkers=320, threads=20,
                           system_kwargs=system_kwargs)

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
pickle.dump(pub.chains, open("../data/mock_1_chain.obj", "wb"))
pickle.dump(pub.lnprobability, open("../data/mock_1_lnprobability.obj", "wb"))
pickle.dump(pub.derived, open("../data/mock_1_derived.obj", "wb"))
