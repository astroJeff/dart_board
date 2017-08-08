import sys
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

sys.path.append("../pyBSE/")
import pybse
import dart_board
from dart_board import sf_history


# Values for mock system 2
# Input values: 10.98 7.42 744.24 0.21 168.87 1.81 2.09 83.2554 -69.9390 36.99
# Output values:  1.30 7.52 31.38 0.0 21.37 8.16e-12 24.44 13 1

system_kwargs = {"ra" : 83.4989 , "dec" : -70.2366 }
pub = dart_board.DartBoard("HMXB", evolve_binary=pybse.evolve,
                           ln_prior_pos=sf_history.lmc.prior_lmc, nwalkers=320,
                           system_kwargs=system_kwargs)

pub.aim_darts()


start_time = time.time()
pub.throw_darts(nburn=20000, nsteps=10000)
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
pickle.dump(pub.chains, open("../data/mock_2_chain.obj", "wb"))
pickle.dump(pub.lnprobability, open("../data/mock_2_lnprobability.obj", "wb"))
pickle.dump(pub.derived, open("../data/mock_2_derived.obj", "wb"))
