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
# Input values: 12.856 7.787 5.19 0.5434 484.64 2.588 0.978 83.2554 -69.9390 25.71 0.008
# Output values: 1.548 8.10 96.42 0.674 46.26 4.617e-12 14.89 13 1

system_kwargs = {"ra" : 83.4989 , "dec" : -70.0366 }
pub = dart_board.DartBoard("HMXB", evolve_binary=pybse.evolve,
                           ln_prior_pos=sf_history.lmc.prior_lmc,
                           nwalkers=320, threads=20,
                           metallicity=0.008,
                           system_kwargs=system_kwargs)

pub.aim_darts()


start_time = time.time()
pub.throw_darts(nburn=2, nsteps=220000)
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
