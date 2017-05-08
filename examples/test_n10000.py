import sys
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

sys.path.append("../pyBSE/")
import pybse
import dart_board

pub = dart_board.DartBoard("HMXB", evolve_binary=pybse.evolv_wrapper, nwalkers=10000)
pub.aim_darts()


start_time = time.time()
pub.throw_darts(nburn=1000, nsteps=1000)
print("Simulation took",time.time()-start_time,"seconds.")

# Acceptance fraction
print("Acceptance fractions:",pub.sampler.acceptance_fraction)

# Autocorrelation length
try:
    print("Autocorrelation length:", pub.sample.acor)
except:
    print("Acceptance fraction is too low.")


print(pub.chains.shape)
print(pub.derived.shape)
print(pub.lnprobability.shape)


import pickle
pickle.dump(pub.sampler.acceptance_fraction, open("test_n10000_accept_frac.dat", "wb"))
