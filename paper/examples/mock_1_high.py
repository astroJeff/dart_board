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
# Output values: 1.48 8.09 179.89 0.68 39.65 1.32e-12 22.12 13 1

LMC_metallicity = 0.008

system_kwargs = {"M2" : 7.7, "M2_err" : 0.5, "ecc" : 0.69, "ecc_err" : 0.05}
pub = dart_board.DartBoard("NSHMXB", evolve_binary=pybse.evolve,
                           nwalkers=320, threads=20,
                           metallicity=LMC_metallicity, thin=100,
                           system_kwargs=system_kwargs)

pub.aim_darts(N_iterations=100000, a_set='high')


start_time = time.time()
pub.throw_darts(nburn=2, nsteps=150000)
print("Simulation took",time.time()-start_time,"seconds.")



# Acceptance fraction
print("Acceptance fractions:",pub.sampler.acceptance_fraction)


# Autocorrelation length
try:
    print("Autocorrelation length:", pub.sample.acor)
except:
    print("Acceptance fraction is too low.")


# Save outputs
np.save("../data/mock_1_high_chain.npy", pub.chains)
np.save("../data/mock_1_high_derived.npy", pub.derived)
np.save("../data/mock_1_high_lnprobability.npy", pub.lnprobability)
