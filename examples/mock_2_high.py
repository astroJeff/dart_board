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
# Input values: 12.856 7.787 5.19 0.5434 384.64 2.588 0.978 83.2554 -69.939 25.71 0.008
# Output values: 1.38 17.17, 59.81, 0.54, 44.14, 7.90e-11, 19.71, 13, 1

LMC_metallicity = 0.008

system_kwargs = {"ra" : 82.84012909 , "dec" : -70.12312498 }
pub = dart_board.DartBoard("HMXB", evolve_binary=pybse.evolve,
                           ln_prior_pos=sf_history.lmc.prior_lmc,
                           nwalkers=320, threads=20,
                           metallicity=LMC_metallicity, thin=100,
                           system_kwargs=system_kwargs)

pub.aim_darts(N_iterations=100000, a_set='high')


start_time = time.time()
pub.throw_darts(nburn=2, nsteps=52000)
print("Simulation took",time.time()-start_time,"seconds.")




# Acceptance fraction
print("Acceptance fractions:",pub.sampler.acceptance_fraction)

# Autocorrelation length
try:
    print("Autocorrelation length:", pub.sample.acor)
except:
    print("Acceptance fraction is too low.")



# Save outputs
np.save("../data/mock_2_high_chain.npy", pub.chains)
np.save("../data/mock_2_high_derived.npy", pub.derived)
np.save("../data/mock_2_high_lnprobability.npy", pub.lnprobability)
