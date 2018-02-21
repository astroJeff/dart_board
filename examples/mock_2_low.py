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
# Input values: 14.113 5.094 45.12 0.624 141.12 1.6982 1.6266 83.2554 -69.939 21.89 0.008
# Output values: 1.46 15.09, 369.49, 0.67, 22.61, 4.59e-13, 16.91, 13, 1

LMC_metallicity = 0.008

system_kwargs = {"ra" : 83.059940, "dec" : -69.904890 }
pub = dart_board.DartBoard("HMXB", evolve_binary=pybse.evolve,
                           ln_prior_pos=sf_history.lmc.prior_lmc,
                           nwalkers=320, threads=20,
                           metallicity=LMC_metallicity, thin=100,
                           system_kwargs=system_kwargs)

pub.aim_darts(N_iterations=100000, a_set='low')


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
np.save("../data/mock_2_low_chain.npy", pub.chains)
np.save("../data/mock_2_low_derived.npy", pub.derived)
np.save("../data/mock_2_low_lnprobability.npy", pub.lnprobability)
