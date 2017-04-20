import sys
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

sys.path.append("../pyBSE/")
import pybse
import dart_board


pub = dart_board.DartBoard("HMXB",
                           evolve_binary=pybse.evolv_wrapper,
                           ln_prior_pos=sf_history.lmc.prior_lmc,
                           nwalkers=320,
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
pickle.dump(pub.sampler.chain, open("../data/HMXB_LMC_chain.obj", "wb"))
pickle.dump(pub.sampler.lnprobability, open("../data/HMXB_LMC_lnprobability.obj", "wb"))
pickle.dump(pub.binary_data, open("../data/HMXB_LMC_binary_data.obj", "wb"))
