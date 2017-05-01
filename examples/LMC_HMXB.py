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
pickle.dump(pub.chains, open("../data/LMC_HMXB_chain.obj", "wb"))
pickle.dump(pub.lnprobability, open("../data/LMC_HMXB_lnprobability.obj", "wb"))
pickle.dump(pub.derived, open("../data/LMC_HMXB_derived.obj", "wb"))
