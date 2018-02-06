import sys
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

sys.path.append("../pyBSE/")
import pybse
import dart_board

pub = dart_board.DartBoard("HMXB", evolve_binary=pybse.evolve, nwalkers=320, threads=20)
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



# Save outputs
np.save("../data/HMXB_chain.npy", pub.chains)
np.save("../data/HMXB_derived.npy", pub.derived)
np.save("../data/HMXB_lnprobability.npy", pub.lnprobability)


# Pickle results
# import pickle
# pickle.dump(pub.chains, open("../data/HMXB_chain.obj", "wb"))
# pickle.dump(pub.lnprobability, open("../data/HMXB_lnprobability.obj", "wb"))
# pickle.dump(pub.derived, open("../data/HMXB_derived.obj", "wb"))
