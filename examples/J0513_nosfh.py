import sys
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

sys.path.append("../pyBSE/")
import pybse
import dart_board
from dart_board import sf_history


# Values for Swift J0513.4-6547 from Coe et al. 2015, MNRAS, 447, 1630
system_kwargs = {"P_orb" : 27.405, "P_orb_err" : 0.5, "ecc_max" : 0.17, "m_f" : 9.9,
                 "m_f_err" : 2.0}
pub = dart_board.DartBoard("NSHMXB", evolve_binary=pybse.evolve, metallicity=0.008,
                           nwalkers=320, threads=20, system_kwargs=system_kwargs)

dart_initial = np.array([2.6760214, 1.76740219, 3.98651669, 0.14954731, 16.45027917, 1.35932997, 3.02603997, 3.01717744]) 
pub.aim_darts(dart=dart_initial)


start_time = time.time()
pub.throw_darts(nburn=2, nsteps=420000)
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
pickle.dump(pub.chains, open("../data/J0513_nosfh_chain.obj", "wb"))
pickle.dump(pub.lnprobability, open("../data/J0513_nosfh_lnprobability.obj", "wb"))
pickle.dump(pub.derived, open("../data/J0513_nosfh_derived.obj", "wb"))
