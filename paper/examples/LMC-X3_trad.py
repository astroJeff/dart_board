import sys
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

sys.path.append("../pyBSE/")
import pybse
import dart_board
from dart_board import sf_history

kwargs = {"M1" : 6.98, "M1_err" : 0.56, "M2" : 3.63, "M2_err" : 0.57, "P_orb" : 1.7, "P_orb_err" : 0.1}
pub = dart_board.DartBoard("BHHMXB", evolve_binary=pybse.evolv_wrapper, nwalkers=320, kwargs=kwargs)
# pub.aim_darts()


start_time = time.time()
pub.scatter_darts(num_darts=100000000)
# pub.throw_darts(nburn=50000, nsteps=50000)
print("Simulation took",time.time()-start_time,"seconds.")


# Pickle results
import pickle
pickle.dump(pub.chains, open("../data/LMC-X3_trad_x_i.obj", "wb"))
pickle.dump(pub.likelihood, open("../data/LMC-X3_trad_likelihood.obj", "wb"))
pickle.dump(pub.derived, open("../data/LMC-X3_trad_derived.obj", "wb"))
