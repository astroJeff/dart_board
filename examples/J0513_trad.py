import sys
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

sys.path.append("../pyBSE/")
import pybse
import dart_board
from dart_board import sf_history

system_kwargs = {"P_orb" : 27.405, "P_orb_err" : 0.5, "ecc_max" : 0.17, "m_f" : 9.9,
                 "m_f_err" : 2.0, "ra" : 78.36775, "dec" : -65.7885278}
pub = dart_board.DartBoard("NSHMXB", evolve_binary=pybse.evolve,
                           ln_prior_pos=sf_history.lmc.prior_lmc,
                           generate_pos=sf_history.lmc.get_random_positions,
                           nwalkers=320, system_kwargs=system_kwargs)
# pub.aim_darts()


start_time = time.time()
pub.scatter_darts(seconds=467648)
# pub.throw_darts(nburn=50000, nsteps=50000)
print("Simulation took",time.time()-start_time,"seconds.")


# Pickle results
import pickle
pickle.dump(pub.chains, open("../data/J0513_trad_x_i.obj", "wb"))
pickle.dump(pub.likelihood, open("../data/J0513_trad_likelihood.obj", "wb"))
pickle.dump(pub.derived, open("../data/J0513_trad_derived.obj", "wb"))
