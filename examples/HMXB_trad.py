import sys
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

sys.path.append("../pyBSE/")
import pybse
import dart_board

pub = dart_board.DartBoard("HMXB", evolve_binary=pybse.evolv_wrapper, nwalkers=320)
# pub.aim_darts()


start_time = time.time()
pub.scatter_darts(num_darts=100000000)
# pub.throw_darts(nburn=50000, nsteps=50000)
print("Simulation took",time.time()-start_time,"seconds.")



# Save outputs
np.save("../data/HMXB_trad_chain.npy", pub.chains)
np.save("../data/HMXB_trad_derived.npy", pub.derived)
np.save("../data/HMXB_trad_lnprobability.npy", pub.lnprobability)


# Pickle results
# import pickle
# pickle.dump(pub.chains, open("../data/HMXB_trad_x_i.obj", "wb"))
# pickle.dump(pub.likelihood, open("../data/HMXB_trad_likelihood.obj", "wb"))
# pickle.dump(pub.derived, open("../data/HMXB_trad_derived.obj", "wb"))
