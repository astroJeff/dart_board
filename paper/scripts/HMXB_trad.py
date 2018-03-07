import sys
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

sys.path.append("../pyBSE/")
import pybse
import dart_board

pub = dart_board.DartBoard("HMXB", evolve_binary=pybse.evolve, nwalkers=320)
# pub.aim_darts()


start_time = time.time()
pub.scatter_darts(seconds=356700)
print("Simulation took",time.time()-start_time,"seconds.")



# Save outputs
np.save("../data/HMXB_trad_chain.npy", pub.chains)
np.save("../data/HMXB_trad_derived.npy", pub.derived)
np.save("../data/HMXB_trad_lnprobability.npy", pub.lnprobability)


