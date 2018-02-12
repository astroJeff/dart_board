import sys
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

sys.path.append("../pyBSE/")
import pybse
import dart_board

pub = dart_board.DartBoard("HMXB", evolve_binary=pybse.evolve, nwalkers=100)
pub.aim_darts()


start_time = time.time()
pub.throw_darts(nburn=2, nsteps=10)
print("Simulation took",time.time()-start_time,"seconds.")



