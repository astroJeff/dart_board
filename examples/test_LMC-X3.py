import sys
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

sys.path.append("../pyBSE/")
import pybse
import dart_board

kwargs = {"M1" : 6.98, "M1_err" : 0.56, "M2" : 3.63, "M2_err" : 0.57, "P_orb" : 1.7, "P_orb_err" : 0.1}
pub = dart_board.DartBoard("BHHMXB", evolve_binary=pybse.evolv_wrapper, nwalkers=320, kwargs=kwargs)
pub.aim_darts_separate()


start_time = time.time()
pub.throw_darts(nburn=100000, nsteps=200000)
print("Simulation took",time.time()-start_time,"seconds.")

# Acceptance fraction
print("Acceptance fractions:",pub.sampler.acceptance_fraction)

# Autocorrelation length
try:
    print("Autocrrelation length:", pub.sample.acor)
except:
    print("Acceptance fraction is too low.")


# Pickle results
import pickle
pickle.dump(pub.sampler.chain, open("../data/LMC-X3_chain.obj", "wb"))
pickle.dump(pub.binary_data, open("../data/LMC-X3_binary_data.obj", "wb"))






# Create a corner plot to show the posterior distribution
import corner
import matplotlib.pyplot as plt
from matplotlib import font_manager

fontProperties = {'family':'serif', 'serif':['Times New Roman'], 'weight':'normal', 'size':12}
ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', \
                                         weight='normal', stretch='normal', size=10)
plt.rc('font', **fontProperties)

# Corner plot

labels = [r"$M_{\rm 1, i}\ (M_{\odot})$", r"$M_{\rm 2, i}\ (M_{\odot})$", r"$a_{\rm i}\ (R_{\odot})$", \
          r"$e_{\rm i}$", r"$v_{\rm k, i}\ ({\rm km}\ {\rm s}^{-1})$", r"$\theta_{\rm k}\ ({\rm rad.})$", \
          r"$\delta_{\rm i}\ ({\rm deg.}) $", r"$t_{\rm i}\ ({\rm Myr})$"]
plt_range = ([7,24], [2.5,15], [0,1500], [0,1], [0,450], [np.pi/4.,np.pi], [0,np.pi], [0,70])
hist2d_kwargs = {"plot_datapoints" : False}
corner.corner(pub.sampler.flatchain, labels=labels, range=plt_range, bins=20, max_n_ticks=4, **hist2d_kwargs)

plt.savefig("../figures/test_LMC-X3_corner.pdf")
