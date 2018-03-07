import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import corner
import matplotlib.gridspec as gridspec
from scipy import stats
from dart_board import sf_history
from dart_board.sf_history.sf_plotting import get_plot_polar
from dart_board.posterior import A_to_P, calculate_L_x

file_name = sys.argv[1]
in_file = "../data/" + file_name + "_chain.obj"


if len(sys.argv) == 2:
    delay = 200
else:
    delay = int(int(sys.argv[2]) / 100)


chains = pickle.load(open(in_file, "rb"))
if chains.ndim == 3:
    chains = chains[:,delay:,:]
elif chains.ndim == 4:
    chains = chains[0,:,delay:,:]
else:
    sys.exit()

n_chains, length, n_var = chains.shape
chains = chains.reshape((n_chains*length, n_var))
print(chains.shape)


chains[:,0] = np.exp(chains[:,0])
chains[:,1] = np.exp(chains[:,1])
chains[:,2] = np.exp(chains[:,2])
chains[:,-1] = np.exp(chains[:,-1])




one_sigma_low = (1.0-0.68)/2.0
one_sigma_high = 1.0 - one_sigma_low

median_idx = int(0.5*n_chains*length)
one_sigma_low_idx = int(one_sigma_low*n_chains*length)
one_sigma_high_idx = int(one_sigma_high*n_chains*length)

for i in range(n_var):
    tmp_x = np.sort(chains[:,i])

    median = tmp_x[median_idx]
    low = median - tmp_x[one_sigma_low_idx]
    high = tmp_x[one_sigma_high_idx] - median

    print("Median=", "{0:.2f}".format(median), "-", "{0:.2f}".format(low), "+", "{0:.2f}".format(high))
