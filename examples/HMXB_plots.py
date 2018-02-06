import numpy as np


import matplotlib
matplotlib.use('Agg')
import corner
import matplotlib.pyplot as plt
from matplotlib import font_manager


# Load chains
chains = np.load("../data/HMXB_chain.npy")
n_chains, length, n_var = chains.shape
chains = chains.reshape((n_chains*length, n_var))
print(chains.shape)


# Move from ln parameters to parameters in chains
chains[:,0] = np.exp(chains[:,0])
chains[:,1] = np.exp(chains[:,1])
chains[:,2] = np.exp(chains[:,2])
chains[:,9] = np.exp(chains[:,9])


# Create a corner plot to show the posterior distribution

fontProperties = {'family':'serif', 'serif':['Times New Roman'], 'weight':'normal', 'size':12}
ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', \
                                         weight='normal', stretch='normal', size=10)
plt.rc('font', **fontProperties)

# Corner plot

labels = [r"$M_{\rm 1, i}\ (M_{\odot})$", r"$M_{\rm 2, i}\ (M_{\odot})$", r"$a_{\rm i}\ (R_{\odot})$", \
          r"$e_{\rm i}$", r"$v_{\rm k, i}\ ({\rm km}\ {\rm s}^{-1})$", r"$\theta_{\rm k}\ ({\rm rad.})$", \
          r"$\delta_{\rm i}\ ({\rm deg.}) $", r"$t_{\rm i}\ ({\rm Myr})$"]
plt_range = ([4,35], [2.5,15], [0,3500], [0,1], [0,650], [np.pi/4.,np.pi], [0,np.pi], [0,1000])
hist2d_kwargs = {"plot_datapoints" : False}
corner.corner(chains, labels=labels, range=plt_range, bins=20, max_n_ticks=4, **hist2d_kwargs)

plt.savefig("../figures/HMXB_corner.pdf")
