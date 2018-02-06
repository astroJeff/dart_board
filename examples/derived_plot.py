import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import sys
import corner
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy import optimize as so
from dart_board import constants as c
from dart_board import sf_history
from dart_board.sf_history.sf_plotting import get_plot_polar
from dart_board.posterior import A_to_P, calculate_L_x
from dart_board.forward_pop_synth import get_theta


# Need this function
def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level

def hist2d(x, y, xlim=None, ylim=None, color='k', bins=100, ax=None, alpha=0.2):

    if xlim is None:
        xlim = [0.95*np.min(x), 1.05*np.max(x)]
    if ylim is None:
        ylim = [0.95*np.min(y), 1.05*np.max(y)]

    data_range = (xlim,ylim)
    nbins_x = bins
    nbins_y = bins
    H, xedges, yedges = np.histogram2d(x, y, bins=(nbins_x,nbins_y),
                                       range=data_range, normed=True)
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,nbins_x))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y,1))
    pdf = (H*(x_bin_sizes*y_bin_sizes))

    one_quad = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.25))
    two_quad = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.50))
    three_quad = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.75))
    X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
    Z = pdf.T


    # Plot contours
    if ax is None:
        levels = [1.0, one_quad, two_quad, three_quad]
        contour = plt.contourf(X, Y, Z, levels=levels[::-1], origin="lower", colors=color,
                               rasterized=True, alpha=alpha)
        levels = [1.0, one_quad, two_quad]
        contour = plt.contourf(X, Y, Z, levels=levels[::-1], origin="lower", colors=color,
                               rasterized=True, alpha=alpha)
        levels = [1.0, one_quad]
        contour = plt.contourf(X, Y, Z, levels=levels[::-1], origin="lower", colors=color,
                               rasterized=True, alpha=alpha)
    else:
        levels = [1.0, one_quad, two_quad, three_quad]
        contour = ax.contourf(X, Y, Z, levels=levels[::-1], origin="lower", colors=color,
                              rasterized=True, alpha=alpha)
        levels = [1.0, one_quad, two_quad]
        contour = ax.contourf(X, Y, Z, levels=levels[::-1], origin="lower", colors=color,
                              rasterized=True, alpha=alpha)
        levels = [1.0, one_quad]
        contour = ax.contourf(X, Y, Z, levels=levels[::-1], origin="lower", colors=color,
                              rasterized=True, alpha=alpha)




delay = 0

# Load chains
file_name = sys.argv[1]
in_file = "../data/" + file_name + "_chain.npy"

chains = np.load(in_file)
if chains.ndim == 4: chains = chains[0]
chains = chains[:,delay:,:]
n_chains, length, n_var = chains.shape
chains = chains.reshape((n_chains*length, n_var))
print(chains.shape)


chains[:,0] = np.exp(chains[:,0])
chains[:,1] = np.exp(chains[:,1])
chains[:,2] = np.exp(chains[:,2])
chains[:,7] = np.exp(chains[:,7])

# Load derived data
file_name = sys.argv[1]
in_file = "../data/" + file_name + "_derived.npy"
derived = np.load(in_file)
if derived.ndim == 4: derived = derived[0]
derived = derived[:,delay:,:]
n_chains, length, n_var = derived.shape
derived = derived.reshape((n_chains*length, n_var))
print(derived.shape)





# Create plot
fig, ax = plt.subplots(5, 1, figsize=(4,12))


gs = gridspec.GridSpec(5, 1, height_ratios=[1, 1, 1 ,1, 1])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])
ax4 = plt.subplot(gs[4])

gs2 = gridspec.GridSpec(5, 2,
                        height_ratios=[1, 1, 1 ,1, 1],
                        width_ratios=[99, 0.1])
#ax3 = plt.subplot(gs2[6])


# Panel 1: P_orb vs ecc
P_orb = A_to_P(derived.T[0], derived.T[1], derived.T[2])
xlim = [0.0, 100.0]
ylim = [-0.05, 1.05]
hist2d(P_orb, derived.T[3], ax=ax0, xlim=xlim, ylim=ylim)
ax0.set_xlabel(r"P$_{\rm orb}$ (days)")
ax0.set_ylabel(r"$e$")

# Panel 2: M_2 vs v_sys
xlim = [0.0, 30.0]
ylim = [0.0, 90.0]
hist2d(derived.T[1], derived.T[4], ax=ax1, xlim=xlim, ylim=ylim)
ax1.set_xlabel(r"M$_2$ ($M_{\odot}$)")
ax1.set_ylabel(r"v$_{\rm sys}$ (km/s)")

# Panel 3: t_travel vs. theta
t_flight = chains.T[7] - derived.T[6]
theta = (t_flight*1.0e6*c.yr_to_sec) * derived.T[4]/c.dist_LMC * np.sin(get_theta(len(derived.T[0]))) * 180.0*60.0/np.pi
#hist2d(t_flight, theta, ax=ax2, xlim=[0, 6], ylim=[0,10])
#ax2.set_yticks([0, 5, 10])
#ax2.set_xlabel(r"$t_{\rm travel}$ (Myr)")
#ax2.set_ylabel(r"$\theta$ (amin)")

#ax_twin = ax2.twinx()
#ylimits = ax2.get_ylim()
factor = 875./60.
#ax_twin.set_ylim(ylimits[0]*factor, ylimits[1]*factor)
#ax_twin.set_ylabel('Separation (pc)')
xlim = (0,50)
ax2.hist(t_flight, range=xlim, histtype='step', bins=30, normed=True, color='k', log=True)
ax2.set_xlabel('Travel Time (Myr)')
ax2.set_yticklabels([])
ax2.set_ylabel('log N')

xlim = (0,60)
ax3.hist(theta, range=xlim, histtype='step', bins=30, normed=True, color='k', log=True)
ax3.set_xlabel(r'$\theta$ (amin)')
ax3.set_yticklabels([])
ax3.set_ylabel('log N')
ax_twin = ax3.twiny()
xlimits = ax3.get_xlim()
factor = 875./60.
ax_twin.set_xlim(xlimits[0]*factor, xlimits[1]*factor)
ax_twin.set_xlabel('Separation (pc)')


# Panel 4: L_x histogram
L_x = np.zeros(len(derived.T[0]))
for k in range(len(L_x)):
    L_x[k] = calculate_L_x(derived.T[0][k], derived.T[5][k], derived.T[7][k])
ax4.hist(np.log10(L_x), histtype='step', color='k', bins=30, normed=True, range=(30, 40), log=True)
ax4.set_xlabel(r"log L$_{\rm x}$ (erg/s)")
ax4.set_ylabel("log N")
ax4.set_yticklabels([])

plt.tight_layout()
plt.savefig("../figures/HMXB_derived.pdf")
