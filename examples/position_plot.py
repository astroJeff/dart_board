import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy import optimize as so
from dart_board import constants as c
from dart_board import sf_history
from dart_board.sf_history.sf_plotting import get_plot_polar
from dart_board.posterior import A_to_P, calculate_L_x
from dart_board.forward_pop_synth import get_theta, get_phi, get_new_ra_dec



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





# Load chains
file_name = sys.argv[1]
in_file = "../data/" + file_name + "_chain.obj"

if len(sys.argv) == 2:
    delay = 200
else:
    delay = int(int(sys.argv[2]) / 100)


chains = pickle.load(open(in_file, "rb"))
chains = chains[:,delay:,:]
n_chains, length, n_var = chains.shape
chains = chains.reshape((n_chains*length, n_var))
print(chains.shape)


chains[:,0] = np.exp(chains[:,0])
chains[:,1] = np.exp(chains[:,1])
chains[:,2] = np.exp(chains[:,2])
chains[:,9] = np.exp(chains[:,9])

# Load derived data
file_name = sys.argv[1]
in_file = "../data/" + file_name + "_derived.obj"

derived = pickle.load(open(in_file, "rb"))
derived = derived[:,delay:,:]
n_chains, length, n_var = derived.shape
derived = derived.reshape((n_chains*length, n_var))
print(derived.shape)



# Calculate travel distances
t_flight = chains.T[9] - derived.T[6]
theta = (t_flight*1.0e6*c.yr_to_sec) * derived.T[4]/c.dist_LMC * get_theta(len(derived.T[0]))
phi = get_phi(len(derived.T[0]))

# Calculate new positions
ra_birth = chains.T[7]
dec_birth = chains.T[8]
ra_new, dec_new = get_new_ra_dec(ra_birth, dec_birth, theta, phi)



fig = plt.figure(figsize=(5.5,9))
ax0 = plt.gca()
ax0.get_xaxis().set_ticks([])
ax0.get_yaxis().set_ticks([])
ax0.axis('off')
gs = gridspec.GridSpec(2, 1, height_ratios=[1,1])



# Load up LMC star formation history
sf_history.lmc.load_sf_history()


# Define the levels for star formation rate contours
levels = np.linspace(1.0e7, 1.5e8, 10) / 1.0e6 * (np.pi/180.0)**2

# Plot star formation rates at two different times
sf_plot, ax1 = get_plot_polar(10.0, sfh_function=sf_history.lmc.get_SFH, fig_in=fig, gs=gs[0],
                              # ra_dist=None, dec_dist=None,
                              ra_dist=ra_new, dec_dist=dec_new, contour_CL='quad',
                              dist_bins=40, sfh_bins=30, sfh_levels=levels, ra=None, dec=None,
                              xcenter=0.0, ycenter=21.0, xwidth=5.0, ywidth=5.0, rot_angle=0.2,
                              xlabel="Right Ascension", ylabel="Declination", xgrid_density=6, ygrid_density=5,
                              color_map='Blues', color_bar=False, contour_alpha=1.0, title="Star Formation Rate at 10 Myr")

sf_plot, ax1 = get_plot_polar(30.0, sfh_function=sf_history.lmc.get_SFH, fig_in=fig, gs=gs[1],
                              # ra_dist=None, dec_dist=None,
                              ra_dist=ra_new, dec_dist=dec_new, contour_CL='quad',
                              dist_bins=40, sfh_bins=30, sfh_levels=levels, ra=None, dec=None,
                              xcenter=0.0, ycenter=21.0, xwidth=5.0, ywidth=5.0, rot_angle=0.2,
                              xlabel="Right Ascension", ylabel="Declination", xgrid_density=6, ygrid_density=5,
                              color_map='Blues', color_bar=False, contour_alpha=1.0, title="Star Formation Rate at 30 Myr")



fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
cb = fig.colorbar(sf_plot, cax=cbar_ax, extend='max')
cb.set_label(r'$\frac{M_{\odot}}{{\rm yr\ deg.}^2}$', rotation=0, labelpad=-20, y=1.1, fontsize=14)
# cb.set_label(r'$M_{\odot}$ yr$^{-1}$ deg.$^{-2}$', rotation=270, labelpad=17)

# Convert from Msun/Myr/rad^2 to Msun/yr/deg^2
ticks = np.linspace(1.0e7, 2.0e8, 10) / 1.0e6 * (np.pi/180.0)**2
cb.set_ticks(ticks)
ticks = np.round(ticks, decimals=3)
cb.set_ticklabels(ticks.astype(str))


plt.subplots_adjust(left=0.15, bottom=0.07, right=0.83, top=0.95,
                    wspace=0.25, hspace=0.25)



# plt.tight_layout()
file_out = "../figures/" + file_name + "_positions.pdf"
plt.savefig(file_out)
