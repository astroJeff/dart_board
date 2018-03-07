import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter
import numpy as np
import sys
import corner
import matplotlib.gridspec as gridspec
from scipy import stats
import scipy.optimize as so
from dart_board import sf_history
from dart_board.sf_history.sf_plotting import get_plot_polar
from dart_board.posterior import A_to_P, calculate_L_x

file_name = sys.argv[1]
in_file_low = "../data/" + file_name + "_low_chain.npy"
in_file_high = "../data/" + file_name + "_high_chain.npy"


if len(sys.argv) == 2:
    delay = 200
else:
    delay = int(int(sys.argv[2]) / 100)



def load_flatchain(in_file, delay=200):
    """ Function to load the chains """

    chains = np.load(in_file)
    if chains.ndim == 4: chains = chains[0]  # If using the PT sampler
    chains = chains[:,delay:,:]

    n_chains, length, n_var = chains.shape
    flatchain = chains.reshape((n_chains*length, n_var))
    print("Chains shape:", flatchain.shape)


    flatchain[:,0] = np.exp(flatchain[:,0])  # M1
    flatchain[:,1] = np.exp(flatchain[:,1])  # M2
    flatchain[:,2] = np.log10(np.exp(flatchain[:,2]))  # A
    flatchain[:,-1] = np.exp(flatchain[:,-1])  # t

    return flatchain



def find_confidence_interval(x, pdf, confidence_level):
    """ Function to find the confidence interval of a pdf. """

    return pdf[pdf > x].sum() - confidence_level


def hist2d(x, y, xlim=None, ylim=None, color='k', bins=100, ax=None, alpha=0.2):
    """ Function to plot 2D histograms. """

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




def plot_corner(flatchain_low, flatchain_high, fig=None, limits=None, max_n_ticks=4,
                truths=None, labels=None, bins=25):

    # Get number of dimensions
    nsamples, ndim = flatchain_low.shape

    # Check dimensionality
    nsamples_high, ndim_high = flatchain_high.shape
    if ndim_high != ndim:
        print("You must provide samples with the same dimensionality.")
        return



    factor = 2.0           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05         # w/hspace size
    plotdim = factor * ndim + factor * (ndim - 1.) * whspace
    dim = lbdim + plotdim + trdim

    # Create the figure if one does not already exist
    if fig is None:
        fig, ax = plt.subplots(ndim, ndim, figsize=(dim, dim))
    else:
        ax = np.array(fig.axes).reshape((ndim, ndim))

    # Format the figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=0.08, bottom=0.08, right=tr, top=tr, wspace=0.0, hspace=0.0)


    # Plot the individual distributions
    for idx1 in range(ndim):
        for idx2 in range(ndim):


            # Get plotting range for each variable
            if limits is None:
                xlim = None
                ylim = None
            else:
                xlim = limits[idx1]
                ylim = limits[idx2]


            # Plot each histogram
            if idx1 < idx2:
                ax[idx1, idx2].set_frame_on(False)
                ax[idx1, idx2].set_xticks([])
                ax[idx1, idx2].set_yticks([])
                continue

            elif idx1 == idx2:
                ax[idx1, idx2].hist(flatchain_low[:,idx1], range=xlim, color='C1', histtype='step', bins=bins, normed=True)
                ax[idx1, idx2].hist(flatchain_high[:,idx1], range=ylim, color='C2', histtype='step', bins=bins, normed=True)


                # ax[idx1, idx2].tick_params(axis='x', direction='in', bottom='on', top='off', left='off', right='off')
                ax[idx1, idx2].set_xticks([])
                ax[idx1, idx2].set_yticks([])

                if truths is not None: ax[idx1, idx2].axvline(truths[idx1], color='C0')

            else:
                hist2d(flatchain_low[:,idx2], flatchain_low[:,idx1], xlim=ylim, ylim=xlim, color='C1', bins=bins, ax=ax[idx1,idx2], alpha=0.3)
                hist2d(flatchain_high[:,idx2], flatchain_high[:,idx1], xlim=ylim, ylim=xlim, color='C2', bins=bins, ax=ax[idx1,idx2], alpha=0.3)

                if truths is not None:
                    ax[idx1, idx2].axhline(truths[idx1], color='C0')
                    ax[idx1, idx2].axvline(truths[idx2], color='C0')

                # Ticks
                ax[idx1, idx2].tick_params(axis='x', direction='in', bottom='on', top='on', left='on', right='on')
                ax[idx1, idx2].tick_params(axis='y', direction='in', bottom='on', top='on', left='on', right='on')


            if idx1 == ndim-1 and idx2 == ndim-1:
                ax[idx1, idx2].xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))



            # Adjust ticks, labeling
            if idx1 != ndim-1:
                ax[idx1, idx2].set_xticklabels([])
            else:
                if labels is not None: ax[idx1, idx2].set_label(labels[idx1])
                [l.set_rotation(45) for l in ax[idx1, idx2].get_xticklabels()]
            if idx2 != 0:
                ax[idx1, idx2].set_yticklabels([])
            else:
                if labels is not None: ax[idx1, idx2].set_label(labels[idx2])




    # # Legend
    # blue_patch = mpatches.Patch(color='C0', label='Channel I', alpha=0.5)
    # orange_patch = mpatches.Patch(color='C1', label='Channel II', alpha=0.5)
    # plt.legend(loc=1, handles=[blue_patch, orange_patch])









# Load the chains
flatchain_low = load_flatchain(in_file_low, delay=delay)
flatchain_high = load_flatchain(in_file_high, delay=delay)









truths = None
if 'mock_1' in file_name:
    truths = [11.77, 8.07, np.log10(4850.81), 0.83, 153.04, 2.05, 2.33, 34.74]
    n_var = 8
    plt_range = ([4,18], [5,11], [1,4], [0,1], [0,650], [0,np.pi], [0,np.pi], [0,70])
elif 'mock_2' in file_name:
    truths = [12.856, 7.787, np.log10(5.19), 0.5434, 384.64, 2.588, 0.978, 83.2554, -69.9390, 25.71]
    n_var = 10
    plt_range = ([4,24], [2.5,15], [1,4], [0,1], [0,650], [np.pi/4.,np.pi], [0,np.pi], [81,85], [-71,-69], [0,75])
elif 'mock_3' in file_name:
    truths = [11.01, 7.42, np.log10(744.19), 0.50, 167.69, 1.79, 2.08, 83.5744461, -69.4876344, 36.59]
    n_var = 10
    plt_range = ([8,14], [6,9], [1,4.5], [0,1], [0,600], [0.0,np.pi], [0,np.pi], [81,86], [-70.4,-68.5], [0,60])
elif file_name == 'HMXB':
    n_var = 8
    plt_range = ([4,40], [2.5,20], [1,4.9], [0,1.05], [0,650], [0.0,np.pi], [0,np.pi], [0,55])
elif file_name == 'LMC_HMXB':
    n_var = 10
    plt_range = ([4,40], [2.5,20], [1,4], [0,1], [0,650], [0.0,np.pi], [0,np.pi], [69,89], [-73,-65], [0,55])
elif 'J0513_nosfh' in file_name:
    n_var = 8
    plt_range = ([4,30], [2.5,20], [1,4.5], [0,1], [0,500], [0.0,np.pi], [0,np.pi], [0,55])
elif 'J0513_flatsfh' in file_name:
    n_var = 10
    plt_range = ([4,30], [2.5,20], [1,4.5], [0,1], [0,500], [0.0,np.pi], [0,np.pi], [77.7,79], [-66.2,-65.4], [0,55])
elif 'J0513' in file_name:
    n_var = 10
    plt_range = ([4,30], [2.5,20], [1,4.5], [0,1], [0,500], [0.0,np.pi], [0,np.pi], [77,80], [-66.5,-65.0], [0,55])




fig, ax = plt.subplots(n_var,n_var, figsize=(10,10))



if n_var == 8:
    labels = [r"$M_{\rm 1, i}\ (M_{\odot})$", r"$M_{\rm 2, i}\ (M_{\odot})$", r"log $a_{\rm i}\ (R_{\odot})$", \
              r"$e_{\rm i}$", r"$v_{\rm k}\ ({\rm km}\ {\rm s}^{-1})$", r"$\theta_{\rm k}\ ({\rm rad.})$", \
              r"$\phi_{\rm k}\ ({\rm deg.}) $",
              r"$t_{\rm i}\ ({\rm Myr})$"]
elif n_var == 10:
    labels = [r"$M_{\rm 1, i}\ (M_{\odot})$", r"$M_{\rm 2, i}\ (M_{\odot})$", r"log $a_{\rm i}\ (R_{\odot})$", \
              r"$e_{\rm i}$", r"$v_{\rm k}\ ({\rm km}\ {\rm s}^{-1})$", r"$\theta_{\rm k}\ ({\rm rad.})$", \
              r"$\phi_{\rm k}\ ({\rm deg.}) $",
              r"$\alpha_{\rm i}\ ({\rm deg.}) $", r"$\delta_{\rm i}\ ({\rm deg.}) $", r"$t_{\rm i}\ ({\rm Myr})$"]




plot_corner(flatchain_low, flatchain_high, fig=fig, bins=30, labels=None, max_n_ticks=4, limits=plt_range, truths=truths)


# plt.tight_layout()
# plt.subplots_adjust(left=0.08, bottom=0.08, hspace=0.05, wspace=0.05)
# plt.subplots_adjust(hspace=0.1, wspace=0.1)


for i in range(n_var):
    ax[n_var-1,i].set_xlabel(labels[i])
    ax[n_var-1,i].xaxis.labelpad=5
    if i > 0:
        ax[i,0].set_ylabel(labels[i])
    ax[i,0].yaxis.labelpad=5

if file_name == 'mock_1':
    gs = gridspec.GridSpec(2, 3,
                           width_ratios=[3,2,2],
                           height_ratios=[2,5]
                           )

    in_file_low = "../data/" + file_name + "_low_derived.npy"
    derived_low = np.load(in_file_low)
    if derived_low.ndim == 4: derived_low = derived_low[0]
    derived_low = derived_low[:,delay:,:]

    n_chains, length, n_var = derived_low.shape
    derived_low = derived_low.reshape((n_chains*length, n_var))
    print(derived_low.shape)

    in_file_high = "../data/" + file_name + "_low_derived.npy"
    derived_high = np.load(in_file_high)
    if derived_high.ndim == 4: derived_high = derived_high[0]
    derived_high = derived_high[:,delay:,:]

    n_chains, length, n_var = derived_high.shape
    derived_high = derived_high.reshape((n_chains*length, n_var))
    print(derived_high.shape)

    # Observable 1 - Orbital period
    ax1 = plt.subplot(gs[1])
    tmp_x = np.linspace(6, 10.5, 100)
    tmp_y = stats.norm.pdf(tmp_x, loc=7.7, scale=0.5)
    ax1.plot(tmp_x, tmp_y, color='r')

    ax1.hist(derived_low.T[1], histtype='step', color='C1', normed=True, bins=30, zorder=10)
    ax1.hist(derived_high.T[1], histtype='step', color='C2', normed=True, bins=30, zorder=10)
    ax1.set_yticks([])
    ax1.set_xlabel(r"M$_2$ ($M_{\odot}$)")

    # Observable 2 - Eccentricity
    ax2 = plt.subplot(gs[2])
    tmp_x = np.linspace(0.45, 0.95, 100)
    tmp_y = stats.norm.pdf(tmp_x, loc=0.69, scale=0.05)
    ax2.plot(tmp_x, tmp_y, color='r')

    ax2.hist(derived_low.T[3], histtype='step', color='C1', normed=True, bins=30, zorder=10)
    ax2.hist(derived_high.T[3], histtype='step', color='C2', normed=True, bins=30, zorder=10)
    ax2.set_yticks([])
    ax2.set_xlabel(r"$e$")


if file_name == 'mock_2':
    gs = gridspec.GridSpec(3, 2,
                           width_ratios=[2,2],
                           height_ratios=[1,28,40]
                           )

    # Observable 1 - Position
    sf_history.lmc.load_sf_history()
    ra_birth_low = flatchain_low.T[7]
    dec_birth_low = flatchain_low.T[8]
    ra_birth_high = flatchain_high.T[7]
    dec_birth_high = flatchain_high.T[8]
    levels = np.linspace(1.0e7, 2.0e8, 10) / 1.0e6 * (np.pi/180.0)**2

    sf_plot, ax = get_plot_polar(30.0, sfh_function=sf_history.lmc.get_SFH, fig_in=fig, gs=gs[3],
                                 ra_dist=ra_birth_low, dec_dist=dec_birth_low, ra_dist_2=ra_birth_high, dec_dist_2=dec_birth_high,
                                 contour_CL='gaussian',
                                 dist_bins=35, sfh_bins=30, sfh_levels=levels, ra=82.84012909, dec=-70.12312498,
                                 xcenter=0.0, ycenter=20.0, xwidth=2.7, ywidth=2.2, rot_angle=0.13,
                                 xlabel="Right Ascension", ylabel="Declination", xgrid_density=5, ygrid_density=5,
                                 color_map='Blues', color_bar=True, contour_alpha=1.0, title="Star Formation Rate at 30 Myr")



if file_name == 'LMC_HMXB':
    gs = gridspec.GridSpec(3, 3,
                           width_ratios=[25,20,1],
                           height_ratios=[2,25,40]
                           )

    # Observable 1 - Position
    sf_history.lmc.load_sf_history()
    ra_birth = chains.T[7]
    dec_birth = chains.T[8]
    levels = np.linspace(1.0e7, 1.5e8, 10) / 1.0e6 * (np.pi/180.0)**2
    get_plot_polar(20.0, sfh_function=sf_history.lmc.get_SFH, fig_in=fig, gs=gs[4],
                   ra_dist=ra_birth, dec_dist=dec_birth, contour_CL='quad',
                   dist_bins=35, sfh_bins=30, sfh_levels=levels, ra=None, dec=None,
                   xcenter=0.0, ycenter=21.0, xwidth=5.0, ywidth=4.5, rot_angle=0.17,
                   xlabel="Right Ascension", ylabel="Declination", xgrid_density=6, ygrid_density=5,
                   color_map='Blues', color_bar=True, colorbar_label_y=1.11, contour_alpha=1.0, title="Star Formation Rate at 20 Myr")



if file_name == 'mock_3':
    gs = gridspec.GridSpec(4, 5,
                           width_ratios=[5,2,2,2,2],
                           height_ratios=[17,1,20,40]
                           )

    in_file_low = "../data/" + file_name + "_low_derived.npy"
    derived_low = np.load(in_file_low)
    if derived_low.ndim == 4: derived_low = derived_low[0]
    derived_low = derived_low[:,delay:,:]
    n_chains, length, n_var = derived_low.shape
    derived_low = derived_low.reshape((n_chains*length, n_var))
    print(derived_low.shape)

    in_file_high = "../data/" + file_name + "_high_derived.npy"
    derived_high = np.load(in_file_high)
    if derived_high.ndim == 4: derived_high = derived_high[0]
    derived_high = derived_high[:,delay:,:]
    n_chains, length, n_var = derived_high.shape
    derived_high = derived_high.reshape((n_chains*length, n_var))
    print(derived_high.shape)


    # Observable 1 - Companion mass
    ax1 = plt.subplot(gs[1])
    tmp_x = np.linspace(6.5, 9.0, 100)
    tmp_y = stats.norm.pdf(tmp_x, loc=7.84, scale=0.25)
    ax1.plot(tmp_x, tmp_y, color='r')

    ax1.hist(derived_low.T[1], histtype='step', color='C1', normed=True, bins=30, zorder=10)
    ax1.hist(derived_high.T[1], histtype='step', color='C2', normed=True, bins=30, zorder=10)
    ax1.set_yticks([])
    ax1.set_xlabel(r"M$_2$ ($M_{\odot}$)")

    # Observable 2 - Eccentricity
    ax2 = plt.subplot(gs[2])
    tmp_x = np.linspace(0.2, 0.6, 100)
    tmp_y = stats.norm.pdf(tmp_x, loc=0.47, scale=0.05)
    ax2.plot(tmp_x, tmp_y, color='r')
    # ax2.axvline(0.3, color='r')

    ax2.hist(derived_low.T[3], histtype='step', color='C1', normed=True, bins=30, zorder=10)
    ax2.hist(derived_high.T[3], histtype='step', color='C2', normed=True, bins=30, zorder=10)
    ax2.set_yticks([])
    ax2.set_xlabel(r"$e$")

    # Observable 3 - Orbital Period
    ax3 = plt.subplot(gs[3])
    tmp_x = np.linspace(10, 18, 100)
    tmp_y = stats.norm.pdf(tmp_x, loc=14.11, scale=1.0)
    ax3.plot(tmp_x, tmp_y, color='r')

    P_orb = A_to_P(derived_low.T[0], derived_low.T[1], derived_low.T[2])
    ax3.hist(P_orb, histtype='step', color='C1', normed=True, bins=30, zorder=10)
    P_orb = A_to_P(derived_high.T[0], derived_high.T[1], derived_high.T[2])
    ax3.hist(P_orb, histtype='step', color='C2', normed=True, bins=30, zorder=10)
    ax3.set_yticks([])
    ax3.set_xlabel(r"P$_{\rm orb}$ (days)")

    # Observable 4 - X-ray Luminosity
    ax4 = plt.subplot(gs[4])
    tmp_x = np.linspace(1.5e33, 2.4e33, 100)
    tmp_y = stats.norm.pdf(tmp_x, loc=1.94e33, scale=1.0e32)
    ax4.plot(tmp_x, tmp_y, color='r')

    L_x_low = np.zeros(len(derived_low))
    for k in range(len(L_x_low)):
        L_x_low[k] = calculate_L_x(derived_low.T[0][k], derived_low.T[5][k], derived_low.T[7][k])
    ax4.hist(L_x_low, histtype='step', color='C1', normed=True, bins=30, zorder=10)
    L_x_high = np.zeros(len(derived_high))
    for k in range(len(L_x_high)):
        L_x_high[k] = calculate_L_x(derived_high.T[0][k], derived_high.T[5][k], derived_high.T[7][k])
    ax4.hist(L_x_high, histtype='step', color='C2', normed=True, bins=30, zorder=10)
    ax4.set_yticks([])
    ax4.set_xlabel(r"L$_{\rm x}$ (erg/s)")


    gs2 = gridspec.GridSpec(3, 3,
                            width_ratios=[48, 25, 1],
                            height_ratios=[35, 30, 48]
                            )

    # Observable 5 - Position
    sf_history.lmc.load_sf_history()
    ra_birth_low = flatchain_low.T[7]
    dec_birth_low = flatchain_low.T[8]
    ra_birth_high = flatchain_high.T[7]
    dec_birth_high = flatchain_high.T[8]
    levels = np.linspace(1.0e7, 2.0e8, 10) / 1.0e6 * (np.pi/180.0)**2

    get_plot_polar(30.0, sfh_function=sf_history.lmc.get_SFH, fig_in=fig, gs=gs2[4],
        ra_dist=ra_birth_low, dec_dist=dec_birth_low, ra_dist_2=ra_birth_high, dec_dist_2=dec_birth_high,
        dist_bins=40, sfh_bins=30, sfh_levels=levels, ra=83.5744461, dec=-69.4876344,
        xcenter=0.0, ycenter=20.7, xwidth=2.5, ywidth=2.5, rot_angle=0.12,
        xlabel="Right Ascension", ylabel="Declination", xgrid_density=5, ygrid_density=5,
        color_map='Blues', color_bar=True, colorbar_label_y=1.17,
        contour_alpha=1.0, title="Star Formation Rate at 30 Myr")


if file_name == 'J0513' or file_name == 'J0513_flatsfh':

    plt.subplots_adjust(top=0.97)

    gs = gridspec.GridSpec(3, 2,
                           width_ratios=[20,18],
                           height_ratios=[1,25,40]
                           )

    # Observable 1 - Position
    sf_history.lmc.load_sf_history()
    ra_birth_1 = flatchain_low[:,7]
    dec_birth_1 = flatchain_low[:,8]
    ra_birth_2 = flatchain_high[:,7]
    dec_birth_2 = flatchain_high[:,8]

    levels = np.linspace(1.0e7, 1.5e8, 10) / 1.0e6 * (np.pi/180.0)**2
    get_plot_polar(25.0, sfh_function=sf_history.lmc.get_SFH, fig_in=fig, gs=gs[3],
        ra_dist=ra_birth_1, dec_dist=dec_birth_1, ra_dist_2=ra_birth_2, dec_dist_2=dec_birth_2,
        dist_bins=70, dist2_bins=40, sfh_bins=30, sfh_levels=levels, ra=78.36775, dec=-65.7885278,
        # xcenter=0.0, ycenter=24.2, xwidth=0.9, ywidth=0.9, rot_angle=0.205,
        xcenter=0.0, ycenter=23.7, xwidth=2.5, ywidth=1.9, rot_angle=0.205,
        xlabel="Right Ascension", ylabel="Declination", xgrid_density=5, ygrid_density=5,
        color_map='Blues', color_bar=True, contour_alpha=1.0, title="Star Formation Rate at 25 Myr")




# plt.tight_layout()

plt.savefig("../figures/" + file_name + "_corner_2dist.pdf")
