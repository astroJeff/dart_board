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


truths = None
if file_name == 'mock_1':
    truths = [11.77, 8.07, 4850.81, 0.83, 153.04, 2.05, 2.33, 34.74]
    n_var = 8
    plt_range = ([4,18], [5,11], [0,6000], [0,1], [0,650], [0,np.pi], [0,np.pi], [0,70])
elif file_name == 'mock_2':
    truths = [10.98, 7.42, 744.24, 0.21, 168.87, 1.81, 2.09, 83.2554, -69.9390, 36.99]
    n_var = 10
    plt_range = ([4,24], [2.5,15], [0,3500], [0,1], [0,650], [np.pi/4.,np.pi], [0,np.pi], [81,85], [-71,-69], [0,75])
elif file_name == 'mock_3':
    truths = [11.01, 7.42, 744.19, 0.20, 167.69, 1.79, 2.08, 81.91118, -70.485899, 36.59]
    n_var = 10
    plt_range = ([10,14], [6,9], [0,2000], [0,1], [0,650], [0.0,np.pi], [0,np.pi], [80,84], [-71.2,-70.3], [0,75])
elif file_name == 'HMXB':
    n_var = 8
    plt_range = ([4,40], [2.5,20], [0,3000], [0,1], [0,650], [0.0,np.pi], [0,np.pi], [0,55])
elif file_name == 'LMC_HMXB':
    n_var = 10
    plt_range = ([4,40], [2.5,20], [0,3000], [0,1], [0,650], [0.0,np.pi], [0,np.pi], [69,89], [-73,-65], [0,55])
elif file_name == 'J0513' or file_name == 'J0513_PT':
    n_var = 10
    plt_range = ([4,35], [2.5,20], [0,150], [0,1], [0,650], [0.0,np.pi], [0,np.pi], [77.5,79], [-66.5,-65.4], [0,55])
elif file_name == 'J0513_flatsfh' or file_name == 'J0513_flatsfh_PT':
    n_var = 10
    plt_range = ([4,35], [2.5,20], [0,150], [0,1], [0,650], [0.0,np.pi], [0,np.pi], [77.5,79], [-66.5,-65.4], [0,55])
elif file_name == 'J0513_nosfh' or file_name == 'J0513_nosfh_PT':
    n_var = 8
    plt_range = ([4,35], [2.5,20], [0,150], [0,1], [0,650], [0.0,np.pi], [0,np.pi], [0,55])





fig, ax = plt.subplots(n_var,n_var, figsize=(10,10))



if n_var == 8:
    labels = [r"$M_{\rm 1, i}\ (M_{\odot})$", r"$M_{\rm 2, i}\ (M_{\odot})$", r"$a_{\rm i}\ (R_{\odot})$", \
              r"$e_{\rm i}$", r"$v_{\rm k, i}\ ({\rm km}\ {\rm s}^{-1})$", r"$\theta_{\rm k, i}\ ({\rm rad.})$", \
              r"$\phi_{\rm k, i}\ ({\rm deg.}) $",
              r"$t_{\rm i}\ ({\rm Myr})$"]
elif n_var == 10:
    labels = [r"$M_{\rm 1, i}\ (M_{\odot})$", r"$M_{\rm 2, i}\ (M_{\odot})$", r"$a_{\rm i}\ (R_{\odot})$", \
              r"$e_{\rm i}$", r"$v_{\rm k, i}\ ({\rm km}\ {\rm s}^{-1})$", r"$\theta_{\rm k, i}\ ({\rm rad.})$", \
              r"$\phi_{\rm k, i}\ ({\rm deg.}) $",
              r"$\alpha_{\rm i}\ ({\rm deg.}) $", r"$\delta_{\rm i}\ ({\rm deg.}) $", r"$t_{\rm i}\ ({\rm Myr})$"]




hist2d_kwargs = {"plot_datapoints" : False}
corner.corner(chains, fig=fig, bins=20, labels=None, max_n_ticks=4, range=plt_range, truths=truths, **hist2d_kwargs)
# corner.corner(chains, fig=fig, labels=labels, bins=20, max_n_ticks=4, range=plt_range, truths=truths, **hist2d_kwargs)


# plt.tight_layout()
plt.subplots_adjust(left=0.08, bottom=0.08, hspace=0.05, wspace=0.05)
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

    in_file = "../data/" + file_name + "_derived.obj"
    derived = pickle.load(open(in_file, "rb"))
    n_chains, length, n_var = derived.shape
    derived = derived.reshape((n_chains*length, n_var))
    print(derived.shape)

    # Observable 1 - Orbital period
    ax1 = plt.subplot(gs[1])
    tmp_x = np.linspace(6, 10.5, 100)
    tmp_y = stats.norm.pdf(tmp_x, loc=8.3, scale=0.5)
    ax1.plot(tmp_x, tmp_y, color='r')

    ax1.hist(derived.T[1], histtype='step', color='k', normed=True, bins=30, zorder=10)
    ax1.set_yticks([])
    ax1.set_xlabel(r"M$_2$ ($M_{\odot}$)")

    # Observable 2 - Eccentricity
    ax2 = plt.subplot(gs[2])
    tmp_x = np.linspace(0.45, 0.95, 100)
    tmp_y = stats.norm.pdf(tmp_x, loc=0.70, scale=0.05)
    ax2.plot(tmp_x, tmp_y, color='r')

    ax2.hist(derived.T[3], histtype='step', color='k', normed=True, bins=30, zorder=10)
    ax2.set_yticks([])
    ax2.set_xlabel(r"$e$")


if file_name == 'mock_2':
    gs = gridspec.GridSpec(3, 2,
                           width_ratios=[3,2],
                           height_ratios=[1,30,40]
                           )

    # Observable 1 - Position
    sf_history.lmc.load_sf_history()
    ra_birth = chains.T[7]
    dec_birth = chains.T[8]
    levels = np.linspace(1.0e7, 1.5e8, 10)
    get_plot_polar(30.0, sfh_function=sf_history.lmc.get_SFH, fig_in=fig, gs=gs[3],
        ra_dist=ra_birth, dec_dist=dec_birth,
        dist_bins=35, sfh_bins=30, sfh_levels=levels, ra=83.4989, dec=-70.0366,
        xcenter=0.0, ycenter=20.0, xwidth=1.0, ywidth=1.0, rot_angle=0.1,
        xlabel="Right Ascension", ylabel="Declination", xgrid_density=5, ygrid_density=5,
        color_map='Blues', color_bar=False, contour_alpha=1.0, title="Star Formation Rate at 30 Myr")



if file_name == 'LMC_HMXB':
    gs = gridspec.GridSpec(3, 2,
                           width_ratios=[3,2],
                           height_ratios=[1,30,40]
                           )

    # Observable 1 - Position
    sf_history.lmc.load_sf_history()
    ra_birth = chains.T[7]
    dec_birth = chains.T[8]
    levels = np.linspace(1.0e7, 1.5e8, 10)
    get_plot_polar(20.0, sfh_function=sf_history.lmc.get_SFH, fig_in=fig, gs=gs[3],
        ra_dist=ra_birth, dec_dist=dec_birth,
        dist_bins=35, sfh_bins=30, sfh_levels=levels, ra=None, dec=None,
        xcenter=0.0, ycenter=21.0, xwidth=5.0, ywidth=4.0, rot_angle=0.2,
        xlabel="Right Ascension", ylabel="Declination", xgrid_density=6, ygrid_density=5,
        color_map='Blues', color_bar=False, contour_alpha=1.0, title="Star Formation Rate at 20 Myr")



if file_name == 'mock_3':
    gs = gridspec.GridSpec(4, 5,
                           width_ratios=[3,2,2,2,2],
                           height_ratios=[20,1,20,40]
                           )

    in_file = "../data/" + file_name + "_derived.obj"
    derived = pickle.load(open(in_file, "rb"))
    n_chains, length, n_var = derived.shape
    derived = derived.reshape((n_chains*length, n_var))
    print(derived.shape)


    # Observable 1 - Companion mass
    ax1 = plt.subplot(gs[1])
    tmp_x = np.linspace(6, 9.0, 100)
    tmp_y = stats.norm.pdf(tmp_x, loc=7.5, scale=0.25)
    ax1.plot(tmp_x, tmp_y, color='r')

    ax1.hist(derived.T[1], histtype='step', color='k', normed=True, bins=30, zorder=10)
    ax1.set_yticks([])
    ax1.set_xlabel(r"M$_2$ ($M_{\odot}$)")

    # Observable 2 - Eccentricity
    ax2 = plt.subplot(gs[2])
    tmp_x = np.linspace(0.4, 0.8, 100)
    tmp_y = stats.norm.pdf(tmp_x, loc=0.6, scale=0.05)
    ax2.plot(tmp_x, tmp_y, color='r')
    # ax2.axvline(0.3, color='r')

    ax2.hist(derived.T[3], histtype='step', color='k', normed=True, bins=30, zorder=10)
    ax2.set_yticks([])
    ax2.set_xlabel(r"$e$")

    # Observable 3 - Orbital Period
    ax3 = plt.subplot(gs[3])
    tmp_x = np.linspace(11, 19, 100)
    tmp_y = stats.norm.pdf(tmp_x, loc=15.0, scale=1.0)
    ax3.plot(tmp_x, tmp_y, color='r')

    P_orb = A_to_P(derived.T[0], derived.T[1], derived.T[2])
    ax3.hist(P_orb, histtype='step', color='k', normed=True, bins=30, zorder=10)
    ax3.set_yticks([])
    ax3.set_xlabel(r"P$_{\rm orb}$ (days)")

    # Observable 4 - X-ray Luminosity
    ax4 = plt.subplot(gs[4])
    tmp_x = np.linspace(1.6e33, 2.2e33, 100)
    tmp_y = stats.norm.pdf(tmp_x, loc=1.9e33, scale=1.0e32)
    ax4.plot(tmp_x, tmp_y, color='r')

    L_x = np.zeros(len(derived))
    for k in range(len(L_x)):
        L_x[k] = calculate_L_x(derived.T[0][k], derived.T[5][k], derived.T[7][k])
    ax4.hist(L_x, histtype='step', color='k', normed=True, bins=30, zorder=10)
    ax4.set_yticks([])
    ax4.set_xlabel(r"L$_{\rm x}$ (erg/s)")


    gs2 = gridspec.GridSpec(3, 3,
                            width_ratios=[60,20,1],
                            height_ratios=[45, 40, 50]
                            )

    # Observable 5 - Position
    sf_history.lmc.load_sf_history()
    ra_birth = chains.T[7]
    dec_birth = chains.T[8]
    get_plot_polar(30.0, sfh_function=sf_history.lmc.get_SFH, fig_in=fig, gs=gs2[4],
        ra_dist=ra_birth, dec_dist=dec_birth,
        dist_bins=35, sfh_bins=30, sfh_levels=None, ra=81.5858, dec=-70.8483,
        xcenter=0.0, ycenter=19.2, xwidth=1.0, ywidth=1.0, rot_angle=0.135,
        xlabel="Right Ascension", ylabel="Declination", xgrid_density=5, ygrid_density=5,
        color_map='Blues', color_bar=False, contour_alpha=1.0, title="Star Formation Rate at 30 Myr")




# plt.tight_layout()

plt.savefig("../figures/" + file_name + "_corner.pdf")
