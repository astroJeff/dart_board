import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import sys
import corner
import matplotlib.gridspec as gridspec
from scipy import stats
from dart_board import sf_history
from dart_board.sf_history.sf_plotting import get_plot_polar
from dart_board.posterior import A_to_P, calculate_L_x

file_name = sys.argv[1]
in_file = "../data/" + file_name + "_chain.npy"


if len(sys.argv) == 2:
    delay = 200
else:
    delay = int(int(sys.argv[2]) / 100)


chains = np.load(in_file)
if chains.ndim == 4: chains = chains[0]
chains = chains[:,delay:,:]

n_chains, length, n_var = chains.shape
chains = chains.reshape((n_chains*length, n_var))
print(chains.shape)


chains[:,0] = np.exp(chains[:,0])
chains[:,1] = np.exp(chains[:,1])
# chains[:,2] = np.exp(chains[:,2])
chains[:,2] = np.log10(np.exp(chains[:,2]))
chains[:,-1] = np.exp(chains[:,-1])




truths = None
if 'mock_1' in file_name:
    truths = [11.77, 8.07, np.log10(4850.81), 0.83, 153.04, 2.05, 2.33, 34.74]
    n_var = 8
    plt_range = ([4,18], [5,11], [1,5], [0,1], [0,650], [0,np.pi], [0,np.pi], [0,70])
elif 'mock_2' in file_name:
    truths = [14.113, 5.094, np.log10(45.12), 0.624, 141.12, 1.6982, 1.6266, 83.2554, -69.939, 21.89]
    n_var = 10
    plt_range = ([4,24], [2.5,15], [1,5], [0,1], [0,650], [np.pi/4.,np.pi], [0,np.pi], [81,85], [-71,-69], [0,75])
elif 'mock_3' in file_name:
    truths = [11.01, 7.42, np.log10(744.19), 0.50, 167.69, 1.79, 2.08, 83.5744461, -69.4876344, 36.59]
    n_var = 10
    plt_range = ([8,14], [6,9], [1,4.5], [0,1], [0,600], [0.0,np.pi], [0,np.pi], [81,86], [-70.4,-68.5], [0,60])
elif file_name == 'HMXB':
    n_var = 8
    plt_range = ([4,40], [2.5,20], [1,4.9], [0,1.05], [0,650], [0.0,np.pi], [0,np.pi], [0,55])
elif file_name == 'LMC_HMXB':
    n_var = 10
    plt_range = ([4,45], [2.5,22], [1,5], [0,1], [0,650], [0.0,np.pi], [0,np.pi], [69,89], [-73,-65], [0,55])
elif 'J0513_nosfh' in file_name:
    n_var = 8
    plt_range = ([4,30], [2.5,20], [1,4.5], [0,1], [0,500], [0.0,np.pi], [0,np.pi], [0,55])
elif 'J0513_flatsfh' in file_name:
    n_var = 10
    plt_range = ([4,30], [2.5,20], [1,4.5], [0,1], [0,500], [0.0,np.pi], [0,np.pi], [77.5,79], [-66.5,-65.4], [0,55])
elif 'J0513' in file_name:
    n_var = 10
    plt_range = ([4,30], [2.5,20], [1,4.5], [0,1], [0,500], [0.0,np.pi], [0,np.pi], [75,81], [-66.5,-65.0], [0,55])




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

if 'mock_1' in file_name:
    gs = gridspec.GridSpec(2, 3,
                           width_ratios=[3,2,2],
                           height_ratios=[2,5]
                           )

    in_file = "../data/" + file_name + "_derived.npy"
    derived = np.load(in_file)
    if derived.ndim == 4: derived = derived[0]
    derived = derived[:,delay:,:]

    n_chains, length, n_var = derived.shape
    derived = derived.reshape((n_chains*length, n_var))
    print(derived.shape)

    # Observable 1 - Orbital period
    ax1 = plt.subplot(gs[1])
    tmp_x = np.linspace(6, 10.5, 100)
    tmp_y = stats.norm.pdf(tmp_x, loc=7.7, scale=0.5)
    ax1.plot(tmp_x, tmp_y, color='r')

    ax1.hist(derived.T[1], histtype='step', color='k', normed=True, bins=30, zorder=10)
    ax1.set_yticks([])
    ax1.set_xlabel(r"M$_2$ ($M_{\odot}$)")

    # Observable 2 - Eccentricity
    ax2 = plt.subplot(gs[2])
    tmp_x = np.linspace(0.45, 0.95, 100)
    tmp_y = stats.norm.pdf(tmp_x, loc=0.69, scale=0.05)
    ax2.plot(tmp_x, tmp_y, color='r')

    ax2.hist(derived.T[3], histtype='step', color='k', normed=True, bins=30, zorder=10)
    ax2.set_yticks([])
    ax2.set_xlabel(r"$e$")


if 'mock_2' in file_name:
    gs = gridspec.GridSpec(3, 2,
                           width_ratios=[2,2],
                           height_ratios=[1,28,40]
                           )

    # Observable 1 - Position
    sf_history.lmc.load_sf_history()
    ra_birth = chains.T[7]
    dec_birth = chains.T[8]
    levels = np.linspace(1.0e7, 2.0e8, 10) / 1.0e6 * (np.pi/180.0)**2

    sf_plot, ax = get_plot_polar(30.0, sfh_function=sf_history.lmc.get_SFH, fig_in=fig, gs=gs[3],
                                 ra_dist=ra_birth, dec_dist=dec_birth, contour_CL='gaussian',
                                 dist_bins=35, sfh_bins=30, sfh_levels=levels, ra=82.84012909, dec=-70.12312498,
                                 xcenter=0.0, ycenter=20.0, xwidth=2.5, ywidth=2.0, rot_angle=0.1,
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



if 'mock_3' in file_name:
    gs = gridspec.GridSpec(4, 5,
                           width_ratios=[5,2,2,2,2],
                           height_ratios=[17,1,20,40]
                           )

    in_file = "../data/" + file_name + "_derived.npy"
    derived = np.load(in_file)
    if derived.ndim == 4: derived = derived[0]
    derived = derived[:,delay:,:]
    n_chains, length, n_var = derived.shape
    derived = derived.reshape((n_chains*length, n_var))
    print(derived.shape)


    # Observable 1 - Companion mass
    ax1 = plt.subplot(gs[1])
    tmp_x = np.linspace(6.5, 9.0, 100)
    tmp_y = stats.norm.pdf(tmp_x, loc=7.84, scale=0.25)
    ax1.plot(tmp_x, tmp_y, color='r')

    ax1.hist(derived.T[1], histtype='step', color='k', normed=True, bins=30, zorder=10)
    ax1.set_yticks([])
    ax1.set_xlabel(r"M$_2$ ($M_{\odot}$)")

    # Observable 2 - Eccentricity
    ax2 = plt.subplot(gs[2])
    tmp_x = np.linspace(0.2, 0.6, 100)
    tmp_y = stats.norm.pdf(tmp_x, loc=0.47, scale=0.05)
    ax2.plot(tmp_x, tmp_y, color='r')
    # ax2.axvline(0.3, color='r')

    ax2.hist(derived.T[3], histtype='step', color='k', normed=True, bins=30, zorder=10)
    ax2.set_yticks([])
    ax2.set_xlabel(r"$e$")

    # Observable 3 - Orbital Period
    ax3 = plt.subplot(gs[3])
    tmp_x = np.linspace(10, 18, 100)
    tmp_y = stats.norm.pdf(tmp_x, loc=14.11, scale=1.0)
    ax3.plot(tmp_x, tmp_y, color='r')

    P_orb = A_to_P(derived.T[0], derived.T[1], derived.T[2])
    ax3.hist(P_orb, histtype='step', color='k', normed=True, bins=30, zorder=10)
    ax3.set_yticks([])
    ax3.set_xlabel(r"P$_{\rm orb}$ (days)")

    # Observable 4 - X-ray Luminosity
    ax4 = plt.subplot(gs[4])
    tmp_x = np.linspace(1.5e33, 2.4e33, 100)
    tmp_y = stats.norm.pdf(tmp_x, loc=1.94e33, scale=1.0e32)
    ax4.plot(tmp_x, tmp_y, color='r')

    L_x = np.zeros(len(derived))
    for k in range(len(L_x)):
        L_x[k] = calculate_L_x(derived.T[0][k], derived.T[5][k], derived.T[7][k])
    ax4.hist(L_x, histtype='step', color='k', normed=True, bins=30, zorder=10)
    ax4.set_yticks([])
    ax4.set_xlabel(r"L$_{\rm x}$ (erg/s)")


    gs2 = gridspec.GridSpec(3, 3,
                            width_ratios=[48, 25, 1],
                            height_ratios=[35, 30, 48]
                            )

    # Observable 5 - Position
    sf_history.lmc.load_sf_history()
    ra_birth = chains.T[7]
    dec_birth = chains.T[8]
    levels = np.linspace(1.0e7, 2.0e8, 10) / 1.0e6 * (np.pi/180.0)**2

    get_plot_polar(30.0, sfh_function=sf_history.lmc.get_SFH, fig_in=fig, gs=gs2[4],
        ra_dist=ra_birth, dec_dist=dec_birth,
        dist_bins=40, sfh_bins=30, sfh_levels=levels, ra=83.5744461, dec=-69.4876344,
        xcenter=0.0, ycenter=20.7, xwidth=2.5, ywidth=2.5, rot_angle=0.135,
        xlabel="Right Ascension", ylabel="Declination", xgrid_density=5, ygrid_density=5,
        color_map='Blues', color_bar=True, colorbar_label_y=1.17,
        contour_alpha=1.0, title="Star Formation Rate at 30 Myr")

if 'J0513_nosfh' in file_name:
    pass

elif 'J0513' in file_name:

    gs = gridspec.GridSpec(3, 2,
                           width_ratios=[20,18],
                           height_ratios=[1,25,40]
                           )

    # Observable 1 - Position
    sf_history.lmc.load_sf_history()
    ra_birth = chains.T[7]
    dec_birth = chains.T[8]
    levels = np.linspace(1.0e7, 1.5e8, 10) / 1.0e6 * (np.pi/180.0)**2
    get_plot_polar(25.0, sfh_function=sf_history.lmc.get_SFH, fig_in=fig, gs=gs[3],
        ra_dist=ra_birth, dec_dist=dec_birth,
        dist_bins=50, sfh_bins=30, sfh_levels=levels, ra=78.36775, dec=-65.7885278,
        xcenter=0.0, ycenter=24.2, xwidth=1.4, ywidth=1.4, rot_angle=0.205,
        xlabel="Right Ascension", ylabel="Declination", xgrid_density=5, ygrid_density=5,
        color_map='Blues', color_bar=True, contour_alpha=1.0, title="Star Formation Rate at 25 Myr")




# plt.tight_layout()

plt.savefig("../figures/" + file_name + "_corner.pdf")
