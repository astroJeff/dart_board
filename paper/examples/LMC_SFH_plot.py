import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

from dart_board import sf_history

# Load LMC's star formation history
sf_history.lmc.load_sf_history()


fig, ax = plt.subplots(2,2, figsize=(10,8))

# Remove ticks
for i in range(2):
    for j in range(2):
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])

# Use gridspec to create multipanel plot
gs = gridspec.GridSpec(2, 2)

sf_history.lmc.plot_lmc_map(15.0, fig_in=fig, gs=gs[0])
sf_history.lmc.plot_lmc_map(30.0, fig_in=fig, gs=gs[1])
sf_history.lmc.plot_lmc_map(45.0, fig_in=fig, gs=gs[2])
sf_plot = sf_history.lmc.plot_lmc_map(60.0, fig_in=fig, gs=gs[3])


fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cb = fig.colorbar(sf_plot, cax=cbar_ax, extend='max')
cb.set_label(r'$\frac{M_{\odot}}{{\rm yr\ deg.}^2}$', rotation=0, labelpad=-20, y=1.1, fontsize=14)
# cb.set_label(r'$M_{\odot}$ yr$^{-1}$ deg.$^{-2}$', rotation=270, labelpad=17)

# Convert from Msun/Myr/rad^2 to Msun/yr/deg^2
ticks = np.linspace(1.0e7, 2.0e8, 10) / 1.0e6 * (np.pi/180.0)**2
cb.set_ticks(ticks)
ticks = np.round(ticks, decimals=3)
cb.set_ticklabels(ticks.astype(str))


plt.subplots_adjust(left=0.07, bottom=0.07, right=0.89, top=0.95,
                    wspace=0.25, hspace=0.25)


# plt.tight_layout()
# plt.savefig("./tmp.pdf")
plt.savefig("../figures/LMC_SFH.pdf")
# plt.show()
