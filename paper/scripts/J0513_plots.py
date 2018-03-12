import numpy as np
import sys

import matplotlib
matplotlib.use('Agg')
import corner
import matplotlib.pyplot as plt
from matplotlib import font_manager




if len(sys.argv) < 2:
    delay = 200
else:
    delay = int(sys.argv[1])


if len(sys.argv) < 3:
    sol = 'low'
else:
    sol = sys.argv[2]
if sol != 'low' and sol != 'high':
    print("You must provide either 'low' or 'high' as a possible solution.")
    sys.exit()



#### Load chains ####

# Regular model
chains = np.load("../data/J0513_" + sol + "_chain.npy")
if chains.ndim == 4: chains = chains[0]
chains = chains[:,delay:,:]
n_chains, length, n_var = chains.shape
chains = chains.reshape((n_chains*length, n_var))
print(chains.shape)

# Flat star formation history
chains_flat = np.load("../data/J0513_flatsfh_" + sol + "_chain.npy")
if chains_flat.ndim == 4: chains_flat = chains_flat[0]
chains_flat = chains_flat[:,delay:,:]
n_chains_flat, length_flat, n_var_flat = chains_flat.shape
chains_flat = chains_flat.reshape((n_chains_flat*length_flat, n_var_flat))
print(chains_flat.shape)


# Model without SFH
chains_nosfh = np.load("../data/J0513_nosfh_" + sol + "_chain.npy")
if chains_nosfh.ndim == 4: chains_nosfh = chains_nosfh[0]
chains_nosfh = chains_nosfh[:,delay:,:]
n_chains_nosfh, length_nosfh, n_var_nosfh = chains_nosfh.shape
chains_nosfh = chains_nosfh.reshape((n_chains_nosfh*length_nosfh, n_var_nosfh))
print(chains_nosfh.shape)




#### Load derived ####
MCMC_derived = np.load("../data/J0513_" + sol + "_derived.npy")
if MCMC_derived.ndim == 4: MCMC_derived = MCMC_derived[0]
MCMC_derived = MCMC_derived[:,delay:,:]
n_chains, length, n_var = MCMC_derived.shape
MCMC_derived = MCMC_derived.reshape((n_chains*length, n_var))



MCMC_derived_flat = np.load("../data/J0513_flatsfh_" + sol + "_derived.npy")
if MCMC_derived_flat.ndim == 4: MCMC_derived_flat = MCMC_derived_flat[0]
MCMC_derived_flat = MCMC_derived_flat[:,delay:,:]
n_chains_flat, length_flat, n_var_flat = MCMC_derived_flat.shape
MCMC_derived_flat = MCMC_derived_flat.reshape((n_chains_flat*length_flat, n_var_flat))


MCMC_derived_nosfh = np.load("../data/J0513_nosfh_" + sol + "_derived.npy")
if MCMC_derived_nosfh.ndim == 4: MCMC_derived_nosfh = MCMC_derived_nosfh[0]
MCMC_derived_nosfh = MCMC_derived_nosfh[:,delay:,:]
n_chains_nosfh, length_nosfh, n_var_nosfh = MCMC_derived_nosfh.shape
MCMC_derived_nosfh = MCMC_derived_nosfh.reshape((n_chains_nosfh*length_nosfh, n_var_nosfh))


# Move from ln parameters to parameters in chains
chains[:,0] = np.exp(chains[:,0])
chains[:,1] = np.exp(chains[:,1])
chains[:,2] = np.exp(chains[:,2])
chains[:,9] = np.exp(chains[:,9])

chains_flat[:,0] = np.exp(chains_flat[:,0])
chains_flat[:,1] = np.exp(chains_flat[:,1])
chains_flat[:,2] = np.exp(chains_flat[:,2])
chains_flat[:,9] = np.exp(chains_flat[:,9])

chains_nosfh[:,0] = np.exp(chains_nosfh[:,0])
chains_nosfh[:,1] = np.exp(chains_nosfh[:,1])
chains_nosfh[:,2] = np.exp(chains_nosfh[:,2])
chains_nosfh[:,7] = np.exp(chains_nosfh[:,7])




# Get indices of derived data for plots
idx_M1 = 0
idx_M2 = 1
idx_a = 2
idx_e = 3
idx_mdot1 = 5
if n_var == 9:
    idx_k1 = 7
elif n_var == 17:
    idx_k1 = 15
else:
    return

# Create a corner plot to show the posterior distribution

#fontProperties = {'family':'serif', 'serif':['Times New Roman'], 'weight':'normal', 'size':12}
#ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', \
#                                         weight='normal', stretch='normal', size=10)
#plt.rc('font', **fontProperties)

# Corner plot

labels = [r"$M_{\rm 1, i}\ (M_{\odot})$",
          r"$M_{\rm 2, i}\ (M_{\odot})$",
          r"$a_{\rm i}\ (R_{\odot})$",
          r"$e_{\rm i}$",
          r"$v_{\rm k, i}\ ({\rm km}\ {\rm s}^{-1})$",
          r"$\theta_{\rm k}\ ({\rm rad.})$",
          r"$\phi_{\rm k}\ ({\rm rad.})$",
          r"$\alpha$ (deg)",
          r"$\delta$ (deg)",
          r"$t_{\rm i}\ ({\rm Myr})$"]

if sol == 'low':
    plt_range = ([8,30], [0,30], [0,150], [0.0,1], [0,550], [0.0,np.pi], [0.0,np.pi], [77.7, 79], [-66, -65.5], [0,45])
else:
    plt_range = ([8,30], [6,30], [0,12000], [0.0,1], [0,550], [0.0,np.pi], [0.0,np.pi], [76, 80], [-67, -65], [0,55])
# plt_range = ([0,25], [0,20], [0,4000], [0.0,1], [0,750], [0.0,np.pi], [0.0,np.pi], [10,60])





# # Load traditional population synthesis results
# trad_x_i = np.load("../data/J0513_trad_chain.npy")
# trad_likelihood = np.load("../data/J0513_trad_likelihood.npy")
# trad_likelihood = np.exp(trad_likelihood)  # Since ln likelihoods are actually provided
# trad_derived = np.load("../data/J0513_trad_derived.npy")
#
# trad_x_i = trad_x_i.reshape((len(trad_likelihood), 13))
# trad_derived = trad_derived.reshape((len(trad_likelihood), 9))
# print(trad_x_i.shape)
# print(trad_derived.shape)


# Plot distribution of initial binary parameters
fig, ax = plt.subplots(2, 5, figsize=(8,4))

for k in range(2):
    for j in range(5):

        i = 5*k+j
        # if i != 7 and i != 8:
            # ax[k,j].hist(trad_x_i.T[i], range=plt_range[i], bins=20, normed=True, weights=trad_likelihood, color='C0', label="Traditional", alpha=0.3)
        ax[k,j].hist(chains.T[i], range=plt_range[i], bins=20, normed=True, color='k', alpha=0.3, label='Full Model')
        ax[k,j].hist(chains_flat.T[i], range=plt_range[i], bins=20, normed=True, histtype='step', color='C2', label='Flat SFH')

        if i < 7:
            ax[k,j].hist(chains_nosfh.T[i], range=plt_range[i], bins=20, normed=True, histtype='step', color='C4', label='No SFH', linestyle='dashed', zorder=10)
        if i == 9:
            ax[k,j].hist(chains_nosfh.T[i-2], range=plt_range[i], bins=20, normed=True, histtype='step', color='C4', label='No SFH', linestyle='dashed', zorder=10)


        ax[k,j].set_xlabel(labels[i])
        ax[k,j].set_yticklabels([])

ax[0,0].legend(loc=1,prop={'size':6})

ax[1,0].set_xticks([0.0, np.pi/2., np.pi])
ax[1,0].set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
ax[1,1].set_xticks([0.0, np.pi/2., np.pi])
ax[1,1].set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])



plt.tight_layout()
plt.savefig("../figures/J0513_" + sol + "_compare_x_i.pdf")




# Plot distribution of final binary parameters
fig, ax = plt.subplots(1, 4, figsize=(9,3))
labels = [r"$M_{\rm 1}\ (M_{\odot})$",
          r"$M_{\rm 2}\ (M_{\odot})$",
          r"$P_{\rm orb}$",
          r"$e$"]
#           r"$m_f$"]

from dart_board import posterior, forward_pop_synth
from scipy import stats

if sol == 'low':
    plt_range = ([1.2,2.6], [0.0,45.0], [24.0,30.0], [0.0, 0.25])
else:
    plt_range = ([1.2,2.6], [0.0,22.0], [24.0,30.0], [0.0, 0.25])

# Plot observational constraints
obs_Porb, obs_Porb_err = 27.405, 0.5
tmp_x = np.linspace(24.0, 30.0, 1000)
ax[2].plot(tmp_x, stats.norm.pdf(tmp_x, loc=obs_Porb, scale=obs_Porb_err), color='k', zorder=0.1)
ax[3].axvline(0.17, color='k', zorder=0.1)

# trad_Porb = posterior.A_to_P(trad_derived.T[0], trad_derived.T[1], trad_derived.T[2])
# Testing with no weights
# ax[0].hist(trad_derived.T[0], range=plt_range[0], bins=20, normed=True, color='C0', alpha=0.3, label='Traditional')
# ax[1].hist(trad_derived.T[1], range=plt_range[1], bins=20, normed=True, color='C0', alpha=0.3)
# ax[2].hist(trad_Porb, range=plt_range[2], bins=20, normed=True, color='C0', alpha=0.3)
# ax[3].hist(trad_derived.T[3], range=plt_range[3], bins=20, normed=True, color='C0', alpha=0.3)
# ax[0].hist(trad_derived.T[0], range=plt_range[0], bins=20, normed=True, weights=trad_likelihood, color='C0', alpha=0.3, label='Traditional')
# ax[1].hist(trad_derived.T[1], range=plt_range[1], bins=20, normed=True, weights=trad_likelihood, color='C0', alpha=0.3)
# ax[2].hist(trad_Porb, range=plt_range[2], bins=20, normed=True, weights=trad_likelihood, color='C0', alpha=0.3)
# ax[3].hist(trad_derived.T[3], range=plt_range[3], bins=20, normed=True, weights=trad_likelihood, color='C0', alpha=0.3)
# sin_i = np.sin(forward_pop_synth.get_theta(len(trad_Porb)))
# m_f = (trad_derived.T[1] * sin_i)**3 / (trad_derived.T[0]+trad_derived.T[1])**2
# ax[4].hist(m_f, range=plt_range[4], bins=20, normed=True, color='C0', alpha=0.3)


# Plot results from MCMC
MCMC_Porb = posterior.A_to_P(MCMC_derived.T[idx_M1], MCMC_derived.T[idx_M2], MCMC_derived.T[idx_a])
ax[0].hist(MCMC_derived.T[idx_M1], range=plt_range[0], bins=20, normed=True,
           color='k', alpha=0.3, label='Full Model', linewidth=2)
ax[1].hist(MCMC_derived.T[idx_M2], range=plt_range[1], bins=20, normed=True, color='k', alpha=0.3, linewidth=2)
ax[2].hist(MCMC_Porb, range=plt_range[2], bins=20, normed=True, color='k', alpha=0.3, linewidth=2)
ax[3].hist(MCMC_derived.T[idx_e], range=plt_range[3], bins=10, normed=True, color='k', alpha=0.3, linewidth=2)
sin_i = np.sin(forward_pop_synth.get_theta(len(MCMC_Porb)))
m_f = (MCMC_derived.T[idx_M2] * sin_i)**3 / (MCMC_derived.T[idx_M1]+MCMC_derived.T[idx_M2])**2
#ax[4].hist(m_f, range=plt_range[4], bins=10, normed=True, histtype='step', color='C1')

MCMC_Porb_flat = posterior.A_to_P(MCMC_derived_flat.T[idx_M1], MCMC_derived_flat.T[idx_M2], MCMC_derived_flat.T[idx_a])
ax[0].hist(MCMC_derived_flat.T[idx_M1], range=plt_range[0], bins=20, normed=True, histtype='step', color='C2', label='Flat SFH', linewidth=2)
ax[1].hist(MCMC_derived_flat.T[idx_M2], range=plt_range[1], bins=20, normed=True, histtype='step', color='C2', linewidth=2)
ax[2].hist(MCMC_Porb_flat, range=plt_range[2], bins=20, normed=True, histtype='step', color='C2', linewidth=2)
ax[3].hist(MCMC_derived_flat.T[idx_e], range=plt_range[3], bins=10, normed=True, histtype='step', color='C2', linewidth=2)
sin_i = np.sin(forward_pop_synth.get_theta(len(MCMC_Porb_flat)))
m_f = (MCMC_derived_flat.T[idx_M2] * sin_i)**3 / (MCMC_derived_flat.T[idx_M1]+MCMC_derived_flat.T[idx_M2])**2
#ax[4].hist(m_f, range=plt_range[4], bins=10, normed=True, histtype='step', color='C2')



MCMC_Porb_nosfh = posterior.A_to_P(MCMC_derived_nosfh.T[idx_M1], MCMC_derived_nosfh.T[idx_M2], MCMC_derived_nosfh.T[idx_a])
ax[0].hist(MCMC_derived_nosfh.T[idx_M1], range=plt_range[0], bins=20, normed=True, histtype='step', color='C4', label='No SFH', linestyle='dashed', zorder=10, linewidth=2)
ax[1].hist(MCMC_derived_nosfh.T[idx_M2], range=plt_range[1], bins=20, normed=True, histtype='step', color='C4', linestyle='dashed', zorder=10, linewidth=2)
ax[2].hist(MCMC_Porb_nosfh, range=plt_range[2], bins=20, normed=True, histtype='step', color='C4', linestyle='dashed', zorder=10, linewidth=2)
ax[3].hist(MCMC_derived_nosfh.T[idx_e], range=plt_range[3], bins=10, normed=True, histtype='step', color='C4', linestyle='dashed', zorder=10, linewidth=2)
sin_i = np.sin(forward_pop_synth.get_theta(len(MCMC_Porb_nosfh)))
m_f = (MCMC_derived_nosfh.T[idx_M2] * sin_i)**3 / ((MCMC_derived_nosfh.T[idx_M1]+MCMC_derived_nosfh.T[idx_M2])**2)
#ax[4].hist(m_f, range=plt_range[4], bins=10, normed=True, histtype='step', color='C3')



for i in range(4):
    ax[i].set_xlabel(labels[i])
    ax[i].set_xlim(plt_range[i])
    ax[i].set_yticklabels([])

ax[0].legend(prop={'size':8})

plt.tight_layout()
plt.savefig("../figures/J0513_" + sol + "_compare_derived.pdf")
