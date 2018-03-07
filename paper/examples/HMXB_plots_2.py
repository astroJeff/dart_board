import numpy as np
import pickle


import matplotlib
matplotlib.use('Agg')
import corner
import matplotlib.pyplot as plt
from matplotlib import font_manager
from dart_board import posterior


# Fraction of data to ignore 
frac = 0.99


# Load chains
chains = np.load("../data/HMXB_chain.npy")
if chains.ndim == 4: chains = chains[0]
n_chains, length, n_var = chains.shape
chains = chains[:,int(length*frac):,:]  
n_chains, length, n_var = chains.shape
chains = chains.reshape((n_chains*length, n_var))
print(chains.shape)



# Load derived
MCMC_derived = np.load("../data/HMXB_derived.npy")
if MCMC_derived.ndim == 4: MCMC_derived = MCMC_derived[0] 
n_chains, length, n_var = MCMC_derived.shape
MCMC_derived = MCMC_derived[:,int(length*frac):,:] 
n_chains, length, n_var = MCMC_derived.shape
MCMC_derived = MCMC_derived.reshape((n_chains*length, n_var))

print(MCMC_derived.shape) 




# Move from ln parameters to parameters in chains
chains[:,0] = np.exp(chains[:,0])
chains[:,1] = np.exp(chains[:,1])
chains[:,2] = np.exp(chains[:,2])
chains[:,7] = np.exp(chains[:,7])


# Create a corner plot to show the posterior distribution

# fontProperties = {'family':'serif', 'serif':['Times New Roman'], 'weight':'normal', 'size':12}
# ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', \
#                                          weight='normal', stretch='normal', size=10)
# plt.rc('font', **fontProperties)

# Corner plot

labels = [r"$M_{\rm 1, i}\ (M_{\odot})$",
          r"$M_{\rm 2, i}\ (M_{\odot})$",
          r"log $a_{\rm i}\ (R_{\odot})$",
          r"$e_{\rm i}$",
          r"$v_{\rm k, i}\ ({\rm km}\ {\rm s}^{-1})$",
          r"$\theta_{\rm k}\ ({\rm rad.})$",
          r"$\phi_{\rm k}\ ({\rm rad.})$",
          r"$t_{\rm i}\ ({\rm Myr})$"]
# plt_range = ([13.9,13.95], [12,13], [0,4000], [0.7,1], [0,750], [0.0,np.pi], [0.0,np.pi], [15,22])
# plt_range = ([0,25], [0,20], [0,4000], [0.0,1], [0,750], [0.0,np.pi], [0.0,np.pi], [10,60])
plt_range = ([0,40], [0,30], [1,4], [0.0,1], [0,750], [0.0,np.pi], [0.0,np.pi], [0,60])


# Load traditional population synthesis results
trad_x_i = np.load("../data/HMXB_trad_chain.npy")
length, ndim = trad_x_i.shape 

trad_likelihood = np.load("../data/HMXB_trad_lnprobability.npy")
trad_derived = np.load("../data/HMXB_trad_derived.npy")

length, ndim = trad_x_i.shape
trad_x_i = trad_x_i[int(length*frac):,:] 
trad_likelihood = trad_likelihood[int(length*frac):]
trad_derived = trad_derived[int(length*frac):,:]

# trad_x_i = trad_x_i.reshape((len(trad_likelihood), 14))
# trad_derived = trad_derived.reshape((len(trad_likelihood), 9))
print(trad_x_i.shape)
print(trad_derived.shape)



# Make the orbital separation distribution in log-scale
chains.T[2] = np.log10(chains.T[2])
trad_x_i.T[2] = posterior.P_to_A(trad_x_i.T[0], trad_x_i.T[1], trad_x_i.T[2])
trad_x_i.T[2] = np.log10(trad_x_i.T[2])


# Plot distribution of initial binary parameters
fig, ax = plt.subplots(2, 4, figsize=(8,4.5))

for k in range(2):
    for j in range(4):

        i = 4*k+j
        trad_i = i
        if i == 7: trad_i = 12
        ax[k,j].hist(trad_x_i.T[trad_i], range=plt_range[i], bins=30, normed=True, histtype='step', color='C0', label="Traditional")
        # ax[k,j].hist(trad_x_i.T[trad_i], range=plt_range[i], bins=20, normed=True, weights=trad_likelihood, color='C0', label="Traditional", alpha=0.3)
        ax[k,j].hist(chains.T[i], range=plt_range[i], bins=30, normed=True, color='k', label='MCMC', alpha=0.3)

        ax[k,j].set_xlabel(labels[i])
        ax[k,j].set_yticklabels([])

ax[0,0].legend(loc=1,prop={'size':6})

ax[1,1].set_xticks([0.0, np.pi/2., np.pi])
ax[1,2].set_xticks([0.0, np.pi/2., np.pi])
ax[1,1].set_xticklabels(["0", r"$\pi$/2", r"$\pi$"])
ax[1,2].set_xticklabels(["0", r"$\pi$/2", r"$\pi$"])



plt.tight_layout()
plt.savefig("../figures/HMXB_compare_x_i.pdf")




# Plot distribution of final binary parameters
fig, ax = plt.subplots(1, 3, figsize=(8,3))
labels = [r"$M_{\rm 1}\ (M_{\odot})$",
          r"$M_{\rm 2}\ (M_{\odot})$",
          r"$P_{\rm orb}$"]

from dart_board import posterior
from scipy import stats
trad_Porb = posterior.A_to_P(trad_derived.T[0], trad_derived.T[1], trad_derived.T[2])

plt_range = ([1.0,2.5], [0.0,60.0], [0.0,500.0])

# ax[0].hist(trad_derived.T[0], range=plt_range[0], bins=20, normed=True, weights=trad_likelihood, color='C0', alpha=0.3, label='Traditional')
# ax[1].hist(trad_derived.T[1], range=plt_range[1], bins=20, normed=True, weights=trad_likelihood, color='C0', alpha=0.3)
# ax[2].hist(trad_Porb, range=plt_range[2], bins=20, normed=True, weights=trad_likelihood, color='C0', alpha=0.3)
ax[0].hist(trad_derived.T[0], range=plt_range[0], bins=40, normed=True, histtype='step', color='C0', label='Traditional')
ax[1].hist(trad_derived.T[1], range=plt_range[1], bins=40, normed=True, histtype='step', color='C0')
ax[2].hist(trad_Porb, range=plt_range[2], bins=40, normed=True, histtype='step', color='C0')


# Plot results from MCMC
MCMC_Porb = posterior.A_to_P(MCMC_derived.T[0], MCMC_derived.T[1], MCMC_derived.T[2])
ax[0].hist(MCMC_derived.T[0], range=plt_range[0], bins=40, normed=True, color='k', label='MCMC', alpha=0.3)
ax[1].hist(MCMC_derived.T[1], range=plt_range[1], bins=40, normed=True, color='k', alpha=0.3)
ax[2].hist(MCMC_Porb, range=plt_range[2], bins=40, normed=True, color='k', alpha=0.3)


for i in range(3):
    ax[i].set_xlabel(labels[i])
    ax[i].set_xlim(plt_range[i])
    ax[i].set_yticklabels([])

ax[0].legend(prop={'size':8})

plt.tight_layout()
plt.savefig("../figures/HMXB_compare_derived.pdf")
