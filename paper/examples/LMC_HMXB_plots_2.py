import numpy as np
import pickle


import matplotlib
matplotlib.use('Agg')
import corner
import matplotlib.pyplot as plt
from matplotlib import font_manager


# Load chains
chains = pickle.load(open("../data/LMC_HMXB_chain.obj", "rb"))
n_chains, length, n_var = chains.shape
chains = chains.reshape((n_chains*length, n_var))
print(chains.shape)


# Load derived
MCMC_derived = pickle.load(open("../data/LMC_HMXB_derived.obj", "rb"))
n_chains, length, n_var = MCMC_derived.shape
MCMC_derived = MCMC_derived.reshape((n_chains*length, n_var))

# Move from ln parameters to parameters in chains
chains[:,0] = np.exp(chains[:,0])
chains[:,1] = np.exp(chains[:,1])
chains[:,2] = np.exp(chains[:,2])
chains[:,7] = np.exp(chains[:,7])


# Create a corner plot to show the posterior distribution

fontProperties = {'family':'serif', 'serif':['Times New Roman'], 'weight':'normal', 'size':12}
ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', \
                                         weight='normal', stretch='normal', size=10)
plt.rc('font', **fontProperties)

# Corner plot

labels = [r"$M_{\rm 1, i}\ (M_{\odot})$",
          r"$M_{\rm 2, i}\ (M_{\odot})$",
          r"$a_{\rm i}\ (R_{\odot})$",
          r"$e_{\rm i}$",
          r"$v_{\rm k, i}\ ({\rm km}\ {\rm s}^{-1})$",
          r"$\theta_{\rm k}\ ({\rm rad.})$",
          r"$\phi_{\rm k}\ ({\rm rad.})$",
          r"$t_{\rm i}\ ({\rm Myr})$"]
# plt_range = ([13.9,13.95], [12,13], [0,4000], [0.7,1], [0,750], [0.0,np.pi], [0.0,np.pi], [15,22])
plt_range = ([0,25], [0,20], [0,4000], [0.0,1], [0,750], [0.0,np.pi], [0.0,np.pi], [10,60])



# Load traditional population synthesis results
trad_x_i = pickle.load(open("../data/LMC_HMXB_trad_x_i.obj", "rb"))
trad_likelihood = pickle.load(open("../data/LMC_HMXB_trad_likelihood.obj", "rb"))
trad_likelihood = np.exp(trad_likelihood)  # Since ln likelihoods are actually provided
trad_derived = pickle.load(open("../data/LMC_HMXB_trad_derived.obj", "rb"))




# Plot distribution of initial binary parameters
fig, ax = plt.subplots(2, 4, figsize=(10,6))

for k in range(2):
    for j in range(4):

        i = 4*k+j
        ax[k,j].hist(trad_x_i.T[i], range=plt_range[i], bins=20, normed=True, weights=trad_likelihood, color='C0', label="Traditional", alpha=0.3)
        ax[k,j].hist(chains.T[i], range=plt_range[i], bins=20, normed=True, histtype='step', color='C1', label='MCMC')

        ax[k,j].set_xlabel(labels[i])

ax[0,0].legend(loc=1,prop={'size':6})

plt.tight_layout()
plt.savefig("../figures/LMC_HMXB_compare_x_i.pdf")




# Plot distribution of final binary parameters
fig, ax = plt.subplots(1, 3, figsize=(8,3))
labels = [r"$M_{\rm 1}\ (M_{\odot})$",
          r"$M_{\rm 2}\ (M_{\odot})$",
          r"$P_{\rm orb}$"]

from dart_board import posterior
from scipy import stats
trad_Porb = posterior.A_to_P(trad_derived.T[0], trad_derived.T[1], trad_derived.T[2])

plt_range = ([1.0,5.0], [0.0,15.0], [0.0,5.0])

ax[0].hist(trad_derived.T[0], range=plt_range[0], bins=20, normed=True, weights=trad_likelihood, color='C0', alpha=0.3, label='Traditional')
ax[1].hist(trad_derived.T[1], range=plt_range[1], bins=20, normed=True, weights=trad_likelihood, color='C0', alpha=0.3)
ax[2].hist(trad_Porb, range=plt_range[2], bins=20, normed=True, weights=trad_likelihood, color='C0', alpha=0.3)


# Plot results from MCMC
MCMC_Porb = posterior.A_to_P(MCMC_derived.T[0], MCMC_derived.T[1], MCMC_derived.T[2])
ax[0].hist(MCMC_derived.T[0], range=plt_range[0], bins=20, normed=True, histtype='step', color='C1', label='MCMC')
ax[1].hist(MCMC_derived.T[1], range=plt_range[1], bins=20, normed=True, histtype='step', color='C1')
ax[2].hist(MCMC_Porb, range=plt_range[2], bins=20, normed=True, histtype='step', color='C1')


for i in range(3):
    ax[i].set_xlabel(labels[i])
    ax[i].set_xlim(plt_range[i])

ax[0].legend(prop={'size':8})

plt.tight_layout()
plt.savefig("../figures/LMC_HMXB_compare_derived.pdf")
