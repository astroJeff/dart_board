import numpy as np
import pickle


import matplotlib
matplotlib.use('Agg')
import corner
import matplotlib.pyplot as plt
from matplotlib import font_manager


#### Load chains ####

# Regular model
chains = pickle.load(open("../data/J0513_chain.obj", "rb"))
n_chains, length, n_var = chains.shape
chains = chains.reshape((n_chains*length, n_var))
print(chains.shape)

# Flat star formation history
chains_flat = pickle.load(open("../data/J0513_flatsfh_chain.obj", "rb"))
n_chains_flat, length_flat, n_var_flat = chains_flat.shape
chains_flat = chains_flat.reshape((n_chains_flat*length_flat, n_var_flat))
print(chains_flat.shape)



#### Load derived ####
MCMC_derived = pickle.load(open("../data/J0513_derived.obj", "rb"))
n_chains, length, n_var = MCMC_derived.shape
MCMC_derived = MCMC_derived.reshape((n_chains*length, n_var))

MCMC_derived_flat = pickle.load(open("../data/J0513_flatsfh_derived.obj", "rb"))
n_chains_flat, length_flat, n_var_flat = MCMC_derived_flat.shape
MCMC_derived_flat = MCMC_derived_flat.reshape((n_chains_flat*length_flat, n_var_flat))


# Move from ln parameters to parameters in chains
chains[:,0] = np.exp(chains[:,0])
chains[:,1] = np.exp(chains[:,1])
chains[:,2] = np.exp(chains[:,2])
chains[:,9] = np.exp(chains[:,9])

print(np.mean(chains[:,2]))

chains_flat[:,0] = np.exp(chains_flat[:,0])
chains_flat[:,1] = np.exp(chains_flat[:,1])
chains_flat[:,2] = np.exp(chains_flat[:,2])
chains_flat[:,9] = np.exp(chains_flat[:,9])

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
          r"$\alpha$ (deg)",
          r"$\delta$ (deg)",
          r"$t_{\rm i}\ ({\rm Myr})$"]
plt_range = ([8,20], [0,15], [0,4000], [0.0,1], [0,750], [0.0,np.pi], [0.0,np.pi], [77.7, 79], [-66, -65.5], [0,45])
# plt_range = ([0,25], [0,20], [0,4000], [0.0,1], [0,750], [0.0,np.pi], [0.0,np.pi], [10,60])





# # Load traditional population synthesis results
trad_x_i = pickle.load(open("../data/J0513_trad_x_i.obj", "rb"))
trad_likelihood = pickle.load(open("../data/J0513_trad_likelihood.obj", "rb"))
trad_likelihood = np.exp(trad_likelihood)  # Since ln likelihoods are actually provided
trad_derived = pickle.load(open("../data/J0513_trad_derived.obj", "rb"))



# Plot distribution of initial binary parameters
fig, ax = plt.subplots(2, 5, figsize=(10,6))

for k in range(2):
    for j in range(5):

        i = 5*k+j
        ax[k,j].hist(trad_x_i.T[i], range=plt_range[i], bins=20, normed=True, weights=trad_likelihood, color='C0', label="Traditional", alpha=0.3)
        ax[k,j].hist(chains.T[i], range=plt_range[i], bins=20, normed=True, histtype='step', color='C1', label='MCMC')
        ax[k,j].hist(chains_flat.T[i], range=plt_range[i], bins=20, normed=True, histtype='step', color='C2', label='MCMC flat')

        ax[k,j].set_xlabel(labels[i])

ax[0,0].legend(loc=1,prop={'size':6})



plt.tight_layout()
plt.savefig("../figures/J0513_compare_x_i.pdf")




# Plot distribution of final binary parameters
fig, ax = plt.subplots(1, 5, figsize=(12,3))
labels = [r"$M_{\rm 1}\ (M_{\odot})$",
          r"$M_{\rm 2}\ (M_{\odot})$",
          r"$P_{\rm orb}$",
          r"$e$",
          r"$m_f$"]

from dart_board import posterior, forward_pop_synth
from scipy import stats
# trad_Porb = posterior.A_to_P(trad_derived.T[0], trad_derived.T[1], trad_derived.T[2])

plt_range = ([1.2,1.7], [0.0,25.0], [24.0,30.0], [0.0, 0.25], [0.0, 15.0])

# Plot observational constraints
obs_Porb, obs_Porb_err = 27.405, 0.5
tmp_x = np.linspace(24.0, 30.0, 1000)
ax[2].plot(tmp_x, stats.norm.pdf(tmp_x, loc=obs_Porb, scale=obs_Porb_err), color='k')
ax[3].axvline(0.17, color='k')

trad_Porb = posterior.A_to_P(trad_derived.T[0], trad_derived.T[1], trad_derived.T[2])
ax[0].hist(trad_derived.T[0], range=plt_range[0], bins=20, normed=True, weights=trad_likelihood, color='C0', alpha=0.3, label='Traditional')
ax[1].hist(trad_derived.T[1], range=plt_range[1], bins=20, normed=True, weights=trad_likelihood, color='C0', alpha=0.3)
ax[2].hist(trad_Porb, range=plt_range[2], bins=20, normed=True, weights=trad_likelihood, color='C0', alpha=0.3)
ax[3].hist(trad_derived.T[3], range=plt_range[3], bins=20, normed=True, weights=trad_likelihood, color='C0', alpha=0.3)
sin_i = np.sin(forward_pop_synth.get_theta(len(trad_Porb)))
m_f = (trad_derived.T[1] * sin_i)**3 / (trad_derived.T[0]+trad_derived.T[1])**2
ax[4].hist(m_f, range=plt_range[4], bins=20, normed=True, color='C0', alpha=0.3)


# Plot results from MCMC
MCMC_Porb = posterior.A_to_P(MCMC_derived.T[0], MCMC_derived.T[1], MCMC_derived.T[2])
ax[0].hist(MCMC_derived.T[0], range=plt_range[0], bins=20, normed=True, histtype='step', color='C1', label='MCMC')
ax[1].hist(MCMC_derived.T[1], range=plt_range[1], bins=20, normed=True, histtype='step', color='C1')
ax[2].hist(MCMC_Porb, range=plt_range[2], bins=20, normed=True, histtype='step', color='C1')
ax[3].hist(MCMC_derived.T[3], range=plt_range[3], bins=10, normed=True, histtype='step', color='C1')
sin_i = np.sin(forward_pop_synth.get_theta(len(MCMC_Porb)))
m_f = (MCMC_derived.T[1] * sin_i)**3 / (MCMC_derived.T[0]+MCMC_derived.T[1])**2
ax[4].hist(m_f, range=plt_range[4], bins=10, normed=True, histtype='step', color='C1')

MCMC_Porb_flat = posterior.A_to_P(MCMC_derived_flat.T[0], MCMC_derived_flat.T[1], MCMC_derived_flat.T[2])
ax[0].hist(MCMC_derived_flat.T[0], range=plt_range[0], bins=20, normed=True, histtype='step', color='C2', label='MCMC flat')
ax[1].hist(MCMC_derived_flat.T[1], range=plt_range[1], bins=20, normed=True, histtype='step', color='C2')
ax[2].hist(MCMC_Porb_flat, range=plt_range[2], bins=20, normed=True, histtype='step', color='C2')
ax[3].hist(MCMC_derived_flat.T[3], range=plt_range[3], bins=10, normed=True, histtype='step', color='C2')
sin_i = np.sin(forward_pop_synth.get_theta(len(MCMC_Porb_flat)))
m_f = (MCMC_derived_flat.T[1] * sin_i)**3 / (MCMC_derived_flat.T[0]+MCMC_derived_flat.T[1])**2
ax[4].hist(m_f, range=plt_range[4], bins=10, normed=True, histtype='step', color='C2')



for i in range(5):
    ax[i].set_xlabel(labels[i])
    ax[i].set_xlim(plt_range[i])

ax[0].legend(prop={'size':8})

plt.tight_layout()
plt.savefig("../figures/J0513_compare_derived.pdf")
