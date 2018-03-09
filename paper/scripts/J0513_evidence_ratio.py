import numpy as np
import dart_board
import pybse
from dart_board import sf_history
from dart_board import constants as c



# Load data
chains = np.load("../data/J0513_evidence_chain.npy")
derived = np.load("../data/J0513_evidence_derived.npy")


# Create flatchains
nwalkers, nsteps, nvars = chains.shape
chains = chains.reshape((nwalkers*nsteps, nvars))

nwalkers, nsteps, nvars = derived.shape
derived = derived.reshape((nwalkers*nsteps, nvars))

print(chains.shape)
print(derived.shape)



# Create dart_board object
LMC_metallicity = 0.008
c.distance = sf_history.lmc.lmc_dist

# Values for Swift J0513.4-6547 from Coe et al. 2015, MNRAS, 447, 1630
system_kwargs = {"P_orb" : 27.405, "P_orb_err" : 0.5, "ecc_max" : 0.17, "m_f" : 9.9,
                 "m_f_err" : 2.0, "ra" : 78.36775, "dec" : -65.7885278}
J0513 = dart_board.DartBoard("NSHMXB", evolve_binary=pybse.evolve, metallicity=LMC_metallicity,
                             ln_prior_pos=sf_history.lmc.prior_lmc,
                             nwalkers=320, threads=20, thin=100,
                             system_kwargs=system_kwargs)



# Calculate likelihoods
likelihood = np.zeros(len(chains))

for i, x_in in enumerate(chains):

    ll = dart_board.posterior.posterior_properties(x_in, derived[i], J0513)

    if np.isinf(ll) or np.isnan(ll):
        likelihood[i] = 0.0
    else:
        likelihood[i] = np.exp(ll)


# Check to make sure sources exist
print(np.any(likelihood != 0.0))
print(len(np.where(likelihood != 0.0)[0]))


# Save data
np.save("../paper/data/J0513_evidence_lnlikelihood.npy", np.log(likelihood))
