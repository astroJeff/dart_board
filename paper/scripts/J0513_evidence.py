import sys
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

sys.path.append("../pyBSE/")
import pybse
import dart_board
from dart_board import sf_history


LMC_metallicity = 0.008


# Load the star formation history
sf_history.lmc.load_sf_history()


def lmc_sfh_J0513(ra, dec, ln_t_b):
    """ Star formation history to guarantee walkers stay near J0513. """

    ra_J0513 = 78.36775
    dec_J0513 = -65.7885278

    # Restrict size of viable region to within 2 degrees of J0513
    if np.abs(ra - ra_J0513)*np.cos(dec*np.pi/180.0) > 2.0: return -np.inf
    if np.abs(dec - dec_J0513) > 2.0: return -np.inf

    return sf_history.lmc.prior_lmc(ra, dec, ln_t_b)






# Values for Swift J0513.4-6547 from Coe et al. 2015, MNRAS, 447, 1630
pub = dart_board.DartBoard("NSHMXB", evolve_binary=pybse.evolve, metallicity=LMC_metallicity,
                           ln_prior_pos=lmc_sfh_J0513,
                           nwalkers=320, threads=20, thin=10)

pub.aim_darts(N_iterations=10000)


start_time = time.time()
pub.throw_darts(nburn=2, nsteps=150000)
print("Simulation took",time.time()-start_time,"seconds.")




# Since emcee_PT does not have a blobs function, we must include the following calculation

if pub.ntemps is not None:

    print("Generating derived values...")

    ntemps, nchains, nsteps, nvar = pub.chains.shape
    pub.derived = np.zeros(shape=(ntemps, nchains, nsteps, 9))

    for i in range(ntemps):
        for j in range(nchains):
            for k in range(nsteps):

                x_i = pub.chains[i,j,k]

                ln_M1, ln_M2, ln_a, ecc, v_kick_1, theta_kick_1, phi_kick_1, ra, dec, ln_t = x_i
                M1 = np.exp(ln_M1)
                M2 = np.exp(ln_M2)
                a = np.exp(ln_a)
                time = np.exp(ln_t)

                P_orb = dart_board.posterior.A_to_P(M1, M2, a)

                output = pybse.evolve(M1, M2, P_orb, ecc, v_kick_1, theta_kick_1, phi_kick_1,
                                      v_kick_1, theta_kick_1, phi_kick_1,
                                      time, LMC_metallicity, False)

                pub.derived[i,j,k] = np.array([output])

    print("...finished.")




# Acceptance fraction
print("Acceptance fractions:",pub.sampler.acceptance_fraction)

# Autocorrelation length
try:
    print("Autocorrelation length:", pub.sample.acor)
except:
    print("Acceptance fraction is too low.")



# Save outputs
np.save("../data/J0513_evidence_chain.npy", pub.chains)
np.save("../data/J0513_evidence_derived.npy", pub.derived)
np.save("../data/J0513_evidence_lnprobability.npy", pub.lnprobability)
