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

# Values for Swift J0513.4-6547 from Coe et al. 2015, MNRAS, 447, 1630
system_kwargs = {"P_orb" : 27.405, "P_orb_err" : 0.5, "ecc_max" : 0.17, "m_f" : 9.9,
                 "m_f_err" : 2.0}
pub = dart_board.DartBoard("NSHMXB", evolve_binary=pybse.evolve, metallicity=LMC_metallicity,
                           nwalkers=320, threads=20,
                           ntemps=10, thin=100,
                           system_kwargs=system_kwargs)

pub.aim_darts_PT()


start_time = time.time()
pub.throw_darts(nburn=2, nsteps=4200, method='emcee_PT')
print("Simulation took",time.time()-start_time,"seconds.")




# Since emcee_PT does not have a blobs function, we must include the following calculation

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
np.save("../data/J0513_nosfh_chain.npy", pub.chains)
np.save("../data/J0513_nosfh_derived.npy", pub.derived)
np.save("../data/J0513_nosfh_lnprobability.npy", pub.lnprobability)


# Pickle results
# import pickle
# pickle.dump(pub.chains, open("../data/J0513_nosfh_chain.obj", "wb"))
# pickle.dump(pub.lnprobability, open("../data/J0513_nosfh_lnprobability.obj", "wb"))
# pickle.dump(pub.derived, open("../data/J0513_nosfh_derived.obj", "wb"))
