import sys
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

sys.path.append("../pyBSE/")
import pybse
import dart_board
from dart_board import sf_history


# Values for mock system 1
# Input values: 11.77 8.07 4850.81 0.83 153.04 2.05 2.33 34.74
# Output values: 1.48 8.09 179.89 0.68 39.65 1.32e-12 22.12 13 1

LMC_metallicity = 0.008

system_kwargs = {"M2" : 7.7, "M2_err" : 0.5, "ecc" : 0.69, "ecc_err" : 0.05}
pub = dart_board.DartBoard("NSHMXB", evolve_binary=pybse.evolve,
                           nwalkers=320, threads=20,
                           metallicity=LMC_metallicity,
                           ntemps=10, thin=100,
                           system_kwargs=system_kwargs)

pub.aim_darts_PT()


start_time = time.time()
pub.throw_darts(nburn=2, nsteps=2200, method='emcee_PT')
print("Simulation took",time.time()-start_time,"seconds.")



# Since emcee_PT does not have a blobs function, we must include the following calculation

print("Generating derived values...")

ntemps, nchains, nsteps, nvar = pub.chains.shape
pub.derived = np.zeros(shape=(ntemps, nchains, nsteps, 9))

for i in range(ntemps):
    for j in range(nchains):
        for k in range(nsteps):

            x_i = pub.chains[i,j,k]

            ln_M1, ln_M2, ln_a, ecc, v_kick_1, theta_kick_1, phi_kick_1, ln_t = x_i
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
np.save("../data/mock_1_chain.npy", pub.chains)
np.save("../data/mock_1_derived.npy", pub.derived)
np.save("../data/mock_1_lnprobability.npy", pub.lnprobability)



# Pickle results
# import pickle
# pickle.dump(pub.chains, open("../data/mock_1_chain.obj", "wb"))
# pickle.dump(pub.lnprobability, open("../data/mock_1_lnprobability.obj", "wb"))
# pickle.dump(pub.derived, open("../data/mock_1_derived.obj", "wb"))
