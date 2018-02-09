import sys
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

sys.path.append("../pyBSE/")
import pybse
import dart_board
from dart_board import sf_history


# Values for mock system 3
# Input values: 11.01 7.42 744.19 0.20 167.69 1.79 2.08 83.2559 -69.9377 36.59
# Output values:  1.31 7.42 112.22 0.62 33.74 3.691e-13 25.58 13 1

LMC_metallicity = 0.008

system_kwargs = {"M2" : 6.92, "M2_err" : 0.25,
                 "P_orb" : 48.3, "P_orb_err" : 1.0,
                 "ecc" : 0.57, "ecc_err" : 0.05,
                 "L_x" : 6.7e32, "L_x_err" : 1.0e32,
                 "ra" : 83.41691225 , "dec" : -70.25999352}
pub = dart_board.DartBoard("HMXB", evolve_binary=pybse.evolve,
                           ln_prior_pos=sf_history.lmc.prior_lmc, nwalkers=320,
                           threads=20,
                           metallicity=LMC_metallicity,
                           ntemps=10, thin=100,
                           system_kwargs=system_kwargs)

# Darts need to be in ln
pub.aim_darts(N_iterations=100000)


start_time = time.time()
pub.throw_darts(nburn=2, nsteps=52000)
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
np.save("../data/mock_3_chain.npy", pub.chains)
np.save("../data/mock_3_derived.npy", pub.derived)
np.save("../data/mock_3_lnprobability.npy", pub.lnprobability)


# Pickle results
# import pickle
# pickle.dump(pub.chains, open("../data/mock_3_chain.obj", "wb"))
# pickle.dump(pub.lnprobability, open("../data/mock_3_lnprobability.obj", "wb"))
# pickle.dump(pub.derived, open("../data/mock_3_derived.obj", "wb"))
