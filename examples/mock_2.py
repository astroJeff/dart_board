import sys
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')

sys.path.append("../pyBSE/")
import pybse
import dart_board
from dart_board import sf_history


# Values for mock system 2
# Input values: 12.856 7.787 5.19 0.5434 384.64 2.588 0.978 83.2554 -69.939 25.71 0.008
# Output values: 1.38 17.17, 59.81, 0.54, 44.14, 7.90e-11, 19.71, 13, 1

LMC_metallicity = 0.008

system_kwargs = {"ra" : 82.84012909 , "dec" : -70.12312498 }
pub = dart_board.DartBoard("HMXB", evolve_binary=pybse.evolve,
                           ln_prior_pos=sf_history.lmc.prior_lmc,
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
np.save("../data/mock_2_chain.npy", pub.chains)
np.save("../data/mock_2_derived.npy", pub.derived)
np.save("../data/mock_2_lnprobability.npy", pub.lnprobability)





# Pickle results
# import pickle
# pickle.dump(pub.chains, open("../data/mock_2_chain.obj", "wb"))
# pickle.dump(pub.lnprobability, open("../data/mock_2_lnprobability.obj", "wb"))
# pickle.dump(pub.derived, open("../data/mock_2_derived.obj", "wb"))
