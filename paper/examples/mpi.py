import sys
import numpy as np
import emcee
from emcee.utils import MPIPool



ndim = 50
nwalkers = 250
p0 = [np.random.rand(ndim) for i in range(nwalkers)]

def lnprob(x):
    return -0.5 * np.sum(x ** 2)



pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)


pool.close()
