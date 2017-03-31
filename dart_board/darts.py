#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An MCMC statistical wrapper on top of a binary population synthesis code.

This wrapper uses "emcee" by Foreman-Mackey et al. (2012): github.com/dfm/emcee

"""

import sys
import numpy as np
import emcee

from . import priors
from . import posterior



class DartBoard():
    """
    The ensemble sampler that searches the initial condition parameter space
    for binaries that fit the input conditions.

    """

    def __init__(self,
                 binary_type,
                 metallicity=0.02,
                 ln_prior_M1=priors.ln_prior_M1,
                 ln_prior_M2=priors.ln_prior_M2,
                 ln_prior_a=priors.ln_prior_a,
                 ln_prior_ecc=priors.ln_prior_ecc,
                 ln_prior_v_kick=priors.ln_prior_v_kick,
                 ln_prior_theta_kick=priors.ln_prior_theta_kick,
                 ln_prior_phi_kick=priors.ln_prior_phi_kick,
                 ln_prior_t=priors.ln_prior_t,
                 ln_prior_pos=None,
                 ln_posterior_function=None,
                 nwalkers=80,
                 threads=1,
                 mpi=False,
                 evolve_binary=None,
                 kwargs={}):

        # First, check that a binary evolution scheme was provided
        if evolve_binary is None:
            print("You must include a binary evolution scheme, e.g. pybse.evolv_wrapper")
            sys.exit(-1)

        # The type of binary we are modeling
        self.binary_type = binary_type

        # Binary evolution parameters
        self.metallicity = metallicity

        # Set the functions for the priors on each parameter
        self.prior_M1 = ln_prior_M1
        self.prior_M2 = ln_prior_M2
        self.prior_a = ln_prior_a
        self.prior_ecc = ln_prior_ecc
        self.prior_v_kick1 = ln_prior_v_kick
        self.prior_theta_kick1 = ln_prior_theta_kick
        self.prior_phi_kick1 = ln_prior_phi_kick
        self.prior_v_kick2 = ln_prior_v_kick
        self.prior_theta_kick2 = ln_prior_theta_kick
        self.prior_phi_kick2 = ln_prior_phi_kick

        self.prior_pos = None
        if ln_prior_pos is None:
            self.prior_t = ln_prior_t
        else:
            self.prior_pos = ln_prior_pos

        # Set the posterior probability function
        if ln_posterior_function is None:
            self.posterior_function = posterior.ln_posterior
        else:
            self.posterior_function = ln_posterior_function

        # emcee parameters
        self.nwalkers = nwalkers
        self.threads = threads
        self.mpi = mpi

        # Current dart positions
        self.p0 = []

        # Binary evolution function
        self.evolve_binary = evolve_binary

        # The type of objects
        self.second_SN = False
        if np.any(binary_type == np.array(["BHBH", "NSNS", "BHNS"])):
            self.second_SN = True

        def print_keyword_args(**kwargs):
            # kwargs is a dict of the keyword args passed to the function
            for key, value in kwargs.iteritems():
                print("%s = %s" % (key, value))

        # Oservables to match
        self.kwargs = kwargs
        if not self.kwargs == {}:
            print_keyword_args(**kwargs)




        # Determine the number of dimensions
        if ln_prior_pos is None:
            if self.second_SN:
                self.dim = 11
            else:
                self.dim = 8
        else:
            if self.second_SN:
                self.dim = 13
            else:
                self.dim = 10


        # Saved data
        self.sampler = []
        self.binary_data = []


    def aim_darts(self):
        """
        Find a viable region of parameter space then create a ball around it.

        """


        # Set walkers
        print("Setting walkers...")


        # Initial values
        M1 = 12.0
        M2 = 10.0
        ecc = 0.41
        metallicity = self.metallicity
        orbital_period = 500.0
        time = 30.0

        # SN kicks
        v_kick1 = 0.0
        theta_kick1 = 0.0
        phi_kick1 = 0.0
        if self.second_SN:
            v_kick2 = 0.0
            theta_kick2 = 0.0
            phi_kick2 = 0.0

        # Iterate randomly through initial conditions until a viable parameter set is found
        for i in np.arange(100000):

            M1 = 5.0 * np.random.uniform(size=1) + 8.0
            M2 = M1 * (0.2 * np.random.uniform(size=1) + 0.8)
            a = 300.0 * np.random.uniform(size=1) + 20.0
            ecc = np.random.uniform(size=1)

            v_kick1 = 300.0 * np.random.uniform(size=1) + 20.0
            theta_kick1 = np.pi * np.random.uniform(size=1)
            phi_kick1 = np.pi * np.random.uniform(size=1)
            if self.second_SN:
                v_kick2 = 300.0 * np.random.uniform(size=1) + 20.0
                theta_kick2 = np.pi * np.random.uniform(size=1)
                phi_kick2 = np.pi * np.random.uniform(size=1)

            time = 40.0 * np.random.uniform(size=1) + 10.0

            if self.second_SN:
                x = M1, M2, a, ecc, v_kick1, theta_kick1, phi_kick1, v_kick2, theta_kick2, phi_kick2, time
            else:
                x = M1, M2, a, ecc, v_kick1, theta_kick1, phi_kick1, time

            # If the system has a viable posterior probability
            if self.posterior_function(x, self)[0] > -10000.0:

                # ...then use it as our starting system
                break


        if i==99999:
            print("Walkers could not be set")
            sys.exit(-1)



        # Now to generate a ball around these parameters

        # Binary parameters
        M1_set = M1 + np.random.normal(0.0, 0.1, self.nwalkers)
        M2_set = M2 + np.random.normal(0.0, 0.1, self.nwalkers)
        ecc_set = ecc + np.random.normal(0.0, 0.01, self.nwalkers)
        a_set = a + np.random.normal(0.0, 2.0, self.nwalkers)

        # SN kick perameters
        v_kick1_set = v_kick1 + np.random.normal(0.0, 1.0, self.nwalkers)
        theta_kick1_set = theta_kick1 + np.random.normal(0.0, 0.01, self.nwalkers)
        phi_kick1_set = phi_kick1 + np.random.normal(0.0, 0.01, self.nwalkers)
        if self.second_SN:
            v_kick2_set = v_kick2 + np.random.normal(0.0, 1.0, self.nwalkers)
            theta_kick2_set = theta_kick2 + np.random.normal(0.0, 0.01, self.nwalkers)
            phi_kick2_set = phi_kick2 + np.random.normal(0.0, 0.01, self.nwalkers)

        # Birth time
        time_set = time + np.random.normal(0.0, 0.2, self.nwalkers)


        # Check if any of these have posteriors with -infinity
        for i in np.arange(self.nwalkers):

            if self.second_SN:
                p = M1_set[i], M2_set[i], a_set[i], ecc_set[i], \
                        v_kick1_set[i], theta_kick1_set[i], phi_kick1_set[i], \
                        v_kick2_set[i], theta_kick2_set[i], phi_kick2_set[i], \
                        time_set[i]
            else:
                p = M1_set[i], M2_set[i], a_set[i], ecc_set[i], \
                        v_kick1_set[i], theta_kick1_set[i], phi_kick1_set[i], \
                        time_set[i]

            ln_posterior = self.posterior_function(p, self)[0]


            while ln_posterior < -10000.0:


                # Binary parameters
                M1_set[i] = M1 + np.random.normal(0.0, 0.1, 1)
                M2_set[i] = M2 + np.random.normal(0.0, 0.1, 1)
                ecc_set[i] = ecc + np.random.normal(0.0, 0.01, 1)
                a_set[i] = a + np.random.normal(0.0, 2.0, 1)

                # SN kick perameters
                v_kick1_set[i] = v_kick1 + np.random.normal(0.0, 1.0, 1)
                theta_kick1_set[i] = theta_kick1 + np.random.normal(0.0, 0.01, 1)
                phi_kick1_set[i] = phi_kick1 + np.random.normal(0.0, 0.01, 1)
                if self.second_SN:
                    v_kick2_set[i] = v_kick2 + np.random.normal(0.0, 1.0, 1)
                    theta_kick2_set[i] = theta_kick2 + np.random.normal(0.0, 0.01, 1)
                    phi_kick2_set[i] = phi_kick2 + np.random.normal(0.0, 0.01, 1)

                # Birth time
                time_set[i] = time + np.random.normal(0.0, 0.2, 1)

                if self.second_SN:
                    p = M1_set[i], M2_set[i], a_set[i], ecc_set[i], \
                            v_kick1_set[i], theta_kick1_set[i], phi_kick1_set[i], \
                            v_kick2_set[i], theta_kick2_set[i], phi_kick2_set[i], \
                            time_set[i]
                else:
                    p = M1_set[i], M2_set[i], a_set[i], ecc_set[i], \
                            v_kick1_set[i], theta_kick1_set[i], phi_kick1_set[i], \
                            time_set[i]

                ln_posterior = self.posterior_function(p, self)[0]



        # Save and return the walker positions
        if self.second_SN:
            self.p0 = np.array([M1_set, M2_set, a_set, ecc_set, v_kick1_set, theta_kick1_set, \
                                phi_kick1_set, v_kick2_set, theta_kick2_set, phi_kick2_set, \
                                time_set]).T
        else:
            self.p0 = np.array([M1_set, M2_set, a_set, ecc_set, v_kick1_set, theta_kick1_set, \
                                phi_kick1_set, time_set]).T


        print("...walkers are set")



    def throw_darts(self, nburn=1000, nsteps=1000):
        """
        Run the emcee sampler.

        Parameters
        ----------

        nburn : int
            Number of burn-in steps

        nsteps : int
            Number of steps to be saved

        """


        # Define sampler
        if self.mpi == True:
            pool = MPIPool()
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            sampler = emcee.EnsembleSampler(nwalkers=self.nwalkers, dim=self.dim, lnpostfn=self.posterior_function, args=[self], pool=pool)

        elif self.threads != 1:
            sampler = emcee.EnsembleSampler(nwalkers=self.nwalkers, dim=self.dim, lnpostfn=self.posterior_function, args=[self], threads=self.threads)
        else:
            sampler = emcee.EnsembleSampler(nwalkers=self.nwalkers, dim=self.dim, lnpostfn=self.posterior_function, args=[self])


        # Burn-in
        print("Starting burn-in...")
        pos,prob,state,binary_data = sampler.run_mcmc(self.p0, N=nburn)
        print("...finished running burn-in")

        # Full run
        print("Starting full run...")
        sampler.reset()
        pos,prob,state,binary_data = sampler.run_mcmc(pos, N=nsteps)
        print("...full run finished")

        self.sampler = sampler
        self.binary_data = binary_data
