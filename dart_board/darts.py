#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An MCMC statistical wrapper on top of a binary population synthesis code.

This wrapper uses "emcee" by Foreman-Mackey et al. (2012): github.com/dfm/emcee

"""

import sys
import numpy as np
import emcee

import time as tm # temporary for testing

from . import priors
from . import posterior
from . import forward_pop_synth
from . import constants as c



class DartBoard():
    """
    The ensemble sampler that searches the initial condition parameter space
    for binaries that fit the input conditions.

    """

    def __init__(self,
                 binary_type,
                 metallicity=0.02,
                 ln_prior_M1=priors.ln_prior_ln_M1,
                 ln_prior_M2=priors.ln_prior_ln_M2,
                 ln_prior_a=priors.ln_prior_ln_a,
                 ln_prior_ecc=priors.ln_prior_ecc,
                 ln_prior_v_kick=priors.ln_prior_v_kick,
                 ln_prior_theta_kick=priors.ln_prior_theta_kick,
                 ln_prior_phi_kick=priors.ln_prior_phi_kick,
                 ln_prior_t=priors.ln_prior_ln_t,
                 ln_prior_pos=None,
                 ln_posterior_function=None,
                 generate_M1=forward_pop_synth.get_M1,
                 generate_M2=forward_pop_synth.get_M2,
                 generate_a=forward_pop_synth.get_a,
                 generate_ecc=forward_pop_synth.get_ecc,
                 generate_v_kick=forward_pop_synth.get_v_kick,
                 generate_theta_kick=forward_pop_synth.get_theta,
                 generate_phi_kick=forward_pop_synth.get_phi,
                 generate_t=forward_pop_synth.get_t,
                 generate_pos=None,
                 ntemps=None,
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

        # Set the functions to generate new values for each parameters
        # This is only needed for forward population synthesis
        self.generate_M1 = generate_M1
        self.generate_M2 = generate_M2
        self.generate_a = generate_a
        self.generate_ecc = generate_ecc
        self.generate_v_kick1 = generate_v_kick
        self.generate_theta_kick1 = generate_theta_kick
        self.generate_phi_kick1 = generate_phi_kick
        self.generate_v_kick2 = generate_v_kick
        self.generate_theta_kick2 = generate_theta_kick
        self.generate_phi_kick2 = generate_phi_kick

        self.generate_pos = None
        self.generate_t = generate_t
        if generate_pos is not None:
            self.generate_pos = generate_pos


        # Set the posterior probability function
        if ln_posterior_function is None:
            self.posterior_function = posterior.ln_posterior
        else:
            self.posterior_function = ln_posterior_function

        # emcee parameters
        self.ntemps = ntemps
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


        # Oservables to match
        self.kwargs = kwargs
        if not self.kwargs == {}:
            for key, value in kwargs.items():
                print("%s = %s" % (key, value))


        # Save observed position
        self.ra_obs = None
        self.dec_obs = None
        for key, value in kwargs.items():
            if key == "ra": self.ra_obs = value
            if key == "dec": self.dec_obs = value


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
        self.chains = []
        self.derived = []
        self.lnprobability = []


    def aim_darts_PT(self):
        """
        Place darts at different temperatures in a viable region of parameter space

        """

        # Set walkers
        print("Setting walkers...")


        # Allocate walkers
        M1_set = np.zeros((self.ntemps, self.nwalkers))
        M2_set = np.zeros((self.ntemps, self.nwalkers))
        a_set = np.zeros((self.ntemps, self.nwalkers))
        ecc_set = np.zeros((self.ntemps, self.nwalkers))
        v_kick1_set = np.zeros((self.ntemps, self.nwalkers))
        theta_kick1_set = np.zeros((self.ntemps, self.nwalkers))
        phi_kick1_set = np.zeros((self.ntemps, self.nwalkers))
        if self.second_SN:
            v_kick2_set = np.zeros((self.ntemps, self.nwalkers))
            theta_kick2_set = np.zeros((self.ntemps, self.nwalkers))
            phi_kick2_set = np.zeros((self.ntemps, self.nwalkers))
        time_set = np.zeros((self.ntemps, self.nwalkers))

        if self.prior_pos is not None:
            # Use first call of sf_history prior to set ra and dec bounds
            tmp = self.prior_pos(0.0, 0.0, 20.0)
            ra_set = np.zeros((self.ntemps, self.nwalkers))
            dec_set = np.zeros((self.ntemps, self.nwalkers))


        # Check if any of these have posteriors with -infinity
        for i in range(self.ntemps):
            for j in range(self.nwalkers):

                # Iterate randomly through initial conditions until a viable parameter set is found
                for k in range(100000):

                    M1 = 30.0 * np.random.uniform(size=1) + 8.0
                    M2 = M1 * (np.random.uniform(size=1))
                    a = 5000.0 * np.random.uniform(size=1) + 20.0
                    ecc = np.random.uniform(size=1)

                    v_kick1 = 300.0 * np.random.uniform(size=1) + 20.0
                    theta_kick1 = np.pi * np.random.uniform(size=1)
                    phi_kick1 = np.pi * np.random.uniform(size=1)
                    if self.second_SN:
                        v_kick2 = 300.0 * np.random.uniform(size=1) + 20.0
                        theta_kick2 = np.pi * np.random.uniform(size=1)
                        phi_kick2 = np.pi * np.random.uniform(size=1)

                    if self.prior_pos is not None:
                        if self.ra_obs is None or self.dec_obs is None:
                            ra = (c.ra_max-c.ra_min) * np.random.uniform(size=1) + c.ra_min
                            dec = (c.dec_max-c.dec_min) * np.random.uniform(size=1) + c.dec_min
                        else:
                            ra = self.ra_obs * (1.0 + np.random.normal(0.0, 0.00001, 1))
                            dec = self.dec_obs * (1.0 + np.random.normal(0.0, 0.00001, 1))

                    time = 40.0 * np.random.uniform(size=1)

                    if self.second_SN:
                        if self.prior_pos is None:
                            x = np.log(M1), np.log(M2), np.log(a), ecc, v_kick1, theta_kick1, phi_kick1, v_kick2, theta_kick2, phi_kick2, np.log(time)
                        else:
                            x = np.log(M1), np.log(M2), np.log(a), ecc, v_kick1, theta_kick1, phi_kick1, v_kick2, theta_kick2, phi_kick2, ra, dec, np.log(time)
                    else:
                        if self.prior_pos is None:
                            x = np.log(M1), np.log(M2), np.log(a), ecc, v_kick1, theta_kick1, phi_kick1, np.log(time)
                        else:
                            x = np.log(M1), np.log(M2), np.log(a), ecc, v_kick1, theta_kick1, phi_kick1, ra, dec, np.log(time)


                    # If the system has a viable posterior probability
                    if self.posterior_function(x, self) > -500.0:

                        M1_set[i,j] = M1
                        M2_set[i,j] = M2
                        a_set[i,j] = a
                        ecc_set[i,j] = ecc
                        v_kick1_set[i,j] = v_kick1
                        theta_kick1_set[i,j] = theta_kick1
                        phi_kick1_set[i,j] = phi_kick1
                        if self.second_SN:
                            v_kick2_set[i,j] = v_kick2
                            theta_kick2_set[i,j] = theta_kick2
                            phi_kick2_set[i,j] = phi_kick2
                        if self.prior_pos is not None:
                            ra_set[i,j] = ra
                            dec_set[i,j] = dec
                        time_set[i,j] = time

                        print("Temp", i, "and Walker", j, "is set. Posterior probability:", self.posterior_function(x, self))

                        # ...then use it as our starting system
                        break

        # Save and return the walker positions
        if self.second_SN:
            if self.prior_pos is None:
                self.p0 = np.array([np.log(M1_set), np.log(M2_set), np.log(a_set), ecc_set, v_kick1_set, theta_kick1_set, \
                                    phi_kick1_set, v_kick2_set, theta_kick2_set, phi_kick2_set, \
                                    np.log(time_set)])
            else:
                self.p0 = np.array([np.log(M1_set), np.log(M2_set), np.log(a_set), ecc_set, v_kick1_set, theta_kick1_set, \
                                    phi_kick1_set, v_kick2_set, theta_kick2_set, phi_kick2_set, \
                                    ra_set, dec_set, np.log(time_set)])
        else:
            if self.prior_pos is None:
                self.p0 = np.array([np.log(M1_set), np.log(M2_set), np.log(a_set), ecc_set, v_kick1_set, theta_kick1_set, \
                                    phi_kick1_set, np.log(time_set)])
            else:
                self.p0 = np.array([np.log(M1_set), np.log(M2_set), np.log(a_set), ecc_set, v_kick1_set, theta_kick1_set, \
                                    phi_kick1_set, ra_set, dec_set, np.log(time_set)])

        # Swap axes for parallel tempered sampler
        self.p0 = np.swapaxes(self.p0, 0, 1)
        self.p0 = np.swapaxes(self.p0, 1, 2)


        print("...walkers are set")

        sys.stdout.flush()


    def aim_darts(self, dart=None):
        """
        Find a viable region of parameter space then create a ball around it.

        """

        # Set walkers
        print("Setting walkers...")


        # Allocate walkers
        M1_set = np.zeros(self.nwalkers)
        M2_set = np.zeros(self.nwalkers)
        a_set = np.zeros(self.nwalkers)
        ecc_set = np.zeros(self.nwalkers)
        v_kick1_set = np.zeros(self.nwalkers)
        theta_kick1_set = np.zeros(self.nwalkers)
        phi_kick1_set = np.zeros(self.nwalkers)
        if self.second_SN:
            v_kick2_set = np.zeros(self.nwalkers)
            theta_kick2_set = np.zeros(self.nwalkers)
            phi_kick2_set = np.zeros(self.nwalkers)
        time_set = np.zeros(self.nwalkers)

        if self.prior_pos is not None:
            # Use first call of sf_history prior to set ra and dec bounds
            tmp = self.prior_pos(0.0, 0.0, 20.0)
            ra_set = np.zeros(self.nwalkers)
            dec_set = np.zeros(self.nwalkers)


        if dart is None:

            # Throw darts around to get a set of starting positions
            for j in range(self.nwalkers):


                # Iterate randomly through initial conditions until a viable parameter set is found
                for i in range(100000):

                    M1 = 30.0 * np.random.uniform(size=1) + 8.0
                    M2 = M1 * (np.random.uniform(size=1))
                    a = 5000.0 * np.random.uniform(size=1) + 20.0
                    ecc = np.random.uniform(size=1)

                    v_kick1 = 300.0 * np.random.uniform(size=1) + 20.0
                    theta_kick1 = np.pi * np.random.uniform(size=1)
                    phi_kick1 = np.pi * np.random.uniform(size=1)
                    if self.second_SN:
                        v_kick2 = 300.0 * np.random.uniform(size=1) + 20.0
                        theta_kick2 = np.pi * np.random.uniform(size=1)
                        phi_kick2 = np.pi * np.random.uniform(size=1)

                    if self.prior_pos is not None:
                        if self.ra_obs is None or self.dec_obs is None:
                            ra = (c.ra_max-c.ra_min) * np.random.uniform(size=1) + c.ra_min
                            dec = (c.dec_max-c.dec_min) * np.random.uniform(size=1) + c.dec_min
                        else:
                            ra = self.ra_obs * (1.0 + np.random.normal(0.0, 0.00001, 1))
                            dec = self.dec_obs * (1.0 + np.random.normal(0.0, 0.00001, 1))

                    time = 40.0 * np.random.uniform(size=1)

                    if self.second_SN:
                        if self.prior_pos is None:
                            x = np.log(M1), np.log(M2), np.log(a), ecc, v_kick1, theta_kick1, phi_kick1, v_kick2, theta_kick2, phi_kick2, np.log(time)
                        else:
                            x = np.log(M1), np.log(M2), np.log(a), ecc, v_kick1, theta_kick1, phi_kick1, v_kick2, theta_kick2, phi_kick2, ra, dec, np.log(time)
                    else:
                        if self.prior_pos is None:
                            x = np.log(M1), np.log(M2), np.log(a), ecc, v_kick1, theta_kick1, phi_kick1, np.log(time)
                        else:
                            x = np.log(M1), np.log(M2), np.log(a), ecc, v_kick1, theta_kick1, phi_kick1, ra, dec, np.log(time)


                    # If the system has a viable posterior probability
                    if self.posterior_function(x, self)[0] > -500.0:

                        M1_set[j] = M1
                        M2_set[j] = M2
                        a_set[j] = a
                        ecc_set[j] = ecc
                        v_kick1_set[j] = v_kick1
                        theta_kick1_set[j] = theta_kick1
                        phi_kick1_set[j] = phi_kick1
                        if self.second_SN:
                            v_kick2_set[j] = v_kick2
                            theta_kick2_set[j] = theta_kick2
                            phi_kick2_set[j] = phi_kick2
                        if self.prior_pos is not None:
                            ra_set[j] = ra
                            dec_set[j] = dec
                        time_set[j] = time

                        print("Walker", j, "is set. Posterior probability:", self.posterior_function(x, self))

                        # ...then use it as our starting system
                        break

        # Save and return the walker positions
        if self.second_SN:
            if self.prior_pos is None:
                self.p0 = np.array([np.log(M1_set), np.log(M2_set), np.log(a_set), ecc_set, v_kick1_set, theta_kick1_set, \
                                    phi_kick1_set, v_kick2_set, theta_kick2_set, phi_kick2_set, \
                                    np.log(time_set)]).T
            else:
                self.p0 = np.array([np.log(M1_set), np.log(M2_set), np.log(a_set), ecc_set, v_kick1_set, theta_kick1_set, \
                                    phi_kick1_set, v_kick2_set, theta_kick2_set, phi_kick2_set, \
                                    ra_set, dec_set, np.log(time_set)]).T
        else:
            if self.prior_pos is None:
                self.p0 = np.array([np.log(M1_set), np.log(M2_set), np.log(a_set), ecc_set, v_kick1_set, theta_kick1_set, \
                                    phi_kick1_set, np.log(time_set)]).T
            else:
                self.p0 = np.array([np.log(M1_set), np.log(M2_set), np.log(a_set), ecc_set, v_kick1_set, theta_kick1_set, \
                                    phi_kick1_set, ra_set, dec_set, np.log(time_set)]).T


        if dart is not None:
            # Load up data from internal dart
            self.p0[0] = dart

            M1_set[0] = np.exp(self.p0[0,0])
            M2_set[0] = np.exp(self.p0[0,1])
            a_set[0] = np.exp(self.p0[0,2])
            ecc_set[0] = self.p0[0,3]
            v_kick1_set[0] = self.p0[0,4]
            theta_kick1_set[0] = self.p0[0,5]
            phi_kick1_set[0] = self.p0[0,6]
            if self.second_SN:
                v_kick2_set[0] = self.p0[0,7]
                theta_kick2_set[0] = self.p0[0,8]
                phi_kick2_set[0] = self.p0[0,9]
                if self.prior_pos is None:
                    time_set[0] = np.exp(self.p0[0,10])
                else:
                    ra_set[0] = self.p0[0,9]
                    dec_set[0] = self.p0[0,10]
                    time_set[0] = np.exp(self.p0[0,11])
            else:
                if self.prior_pos is None:
                    time_set[0] = np.exp(self.p0[0,7])
                else:
                    ra_set[0] = self.p0[0,7]
                    dec_set[0] = self.p0[0,8]
                    time_set[0] = np.exp(self.p0[0,9])




        print("Initial parameter space explored.")
        print("Iterating to do better...")




        # Now, we move to the best position
        ln_posteriors_set = -1.0e4 * np.ones(self.nwalkers)
        for i, p in enumerate(self.p0):
            ln_posteriors_set[i] = self.posterior_function(p, self)[0]
            print(p, ln_posteriors_set[i])
        ln_posterior_best = np.max(ln_posteriors_set)
        idx = np.argmax(ln_posteriors_set)

        # Find values of best data point
        M1 = M1_set[idx]
        M2 = M2_set[idx]
        a = a_set[idx]
        ecc = ecc_set[idx]
        v_kick1 = v_kick1_set[idx]
        theta_kick1 = theta_kick1_set[idx]
        phi_kick1 = phi_kick1_set[idx]
        if self.second_SN:
            v_kick2 = v_kick2_set[idx]
            theta_kick2 = theta_kick2_set[idx]
            phi_kick2 = phi_kick2_set[idx]
        if self.prior_pos is not None:
            ra = ra_set[idx]
            dec = dec_set[idx]
        time = time_set[idx]


        # Iterate around data point until solution is stable
        C = 0.002


        lp_prev = -1.0e4 * np.ones(20)


        while 1:

            ln_posterior_best = np.max(ln_posteriors_set)
            idx = np.argmax(ln_posteriors_set)

            # Shift record of previous posterior probabilities
            for i in range(19):
                lp_prev[i] = lp_prev[i+1]
            lp_prev[19] = ln_posterior_best

            # FOR TESTING
            if lp_prev[17] > -100.0: break
            if(abs(lp_prev[0] - lp_prev[19]) < 0.2): break

            print("ln_posterior:", idx, ln_posterior_best, lp_prev[0])


            M1 = M1_set[idx]
            M2 = M2_set[idx]
            a = a_set[idx]
            ecc = ecc_set[idx]
            v_kick1 = v_kick1_set[idx]
            theta_kick1 = theta_kick1_set[idx]
            phi_kick1 = phi_kick1_set[idx]
            if self.second_SN:
                v_kick2 = v_kick2_set[idx]
                theta_kick2 = theta_kick2_set[idx]
                phi_kick2 = phi_kick2_set[idx]
            if self.prior_pos is not None:
                ra = ra_set[idx]
                dec = dec_set[idx]
            time = time_set[idx]





            for i in range(self.nwalkers):

                if i == idx: continue

                if self.second_SN:
                    if self.prior_pos is None:
                        p = np.log(M1_set[i]), np.log(M2_set[i]), np.log(a_set[i]), ecc_set[i], \
                                v_kick1_set[i], theta_kick1_set[i], phi_kick1_set[i], \
                                v_kick2_set[i], theta_kick2_set[i], phi_kick2_set[i], \
                                np.log(time_set[i])
                    else:
                        p = np.log(M1_set[i]), np.log(M2_set[i]), np.log(a_set[i]), ecc_set[i], \
                                v_kick1_set[i], theta_kick1_set[i], phi_kick1_set[i], \
                                v_kick2_set[i], theta_kick2_set[i], phi_kick2_set[i], \
                                ra_set[i], dec_set[i], np.log(time_set[i])

                else:
                    if self.prior_pos is None:
                        p = np.log(M1_set[i]), np.log(M2_set[i]), np.log(a_set[i]), ecc_set[i], \
                                v_kick1_set[i], theta_kick1_set[i], phi_kick1_set[i], \
                                np.log(time_set[i])
                    else:
                        p = np.log(M1_set[i]), np.log(M2_set[i]), np.log(a_set[i]), ecc_set[i], \
                                v_kick1_set[i], theta_kick1_set[i], phi_kick1_set[i], \
                                ra_set[i], dec_set[i], np.log(time_set[i])


                ln_posterior = self.posterior_function(p, self)[0]


                k = 0
                while k == 0 or ln_posterior + 4.0 < ln_posterior_best or np.isinf(ln_posterior):

                    k = k + 1

                    # Binary parameters
                    M1_set[i] = M1*(1.0 + np.random.normal(0.0, C, 1))
                    M2_set[i] = M2*(1.0 + np.random.normal(0.0, C, 1))
                    ecc_set[i] = ecc*(1.0 + np.random.normal(0.0, C, 1))
                    a_set[i] = a*(1.0 + np.random.normal(0.0, C, 1))

                    # SN kick perameters
                    v_kick1_set[i] = v_kick1*(1.0 + np.random.normal(0.0, C, 1))
                    theta_kick1_set[i] = theta_kick1*(1.0 + np.random.normal(0.0, C, 1))
                    phi_kick1_set[i] = phi_kick1*(1.0 + np.random.normal(0.0, C, 1))
                    if self.second_SN:
                        v_kick2_set[i] = v_kick2*(1.0 + np.random.normal(0.0, C, 1))
                        theta_kick2_set[i] = theta_kick2*(1.0 + np.random.normal(0.0, C, 1))
                        phi_kick2_set[i] = phi_kick2*(1.0 + np.random.normal(0.0, C, 1))

                    # Position
                    if self.prior_pos is not None:
                        ra_set[i] = ra*(1.0 + np.random.normal(0.0, 0.001*C, 1))
                        dec_set[i] = dec*(1.0 + np.random.normal(0.0, 0.001*C, 1))


                    # Birth time
                    time_set[i] = time*(1.0 + np.random.normal(0.0, C, 1))


                    if self.second_SN:
                        if self.prior_pos is None:
                            p = np.log(M1_set[i]), np.log(M2_set[i]), np.log(a_set[i]), ecc_set[i], \
                                    v_kick1_set[i], theta_kick1_set[i], phi_kick1_set[i], \
                                    v_kick2_set[i], theta_kick2_set[i], phi_kick2_set[i], \
                                    np.log(time_set[i])
                        else:
                            p = np.log(M1_set[i]), np.log(M2_set[i]), np.log(a_set[i]), ecc_set[i], \
                                    v_kick1_set[i], theta_kick1_set[i], phi_kick1_set[i], \
                                    v_kick2_set[i], theta_kick2_set[i], phi_kick2_set[i], \
                                    ra_set[i], dec_set[i], np.log(time_set[i])

                    else:
                        if self.prior_pos is None:
                            p = np.log(M1_set[i]), np.log(M2_set[i]), np.log(a_set[i]), ecc_set[i], \
                                    v_kick1_set[i], theta_kick1_set[i], phi_kick1_set[i], \
                                    np.log(time_set[i])
                        else:
                            p = np.log(M1_set[i]), np.log(M2_set[i]), np.log(a_set[i]), ecc_set[i], \
                                    v_kick1_set[i], theta_kick1_set[i], phi_kick1_set[i], \
                                    ra_set[i], dec_set[i], np.log(time_set[i])


                    ln_posterior = self.posterior_function(p, self)[0]


                ln_posteriors_set[i] = ln_posterior



        # Save and return the walker positions
        if self.second_SN:
            if self.prior_pos is None:
                self.p0 = np.array([np.log(M1_set), np.log(M2_set), np.log(a_set), ecc_set, v_kick1_set, theta_kick1_set, \
                                    phi_kick1_set, v_kick2_set, theta_kick2_set, phi_kick2_set, \
                                    np.log(time_set)]).T
            else:
                self.p0 = np.array([np.log(M1_set), np.log(M2_set), np.log(a_set), ecc_set, v_kick1_set, theta_kick1_set, \
                                    phi_kick1_set, v_kick2_set, theta_kick2_set, phi_kick2_set, \
                                    ra_set, dec_set, np.log(time_set)]).T
        else:
            if self.prior_pos is None:
                self.p0 = np.array([np.log(M1_set), np.log(M2_set), np.log(a_set), ecc_set, v_kick1_set, theta_kick1_set, \
                                    phi_kick1_set, np.log(time_set)]).T
            else:
                self.p0 = np.array([np.log(M1_set), np.log(M2_set), np.log(a_set), ecc_set, v_kick1_set, theta_kick1_set, \
                                    phi_kick1_set, ra_set, dec_set, np.log(time_set)]).T


        print("...walkers are set")

        sys.stdout.flush()




    def throw_darts(self, nburn=1000, nsteps=1000, method='emcee'):
        """
        Run the sampler.

        Parameters
        ----------

        nburn : int
            Number of burn-in steps

        nsteps : int
            Number of steps to be saved

        """


        if method == 'emcee':

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


            # Save only every 10th sample
            self.chains = sampler.chain[:,::10,:]
            self.derived = np.swapaxes(np.array(sampler.blobs), 0, 1)[:,::10,0,:]
            self.lnprobability = sampler.lnprobability[:,::10]

            self.sampler = sampler


        elif method == 'emcee_PT':

            # THIS DOES NOT YET WORK #

            # Define sampler
            if self.mpi == True:
                pool = MPIPool()
                if not pool.is_master():
                    pool.wait()
                    sys.exit(0)
                sampler = emcee.PTSampler(ntemps=self.ntemps, nwalkers=self.nwalkers, dim=self.dim,
                                          logl=posterior.ln_likelihood, logp=priors.ln_prior,
                                          loglargs=(self,), logpargs=(self,), pool=pool)

            elif self.threads != 1:
                sampler = emcee.PTSampler(ntemps=self.ntemps, nwalkers=self.nwalkers, dim=self.dim,
                                          logl=posterior.ln_likelihood, logp=priors.ln_prior,
                                          loglargs=(self,), logpargs=(self,), threads=self.threads)
            else:
                sampler = emcee.PTSampler(ntemps=self.ntemps, nwalkers=self.nwalkers, dim=self.dim,
                                          logl=posterior.ln_likelihood, logp=priors.ln_prior,
                                          loglargs=(self,), logpargs=(self,))



            # Burn-in
            print("Starting burn-in...")
            for pos,prob,state in sampler.sample(self.p0, iterations=nburn):
                pass
            print("...finished running burn-in")

            # Full run
            print("Starting full run...")
            sampler.reset()
            for pos,prob,state in sampler.sample(pos, iterations=nsteps):
                pass
            print("...full run finished")



            # Save only every 10th sample
            self.chains = sampler.chain[:,::10,:]
            # self.derived = np.swapaxes(np.array(sampler.blobs), 0, 1)[:,::10,0,:]
            self.lnprobability = sampler.lnprobability[:,::10]

            self.sampler = sampler



        elif method == 'nestle':

            print("nestle nested sampling is not yet implemented.")







        else:
            print("Your chosen method is not supported by dart_board.")



    def scatter_darts(self, num_darts=100000):
        """
        Rather than use the MCMC sampler, run a forward population synthesis analysis.

        Parameters
        ----------
        num_darts : int
            Number of darts to throw

        """

        # Generate the population
        M1, M2, orbital_period, ecc, v_kick1, theta_kick1, phi_kick1, v_kick2, theta_kick2, \
                phi_kick2, ra, dec, t_b = forward_pop_synth.generate_population(self, num_darts)

        # Get ln of parameters
        ln_M1 = np.log(M1)
        ln_M2 = np.log(M2)
        ln_a = np.log(posterior.P_to_A(M1, M2, orbital_period))
        ln_t_b = np.log(t_b)

        # Now, we want to zero the outputs
        self.chains = np.zeros((num_darts, 13))
        self.derived = np.zeros((num_darts, 9))
        self.likelihood = np.zeros(num_darts, dtype=float)
        success = np.zeros(num_darts, dtype=bool)


        # Run binary evolution - must run one at a time
        for i in range(num_darts):
            output = self.evolve_binary(1, M1[i], M2[i], orbital_period[i], ecc[i],
                                        v_kick1[i], theta_kick1[i], phi_kick1[i],
                                        v_kick2[i], theta_kick2[i], phi_kick2[i],
                                        t_b[i], self.metallicity, False)

            if posterior.check_output(output, self.binary_type):

                # For likelihood function, we need to input specific number of parameters
                if self.second_SN:
                    if self.prior_pos is None:
                        x_i = ln_M1[i], ln_M2[i], ln_a[i], ecc[i], v_kick1[i], theta_kick1[i], phi_kick1[i], v_kick2[i], theta_kick2[i], phi_kick2[i], ln_t_b[i]
                    else:
                        x_i = ln_M1[i], ln_M2[i], ln_a[i], ecc[i], v_kick1[i], theta_kick1[i], phi_kick1[i], v_kick2[i], theta_kick2[i], phi_kick2[i], ra[i], dec[i], ln_t_b[i]
                else:
                    if self.prior_pos is None:
                        x_i = ln_M1[i], ln_M2[i], ln_a[i], ecc[i], v_kick1[i], theta_kick1[i], phi_kick1[i], ln_t_b[i]
                    else:
                        x_i = ln_M1[i], ln_M2[i], ln_a[i], ecc[i], v_kick1[i], theta_kick1[i], phi_kick1[i], ra[i], dec[i], ln_t_b[i]

                self.likelihood[i] = posterior.posterior_properties(x_i, output, self)

                # Only store if it likelihood is finite
                if(np.isinf(self.likelihood[i])): continue



                x_i = M1[i], M2[i], orbital_period[i], ecc[i], v_kick1[i], theta_kick1[i], \
                        phi_kick1[i], v_kick2[i], theta_kick2[i], phi_kick2[i], ra[i], dec[i], t_b[i]

                # Save chains and derived
                self.chains[i] = np.array([x_i])
                self.derived[i] = np.array([output])


                success[i] = True

        # Now, let's only return successful values in binary_x_i, binary_data
        self.chains = self.chains[success]
        self.derived = self.derived[success]
        self.likelihood = self.likelihood[success]
