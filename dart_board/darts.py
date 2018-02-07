#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
An MCMC statistical wrapper on top of a binary population synthesis code.

This wrapper uses "emcee" by Foreman-Mackey et al. (2012): github.com/dfm/emcee

"""

import sys
import numpy as np
import emcee
import copy

import time as tm # temporary for testing

from . import priors
from . import posterior
from . import forward_pop_synth
from . import constants as c



class DartBoard():
    """
    A collection of prior probabilities and likelihoods that
    define either a class of binaries or a specific binary.

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
                 ln_prior_z=None,
                 ln_posterior_function=None,
                 generate_M1=forward_pop_synth.get_M1,
                 generate_M2=forward_pop_synth.get_M2,
                 generate_a=forward_pop_synth.get_a,
                 generate_ecc=forward_pop_synth.get_ecc,
                 generate_v_kick=forward_pop_synth.get_v_kick,
                 generate_theta_kick=forward_pop_synth.get_theta,
                 generate_phi_kick=forward_pop_synth.get_phi,
                 generate_t=forward_pop_synth.get_t,
                 generate_z=forward_pop_synth.get_z,
                 generate_pos=None,
                 ntemps=None,
                 nwalkers=80,
                 threads=1,
                 mpi=False,
                 evolve_binary=None,
                 thin=100,
                 prior_kwargs={},
                 system_kwargs={},
                 model_kwargs={}):

        """

        Args:
            binary_type : string, type of binary selected for. Available options
                can be found in dart_board/posterior.py.
            metallicity : float (default: 0.02), metallicity for binaries.
            ln_prior_M1 : function (default: priors.ln_prior_ln_M1), prior
                probability on primary mass.
            ln_prior_M2 : function (default: priors.ln_prior_ln_M2), prior
                probability on secondary mass.
            ln_prior_a : function (default: priors.ln_prior_ln_a), prior
                probability on orbital separation.
            ln_prior_ecc : function (default: priors.ln_prior_ecc), prior
                probability on eccentricity.
            ln_prior_v_kick : function (default: priors.ln_prior_v_kick), prior
                probability on the supernova kick magnitude.
            ln_prior_theta_kick : function (default: priors.ln_prior_theta_kick),
                prior probability on the kick polar angle.
            ln_prior_phi_kick : function (default: priors.ln_prior_phi_kick),
                prior probability on the kick azimuthal angle.
            ln_prior_t : function (default: priors.ln_prior_ln_t), prior
                probability on the birth time.
            ln_prior_pos : function (default: None), when provided, birth
                coordinates are model parameters. This function is the prior
                probability on the birth location. Typically this can be
                provided in the form of a star formation history map.
            ln_prior_z : function (default: None), when provided, metallicity
                is a model parameter. This function is the prior probability on
                the metallicity.
            ln_posterior_function : function (default: None), the posterior
                function evaluated by the sampler.
            generate_M1 : function (default: forward_pop_synth.get_M1), the
                function providing randomly drawn values of the primary mass.
            generate_M2 : function (default: forward_pop_synth.get_M2), the
                function providing randomly drawn values of the secondary mass.
            generate_a : function (default: forward_pop_synth.get_a), the
                function providing randomly drawn values of the orbital separation.
            generate_ecc : function (default: forward_pop_synth.get_ecc), the
                function providing randomly drawn values of the eccentricity.
            generate_v_kick : function (default: forward_pop_synth.get_v_kick),
                the function providing randomly drawn values of the supernova
                kick magnitude.
            generate_theta_kick : function (default: forward_pop_synth.get_theta),
                the function providing randomly drawn values of the kick polar angle.
            generate_phi_kick : function (default: forward_pop_synth.get_phi),
                the function providing randomly drawn values of the kick
                azimuthal angle.
            generate_t : function (default: forward_pop_synth.get_t), the
                function providing randomly drawn values of the birth time.
            generate_z : function (default: forward_pop_synth.get_z), the
                function providing randomly drawn values of the birth metallicity.
            generate_pos: function (default: None), the function providing
                randomly drawn birth position coordinates.
            ntemps : int (default: None), when provided, the number of temperatures
                for emcee's parallel tempering algorithm.
            nwalkers : int (default: 80), the number of walkers for emcee's
                ensemble sampler algorithm.
            threads : int (default: 1), the number of threads to use for
                parallel processing using multi-threading.
            mpi : bool (default: False), when true, use mpi for parallel processing.
            evolve_binary : function (default: None), the binary evolution
                function. THIS INPUT IS REQUIRED FOR DART_BOARD TO RUN.
            thin : int (default: 100), thin posterior samples by this number.
            prior_kwargs : kwargs (default: {}), arguments for altering the
                prior probabilities (e.g., maximum orbital separation, IMF
                index, etc.). See constants.py for options.
            system_kwargs : kwargs (default: {}), arguments for defining
                observational constraints for the system (e.g., P_orb and
                P_orb_err, ecc_max, etc.).
            model_kwargs : kwargs (default: {}), arguments for altering the
                binary evolution physics (e.g., common envelope efficiency).
                See pyBSE/pybse/bse_wrapper.py for available options.

        """

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

        self.model_metallicity = False
        if ln_prior_z is not None:
            self.prior_z = ln_prior_z
            self.model_metallicity = True

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
        self.generate_z = generate_z
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
        self.first_SN = False
        if "NS" in binary_type or "BH" in binary_type or "HMXB" in binary_type:
            self.first_SN = True

        self.second_SN = False
        if np.any(binary_type == np.array(["BHBH", "NSNS", "BHNS"])):
            self.second_SN = True


        # Oservables to match
        self.system_kwargs = system_kwargs
        if not self.system_kwargs == {}:
            for key, value in system_kwargs.items():
                print("%s = %s" % (key, value))


        # Save observed position
        self.ra_obs = None
        self.dec_obs = None
        for key, value in system_kwargs.items():
            if key == "ra": self.ra_obs = value
            if key == "dec": self.dec_obs = value

        # Stellar evolution options for BSE
        self.model_kwargs = model_kwargs

        # Options for prior probabilities
        self.prior_kwargs = prior_kwargs

        # Determine the number of dimensions
        self.dim = 4  # M1, M2, a, ecc
        if self.first_SN: self.dim += 3  # First SN parameters
        if self.second_SN: self.dim += 3  # Second SN parameters
        if self.prior_pos is not None: self.dim += 2  # RA and Dec
        if self.model_metallicity: self.dim += 1  # for modeling the metallicity
        self.dim += 1  # Birth time


        # Saved data
        self.thin = thin
        self.sampler = []
        self.chains = []
        self.derived = []
        self.lnprobability = []


    def iterate_to_initialize(self, N_iterations=10000):

        lp_best = -100000

        # Iterate randomly through initial conditions until a viable parameter set is found
        for i in range(N_iterations):

            if "WD" in self.binary_type:
                M1 = 3.0 * np.random.uniform(size=1) + 8.0
            else:
                M1 = 30.0 * np.random.uniform(size=1) + 8.0
            M2 = M1 * (np.random.uniform(size=1))
            a = 5000.0 * np.random.uniform(size=1) + 20.0
            ecc = np.random.uniform(size=1)

            if self.first_SN:
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

            if self.model_metallicity: z = np.exp(np.random.normal(np.log(0.02), 0.001, 1))

            if "WD" in self.binary_type:
                time = 1.4e4 * np.random.uniform(size=1)
            else:
                time = 40.0 * np.random.uniform(size=1)


            # Create tuple of model parameters
            x = np.log(M1), np.log(M2), np.log(a), ecc
            if self.first_SN: x += v_kick1, theta_kick1, phi_kick1
            if self.second_SN: x += v_kick2, theta_kick2, phi_kick2
            if self.prior_pos is not None: x += ra, dec
            if self.model_metallicity: x+= (np.log(z),)
            x += (np.log(time),)


            # Calculate the posterior probability for x
            lp = self.posterior_function(x, self)


            # Keep the best set of model parameter
            if lp > lp_best:
                lp_best = lp
                x_best = copy.copy(x)


        return x_best


    def aim_darts_PT(self):
        """
        Place darts at different temperatures in a viable region of parameter space.

        """

        # Set walkers
        print("Setting walkers...")


        # Iterate to find position for focusing walkers
        x_best = iterate_to_initialize(self, N_iterations=10000)


        # Allocate walkers
        M1_set = np.zeros((self.ntemps, self.nwalkers))
        M2_set = np.zeros((self.ntemps, self.nwalkers))
        a_set = np.zeros((self.ntemps, self.nwalkers))
        ecc_set = np.zeros((self.ntemps, self.nwalkers))
        if self.first_SN:
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

        if self.model_metallicity: z_set = np.zeros((self.ntemps, self.nwalkers))


        # Check if any of these have posteriors with -infinity
        for i in range(self.ntemps):
            for j in range(self.nwalkers):


                lp = -100000

                while lp < lp_best - 5.0:


                    # Create x_new which holds new set of model parameters
                    x = []
                    for x_i in x_best:
                        x += (x_i*np.random.normal(loc=1.0, scale=0.001, size=1)[0], )


                    # Calculate the posterior probability for x
                    lp = self.posterior_function(x, self)


                # Save model parameters to variables
                ln_M1, ln_M2, ln_a, ecc = x[0:4]
                x = x[4:]
                if dart.first_SN:
                    v_kick1, theta_kick1, phi_kick1 = x[0:3]
                    x = x[3:]
                if dart.second_SN:
                    v_kick2, theta_kick2, phi_kick2 = x[0:3]
                    x = x[3:]
                if dart.prior_pos is not None:
                    ra_b, dec_b = x[0:2]
                    x = x[2:]
                if dart.model_metallicity:
                    ln_z = x[0]
                    z = np.exp(ln_z)
                    x = x[1:]
                else:
                    z = dart.metallicity
                ln_t_b = x[0]



                M1_set[i,j] = np.exp(ln_M1)
                M2_set[i,j] = np.exp(ln_M2)
                a_set[i,j] = np.exp(ln_a)
                ecc_set[i,j] = ecc
                if self.first_SN:
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
                if self.model_metallicity: z_set[i,j] = z
                time_set[i,j] = np.exp(ln_t_b)


        # Save and return the walker positions
        self.p0 = np.array([np.log(M1_set), np.log(M2_set), np.log(a_set), ecc_set])
        if self.first_SN: self.p0 = np.vstack((self.p0, v_kick1_set[np.newaxis,:,:], theta_kick1_set[np.newaxis,:,:], phi_kick1_set[np.newaxis,:,:]))
        if self.second_SN: self.p0 = np.vstack((self.p0, v_kick2_set[np.newaxis,:,:], theta_kick2_set[np.newaxis,:,:], phi_kick2_set[np.newaxis,:,:]))
        if self.prior_pos is not None: self.p0 = np.vstack((self.p0, ra_set[np.newaxis,:,:], dec_set[np.newaxis,:,:]))
        if self.model_metallicity: self.p0 = np.vstack((self.p0, np.log(z_set[np.newaxis,:,:])))
        self.p0 = np.vstack((self.p0, np.log(time_set[np.newaxis,:,:])))


        # Swap axes for parallel tempered sampler
        self.p0 = np.swapaxes(self.p0, 0, 1)
        self.p0 = np.swapaxes(self.p0, 1, 2)


        print("...walkers are set.")

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
        if self.first_SN:
            v_kick1_set = np.zeros(self.nwalkers)
            theta_kick1_set = np.zeros(self.nwalkers)
            phi_kick1_set = np.zeros(self.nwalkers)
        if self.second_SN:
            v_kick2_set = np.zeros(self.nwalkers)
            theta_kick2_set = np.zeros(self.nwalkers)
            phi_kick2_set = np.zeros(self.nwalkers)
        if self.model_metallicity: z_set = np.zeros(self.nwalkers)
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

                    if self.binary_type == "ELMWD" or self.binary_type == "ELMWD_WD" or self.binary_type == "WDWD":
                        M1 = 3.0 * np.random.uniform(size=1) + 8.0
                    else:
                        M1 = 30.0 * np.random.uniform(size=1) + 8.0
                    M2 = M1 * (np.random.uniform(size=1))
                    a = 5000.0 * np.random.uniform(size=1) + 20.0
                    ecc = np.random.uniform(size=1)

                    if self.first_SN:
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

                    if self.model_metallicity: z = np.exp(np.random.normal(np.log(0.02), 0.001, 1))

                    if self.binary_type == "ELMWD" or self.binary_type == "ELMWD_WD" or self.binary_type == "WDWD":
                        time = 1.4e4 * np.random.uniform(size=1)
                    else:
                        time = 40.0 * np.random.uniform(size=1)


                    # Create tuple of model parameters
                    x = np.log(M1), np.log(M2), np.log(a), ecc
                    if self.first_SN: x += v_kick1, theta_kick1, phi_kick1
                    if self.second_SN: x += v_kick2, theta_kick2, phi_kick2
                    if self.prior_pos is not None: x += ra, dec
                    if self.model_metallicity: x+= (np.log(z),)
                    x += (np.log(time),)



                    # If the system has a viable posterior probability
                    if self.posterior_function(x, self)[0] > -500.0:

                        M1_set[j] = M1
                        M2_set[j] = M2
                        a_set[j] = a
                        ecc_set[j] = ecc
                        if self.first_SN:
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
                        if self.model_metallicity: z_set[j] = z
                        time_set[j] = time

                        print("Walker", j, "is set. Posterior probability:", self.posterior_function(x, self))

                        # ...then use it as our starting system
                        break

        # Save and return the walker positions
        self.p0 = np.array([np.log(M1_set), np.log(M2_set), np.log(a_set), ecc_set])
        if self.first_SN: self.p0 = np.vstack((self.p0, v_kick1_set, theta_kick1_set, phi_kick1_set))
        if self.second_SN: self.p0 = np.vstack((self.p0, v_kick2_set, theta_kick2_set, phi_kick2_set))
        if self.prior_pos is not None: self.p0 = np.vstack((self.p0, ra_set, dec_set))
        if self.model_metallicity: self.p0 = np.vstack((self.p0, np.log(z_set)))
        self.p0 = np.vstack((self.p0, np.log(time_set))).T

        # if self.second_SN:
        #     if self.prior_pos is None:
        #         self.p0 = np.array([np.log(M1_set), np.log(M2_set), np.log(a_set), ecc_set, v_kick1_set, theta_kick1_set, \
        #                             phi_kick1_set, v_kick2_set, theta_kick2_set, phi_kick2_set, \
        #                             np.log(time_set)]).T
        #     else:
        #         self.p0 = np.array([np.log(M1_set), np.log(M2_set), np.log(a_set), ecc_set, v_kick1_set, theta_kick1_set, \
        #                             phi_kick1_set, v_kick2_set, theta_kick2_set, phi_kick2_set, \
        #                             ra_set, dec_set, np.log(time_set)]).T
        # else:
        #     if self.prior_pos is None:
        #         self.p0 = np.array([np.log(M1_set), np.log(M2_set), np.log(a_set), ecc_set, v_kick1_set, theta_kick1_set, \
        #                             phi_kick1_set, np.log(time_set)]).T
        #     else:
        #         self.p0 = np.array([np.log(M1_set), np.log(M2_set), np.log(a_set), ecc_set, v_kick1_set, theta_kick1_set, \
        #                             phi_kick1_set, ra_set, dec_set, np.log(time_set)]).T


        if dart is not None:
            # Load up data from internal dart
            self.p0[0] = dart

            M1_set[0] = np.exp(self.p0[0,0])
            M2_set[0] = np.exp(self.p0[0,1])
            a_set[0] = np.exp(self.p0[0,2])
            ecc_set[0] = self.p0[0,3]
            i = 4
            if self.first_SN:
                v_kick1_set[0] = self.p0[0,i]
                theta_kick1_set[0] = self.p0[0,i+1]
                phi_kick1_set[0] = self.p0[0,i+2]
                i += 3
            if self.second_SN:
                v_kick2_set[0] = self.p0[0,i]
                theta_kick2_set[0] = self.p0[0,i+1]
                phi_kick2_set[0] = self.p0[0,i+2]
                i += 3
            if self.prior_pos is not None:
                ra_set[0] = self.p0[0,i]
                dec_set[0] = self.p0[0,i+1]
                i += 2
            if self.model_metallicity:
                z_set[0] = self.p0[0,i]
                i += 1
            time_set[0] = np.exp(self.p0[0,i])




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
        if self.first_SN:
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
        if self.model_metallicity: z = z_set[idx]
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
            if self.first_SN:
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
            if self.model_metallicity: z = z_set[idx]
            time = time_set[idx]





            for i in range(self.nwalkers):

                if i == idx: continue


                # Create tuple of model parameters
                p = np.log(M1_set[i]), np.log(M2_set[i]), np.log(a_set[i]), ecc_set[i]
                if self.first_SN: p += v_kick1_set[i], theta_kick1_set[i], phi_kick1_set[i]
                if self.second_SN: p += v_kick2_set[i], theta_kick2_set[i], phi_kick2_set[i]
                if self.prior_pos is not None: p += ra_set[i], dec_set[i]
                if self.model_metallicity: p += (np.log(z_set[i]),)
                p += (np.log(time_set[i]),)



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
                    if self.first_SN:
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

                    # Metallicity
                    if self.model_metallicity: z_set[i] = np.exp(np.random.normal(np.log(0.02), 0.001*C, 1))

                    # Birth time
                    time_set[i] = time*(1.0 + np.random.normal(0.0, C, 1))


                    # Create tuple of model parameters
                    p = np.log(M1_set[i]), np.log(M2_set[i]), np.log(a_set[i]), ecc_set[i]
                    if self.first_SN: p += v_kick1_set[i], theta_kick1_set[i], phi_kick1_set[i]
                    if self.second_SN: p += v_kick2_set[i], theta_kick2_set[i], phi_kick2_set[i]
                    if self.prior_pos is not None: p += ra_set[i], dec_set[i]
                    if self.model_metallicity: p += (np.log(z_set[i]),)
                    p += (np.log(time_set[i]),)


                    # Calculate posterior probability of tested data point
                    ln_posterior = self.posterior_function(p, self)[0]


                ln_posteriors_set[i] = ln_posterior



        # Save and return the walker positions
        self.p0 = np.array([np.log(M1_set), np.log(M2_set), np.log(a_set), ecc_set])
        if self.first_SN: self.p0 = np.vstack((self.p0, v_kick1_set, theta_kick1_set, phi_kick1_set))
        if self.second_SN: self.p0 = np.vstack((self.p0, v_kick2_set, theta_kick2_set, phi_kick2_set))
        if self.prior_pos is not None: self.p0 = np.vstack((self.p0, ra_set, dec_set))
        if self.model_metallicity: self.p0 = np.vstack((self.p0, np.log(z_set)))
        self.p0 = np.vstack((self.p0, np.log(time_set))).T


        print("...walkers are set")

        sys.stdout.flush()




    def throw_darts(self, nburn=1000, nsteps=1000, method='emcee'):
        """
        Run the sampler.

        Args:
            nburn : int (default: 1000), number of burn-in steps.
            nsteps : int (default: 1000), number of steps to be saved.
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


            # Save only every 100th sample
            self.chains = sampler.chain[:,::self.thin,:]
            self.derived = np.swapaxes(np.array(sampler.blobs), 0, 1)[:,::self.thin,0,:]
            self.lnprobability = sampler.lnprobability[:,::self.thin]

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
            for pos,prob,state in sampler.sample(pos, iterations=nsteps, thin=self.thin):
                pass
            print("...full run finished")



            self.chains = sampler.chain
            # self.derived = np.swapaxes(np.array(sampler.blobs), 0, 1)
            self.lnprobability = sampler.lnprobability
            self.sampler = sampler



        elif method == 'nestle':

            print("Nested sampling is not yet implemented.")



        else:
            print("Your chosen method is not supported by dart_board.")



    def scatter_darts(self, num_darts=-1, seconds=-1, batch_size=1000, output_frequency=100000):
        """
        Rather than use the MCMC sampler, run a forward population synthesis analysis. results
        are saved as objects `chains`, `derived`, and `likelihood`, as objects
        within the DartBoard class.

        Args:
            num_darts : int (default: -1), number of darts to throw. Either this
                or `seconds` options must be positive for run.
            seconds : int (default: -1), number of seconds to run.
            batch_size : int (default: 1000), run simulation in batches of
                batch_size. This is to save memory. Larger values will reduce
                overhead while smaller values will not overshoot limitations
                imposed by `num_darts` or `seconds`.
            output_frequency : int (default: 100000), frequency of command line
                output for progress updates.
        """

        if num_darts == -1 and seconds == -1:
            print("You must provide either the number of systems you would like to run")
            print("or the number of seconds you would like to run dart_board for.")
            sys.exit(-1)

        # We will save the likelihoods into self.likelihood
        self.likelihood = []

        # Initialize time and counter trackers
        start_time = tm.time()
        num_ran = 0

        # Set the metallicity
        z = self.metallicity

        # Run for as long as constraints allow
        while(num_ran < num_darts or (tm.time()-start_time) < seconds):

            # Run in batches, or whatever is left over
            if num_darts == -1:
                batch_size = batch_size
            else:
                batch_size = np.min([batch_size, num_darts - num_ran])


            # Generate the population
            if self.ra_obs is None or self.dec_obs is None:
                M1, M2, orbital_period, ecc, v_kick1, theta_kick1, phi_kick1, v_kick2, theta_kick2, \
                        phi_kick2, ra, dec, t_b = forward_pop_synth.generate_population(self, batch_size)
            else:
                M1, M2, orbital_period, ecc, v_kick1, theta_kick1, phi_kick1, v_kick2, theta_kick2, \
                        phi_kick2, ra, dec, t_b = forward_pop_synth.generate_population(self, batch_size, \
                        ra_in=self.ra_obs, dec_in=self.dec_obs)

            # Override the previously set metallicity if we are including metallicity models
            if self.model_metallicity: ln_z = np.log(self.generate_z(t_b, batch_size))


            # Get ln of parameters
            ln_M1 = np.log(M1)
            ln_M2 = np.log(M2)
            ln_a = np.log(posterior.P_to_A(M1, M2, orbital_period))
            ln_t_b = np.log(t_b)


            # Now, we want to zero the outputs
            chains = np.zeros((batch_size, 14))
            derived = np.zeros((batch_size, 9))
            likelihood = np.zeros(batch_size, dtype=float)
            success = np.zeros(batch_size, dtype=bool)

            # Run binary evolution - must run one at a time
            for i in range(batch_size):

                if self.model_metallicity: z = np.exp(ln_z[i])

                output = self.evolve_binary(M1[i], M2[i], orbital_period[i], ecc[i],
                                            v_kick1[i], theta_kick1[i], phi_kick1[i],
                                            v_kick2[i], theta_kick2[i], phi_kick2[i],
                                            t_b[i], z, False, **self.model_kwargs)

                # Increment counter to keep track of the number of binaries run
                num_ran += 1

                if num_ran%output_frequency == 0:
                    print ("Number run:", num_ran, "in", tm.time() - start_time, "seconds. Found", len(self.chains), "successful binaries.")


                if posterior.check_output(output, self.binary_type):



                    # Create tuple of model parameters
                    x_i = ln_M1[i], ln_M2[i], ln_a[i], ecc[i]
                    if self.first_SN: x_i += v_kick1[i], theta_kick1[i], phi_kick1[i]
                    if self.second_SN: x_i += v_kick2[i], theta_kick2[i], phi_kick2[i]
                    if self.prior_pos is not None: x_i += ra[i], dec[i]
                    if self.model_metallicity: x_i += (ln_z[i],)
                    x_i += (ln_t_b[i],)

                    # Calculate the likelihood function
                    likelihood[i] = posterior.posterior_properties(x_i, output, self)


                    # Only store if it likelihood is finite
                    if(np.isinf(likelihood[i])): continue



                    x_i = M1[i], M2[i], orbital_period[i], ecc[i], v_kick1[i], theta_kick1[i], \
                            phi_kick1[i], v_kick2[i], theta_kick2[i], phi_kick2[i], ra[i], dec[i], \
                            t_b[i], z

                    # Save chains and derived
                    chains[i] = np.array([x_i])
                    derived[i] = np.array([output])

                    success[i] = True

            # Select those binaries with non-zero posteriors
            chains_good = np.copy(chains[success])
            derived_good = np.copy(derived[success])
            likelihood_good = np.copy(likelihood[success])

            # Only save successful values in binary_x_i, binary_data
            if len(self.chains) == 0:
                self.chains = chains_good
                self.derived = derived_good
                self.likelihood = likelihood_good
            else:
                self.chains = np.concatenate((self.chains, chains_good))
                self.derived = np.concatenate((self.derived, derived_good))
                self.likelihood = np.concatenate((self.likelihood, likelihood_good))
