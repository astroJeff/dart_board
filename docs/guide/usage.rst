
Using dart_board for your needs
======================================

**This Page is Under Construction**


dart_board can be used with relatively little effort for two broadly defined tasks: modeling binary populations and fitting individual systems. We provide broad instruction on the usage for each of these cases below. We will assume that dart_board and pybse have both been installed on your system.



Modeling populations
---------------------------------

One may wish to generate populations of stellar binaries, a task which essentially covers any exercise previously performed in the literature by the technique labeled "binary population synthesis." In this case, the likelihood function for dart_board is the indicator function: if the model parameters produce a system of interest, it has a likelihood of unity, the likelihood is zero. Model parameters for priors are based on distributions commonly used in the binary population synthesis literature.

First, our basic example to generate a population of high mass X-ray binaries::

    import numpy as np
    import dart_board
    import pybse

    # Set up the sampler
    pub = dart_board.DartBoard("HMXB",
                               evolve_binary=pybse.evolve,
                               nwalkers=320)

    # Initialize the walkers
    pub.aim_darts()

    # Run the sampler
    pub.throw_darts(nburn=20000, nsteps=100000)

    # Save the chains
    np.save("HMXB_posterior_chains.npy", pub.chains)

Let's go over this line-by-line. We start by importing our necessary libraries::

    import numpy as np
    import dart_board
    import pybse

We will need numpy for manipulating the resulting set of posterior samples, which exists as a numpy array. dart_board is the library created upon installation, and pybse is the library that provides python bindings for bse.

Next, initialize the DartBoard object with::

    # Set up the sampler
    pub = dart_board.DartBoard("HMXB",
                               evolve_binary=pybse.evolve,
                               nwalkers=320)

The first argument ``"HMXB"`` is the only required argument, as this tells the likelihood function within dart_board to only select for high mass X-ray binaries. Currently there are only a handful of possible options, but we are working on this. It is simple to add your own to ``dart_board/posterior.py``. The next argument ``evolve_binary=pybse.evolve`` tells dart_board the name of the rapid binary evolution function. The default is pybse, but if you wish to add your own, check out our guide for importing your rapid binary evolution script to dart_board. We also tell dart_board to use 320 walkers. Consult the function definition for a guide to the many other optional arguments.

Now, we must initialize the walkers in an initial part of the parameter space. This is done through an iterative process that could probably be improved, but in practice is rarely a bottleneck. Although this function must be called, it requires no arguments::

    # Initialize the walkers
    pub.aim_darts()

Next, we can actually run the sampler::

    # Run the sampler
    pub.throw_darts(nburn=20000, nsteps=100000)

We have told the sampler to run for a burn-in of 20000 and then save the next 100000 steps.

The results of the sampler are saved as part of the DartBoard class. Principal among these are the chains which contain the posterior samples. we will save these to data for further use in post-processing with::

    # Save the chains
    np.save("HMXB_posterior_chains.npy", pub.chains)







Modeling individual systems
---------------------------------

The only difference between modeling populations and modeling individual systems is the inclusion of observational constraints. This is allowed through a ``kwargs`` argument passed to the DartBoard upon initialization. For instance, if we want to model the formation of the LMC high mass X-ray binary LMC X-3, we want to add constraints on the black hole mass (:math:`M_1 = 6.98\pm0.56\ M_{\odot}`), the donor mass (:math:`M_2 = 3.63\pm0.57\ M_{\odot}`), and the orbital period (:math:`P_{\rm orb} = 1.7\pm1` days). Our new code now looks like::

    import numpy as np
    import dart_board
    import pybse

    system_kwargs = {"M1" : 6.98, "M1_err" : 0.56, "M2" : 3.63, "M2_err" : 0.57, "P_orb" : 1.7, "P_orb_err" : 0.1}

    pub = dart_board.DartBoard("BHHMXB",
                               evolve_binary=pybse.evolve,
                               nwalkers=320,
                               system_kwargs=system_kwargs)

    pub.aim_darts()

    pub.throw_darts(nburn=20000, nsteps=10000)

    np.save("LMC-X3_posterior_chains.npy", pub.chains)

Notice that the only things that changed are the creation of system_kwargs, which is then passed as an argument when DartBoard is initialized, and the conversion of ``"HMXB"`` to ``"BHHMXB"``, to specify that we only care about high mass X-ray binaries with black hole accretors.

Note that uncertainties (assumed to be Gaussian) must be provided along with any observational values, otherwise the likelihood function is ill-defined.

A variety of observational constraints can be included. Look at the function definition for posterior.posterior_properties for a list and description of the available options.



Writing your own prior, likelihood, and posterior functions
-----------------------------------------------------------

This section is under construction.
