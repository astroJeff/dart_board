.. dart_board documentation master file, created by
   sphinx-quickstart on Sun Feb  4 18:09:00 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**dart_board**
======================================



dart_board makes it easy to model the formation and evolution of binary stars. It acts as a statistical wrapper around rapid binary evolution codes.

This documentation will provide a guide to using our code, but for details on the statistical approach, some tests, and various applications, check out `our paper <http://cdsads.u-strasbg.fr/abs/2017arXiv171011030A>`_.




a first example
--------------------------------------

To generate a population of high mass X-ray binaries (HMXB), use the following example ::

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


getting started
--------------------------------------

You will want to start with the installation guide to get dart_board running. Until we have our quickstart tutorial completed, the documentation we provide below is probably the best way to learn how to use dart_board for your needs. For more concrete examples, we provide the source code used to generate all the examples and figures in `our paper <http://cdsads.u-strasbg.fr/abs/2017arXiv171011030A>`_. If you have problems, feel free to `label an issue <https://github.com/astroJeff/dart_board/issues>`_.


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/install
   guide/dart_board
   guide/usage
   guide/sf_history
   guide/import


copyright and licensing
--------------------------------------

Copyright (c) 2017 Jeff J. Andrews & contributors

dart_board is made freely available under the MIT license.

If you use dart_board in your research, please cite `our paper <http://cdsads.u-strasbg.fr/abs/2017arXiv171011030A>`_.


changelog
--------------------------------------

Version 1.0.0 - Feb. 2018




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
