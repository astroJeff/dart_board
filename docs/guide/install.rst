.. dart_board documentation installation instructions

Installation Guide
======================================



Installing dart_board
--------------------------------------


Currently, dart_board can only be installed from source. Assuming you have python, pip, and git installed, use the code snippet below to install dart_board::

    git clone https://github.com/astroJeff/dart_board.git
    cd dart_board
    pip install .



python bindings for BSE
--------------------------------------

dart_board does not come with a black box rapid binary evolution code, however we provide instructions for getting set up with bse. All examples and tutorials assume you have bse set up for use.

There are two separate pieces to using BSE within dart_board. First, we need to create python bindings for BSE using the compiler ```f2py``. Check out `this website <https://docs.scipy.org/doc/numpy-dev/f2py/>`_ for documentation on ``f2py``. Second, we need to create a separate python library, ``pyBSE``, that serves as a wrapper around the bse executable created with f2py.

* Start by downloading BSE (not SSE) from `Jarrod Hurley's website <http://astronomy.swin.edu.au/~jhurley/bsedload.html>`_. Place the downloaded tarball (``bse.tar``) into dart_board/pyBSE/
* Change directories to dart_board/pyBSE/ and unpack the tarball you just downloaded::

    cd pyBSE
    tar -xvf bse.tar

  You should now see a bunch of new files that contains the rapid binary evolution files from SSE/BSE.
* Move to the parent dart_board directory and apply the included BSE patch file::

    cd ..
    patch -s -p0 < BSE_updates.patch
  This will update all the BSE files to interface correctly with dart_board.
* Change directories back to pyBSE and make the bse executable::

    cd pyBSE
    make pybse
* Now, we need to create the pyBSE python library so that it is accessible to any directory. We have included a setup.py script for use with pip::

    pip install -e .



python dependencies
--------------------------------------

* numpy
* scipy
* matplotlib
* pickle
* astropy
* emcee
* corner (optional)
