# -*- coding: utf-8 -*-


from .darts import *
from .priors import *
from .posterior import *
from .plotting import *
from .utils import *
from .plot_system_evolution import *

__version__ = "1.1.0"

# try:
#     __DART_BOARD_SETUP__
# except NameError:
#     __DART_BOARD_SETUP__ = False
#
# if not __DART_BOARD_SETUP__:
#     __all__ = ["DartBoard"]
