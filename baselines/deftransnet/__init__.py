"""DefTransNet: A Transformer-based Method for Non-Rigid Point Cloud Registration.

Vendored from https://github.com/m-kinz/DefTransNet — modularized from notebook.
"""

from .model import DefTransNet
from .inference import smooth_lbp

__all__ = ["DefTransNet", "smooth_lbp"]
