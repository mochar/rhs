"""RHS"""

__version__ = '0.0.1'

from .rhs import *
from rhs.model import Configuration, Reparam, ReparamIG, ReparamII
from rhs.model import GuideUnstructured, GuideMatrix, GuidePairCond, GuidePairMv
import rhs.datasets
