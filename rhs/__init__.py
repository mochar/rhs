"""RHS"""

__version__ = '0.0.1'

from .rhs import *
from rhs.model import Configuration, Reparam, ReparamIG, ReparamII
from rhs.model import GuideUnstructured, GuideFullMatrix, GuidePairCond, GuidePairMv, GuidePairCondCorr
import rhs.datasets
