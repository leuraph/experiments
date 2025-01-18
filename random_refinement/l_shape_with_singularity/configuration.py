"""
This script considers the problem
-laplace u = 1, on (0,1)^2
u=0 on boundary of (0,1)^2

where the exact solution u is analytically unknown but
\|u\|_a^2 approximated by the code from Patrick Bammer
"""
import numpy as np
from p1afempy.data_structures import CoordinatesType


def f(r: CoordinatesType) -> float:
    """returns homogeneous boundary conditions"""
    return np.ones(r.shape[0], dtype=float)


def uD(r: CoordinatesType) -> np.ndarray:
    """returns homogeneous boundary conditions"""
    return np.zeros(r.shape[0], dtype=float)
