"""
This script considers the problem
-laplace u = f, on (0,1)^2
u=0 on boundary of (0,1)^2

where the exact solution is imposed to be
u(x, y) = y(y-1)x(x-1) exp(-sigma_x(x-mu_x)^2 -sigma_y(y-mu_y)^2)
f(x, y) = -laplace u

Notes
-----
- the symbolic computations are outsourced to a jupyter notebook using `sympy`
"""
import numpy as np


sigma_x = 100.
sigma_y = 40.
mu_x = 37./73
mu_y = 41./73.

# calculated using simpson's rule,
# see `variational_adaptivity/gauss_kick/experiment_notes.ipynb`
energy_squared_exact = 0.01305598516695022655


class ExperimentConfig:
    N_REFINEMENTS = 9
    N_INITIAL_REFINEMENTS = 2


def analytical(r: np.ndarray) -> np.ndarray:
    """returns the imposed analytical solution"""
    x, y = r[:, 0], r[:, 1]
    u = (x*y*(x - 1)*(y - 1)*np.exp(-sigma_x *
         (-mu_x + x)**2 - sigma_y*(-mu_y + y)**2))
    return u


def f(r: np.ndarray) -> float:
    """returns -((d/dx)^2 + (d/dy)^2)analytical(x,y)"""
    x, y = r[:, 0], r[:, 1]
    laplace = (2*(x*(x - 1)*(2*sigma_y*y*(mu_y - y) + sigma_y*y*(y - 1)*(2*sigma_y*(mu_y - y)**2 - 1) + 2*sigma_y*(mu_y - y)*(y - 1) + 1) + y*(y - 1)*(2*sigma_x *
               x*(mu_x - x) + sigma_x*x*(x - 1)*(2*sigma_x*(mu_x - x)**2 - 1) + 2*sigma_x*(mu_x - x)*(x - 1) + 1))*np.exp(-sigma_x*(-mu_x + x)**2 - sigma_y*(-mu_y + y)**2))
    return -laplace


def uD(r: np.ndarray) -> np.ndarray:
    """returns homogeneous boundary conditions"""
    return np.zeros(r.shape[0])
