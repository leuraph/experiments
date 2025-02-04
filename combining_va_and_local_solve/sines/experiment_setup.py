from p1afempy.data_structures import CoordinatesType, ElementsType, BoundaryType
import numpy as np
from p1afempy import solvers
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix


def u(r: CoordinatesType) -> np.ndarray:
    """exact solution to the problem at hand"""
    return np.sin(2.*np.pi * r[:, 0])*np.sin(3.*np.pi * r[:, 1])


def f(r: CoordinatesType) -> float:
    """returns -((d/dx)^2 + (d/dy)^2)analytical(x,y)"""
    return 13.*np.pi**2*np.sin(2.*np.pi * r[:, 0])*np.sin(3.*np.pi * r[:, 1])


def uD(r: np.ndarray) -> np.ndarray:
    """returns homogeneous boundary conditions"""
    return np.zeros(r.shape[0])


def grad_u(r: CoordinatesType) -> np.ndarray:
    """gradient of the exact solution to the problem at hand"""
    xs = r[:, 0]
    ys = r[:, 1]
    tmp_x = np.cos(2. * np.pi * xs) * np.sin(3. * np.pi * ys)
    tmp_y = np.sin(2. * np.pi * xs) * np.cos(3. * np.pi * ys)
    return np.pi * np.column_stack([2.*tmp_x, 3.*tmp_y])


def get_exact_galerkin_solution(
        coordinates: CoordinatesType,
        elements: ElementsType,
        boundaries: list[BoundaryType]
) -> np.ndarray:
    n_vertices = coordinates.shape[0]
    indices_of_free_nodes = np.setdiff1d(
        ar1=np.arange(n_vertices),
        ar2=np.unique(boundaries[0].flatten()))
    free_nodes = np.zeros(n_vertices, dtype=bool)
    free_nodes[indices_of_free_nodes] = 1

    right_hand_side = solvers.get_right_hand_side(
        coordinates=coordinates,
        elements=elements,
        f=f)

    # assembly of the stiffness matrix
    stiffness_matrix = csr_matrix(solvers.get_stiffness_matrix(
        coordinates=coordinates,
        elements=elements))

    reduced_exact_solution = spsolve(
        stiffness_matrix[free_nodes, :][:, free_nodes],
        right_hand_side[free_nodes])

    full_solution = np.zeros(n_vertices)
    full_solution[free_nodes] = reduced_exact_solution

    return full_solution
