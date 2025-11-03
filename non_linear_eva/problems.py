from p1afempy.data_structures import \
    BoundaryConditionType, CoordinatesType, ElementsType, BoundaryType
from typing import Callable
import numpy as np


class Mesh:
    coordinates: CoordinatesType
    elements: ElementsType
    boundaries: list[BoundaryType]

    def __init__(
            self,
            coordinates: CoordinatesType,
            elements: ElementsType,
            boundaries: list[BoundaryType]):
        self.coordinates = coordinates
        self.elements= elements
        self.boundaries = boundaries


class Problem:
    """
    This class resembles the
    homogeneous BVP problem
    nabla (A(x) nabla u(x)) + phi(u(x)) = f(x),
    where A(x)_ij = a_ij(x) is a 2x2 Matrix,
    phi: Omega -> R is C^1 non-linearity,
    phi' its derivative, and
    Phi its indefinite integral (without a constant)
    """
    f: BoundaryConditionType
    a_11: BoundaryConditionType
    a_12: BoundaryConditionType
    a_21: BoundaryConditionType
    a_22: BoundaryConditionType
    phi: Callable[[np.ndarray], np.ndarray]
    phi_prime: Callable[[np.ndarray], np.ndarray]
    Phi: Callable[[np.ndarray], np.ndarray]

    # a function that returns a coarse mesh of the problem's domain
    get_coarse_initial_mesh: Callable[[], Mesh]

    def __init__(
            self,
            f: BoundaryConditionType,
            a_11: BoundaryConditionType,
            a_12: BoundaryConditionType,
            a_21: BoundaryConditionType,
            a_22: BoundaryConditionType,
            phi: Callable[[np.ndarray], np.ndarray],
            phi_prime: Callable[[np.ndarray], np.ndarray],
            Phi: Callable[[np.ndarray], np.ndarray],
            get_coarse_initial_mesh: Callable[[], Mesh]):
        self.f = f
        self.a_11 = a_11
        self.a_12 = a_12
        self.a_21 = a_21
        self.a_22 = a_22
        self.phi = phi
        self.phi_prime = phi_prime
        self.Phi = Phi
        self.get_coarse_initial_mesh = get_coarse_initial_mesh


def get_coarse_L_shape_mesh() -> Mesh:
    """
    returns a coarse mesh for the L-shaped domain
    Omega = (-1, 1)^2 \ (0, 1)x(-1, 0) with
    homogeneous Dirichlet boundary conditions
    """

    coordinates = np.array([
        [-1, -1],
        [0, -1],
        [-1, 0],
        [0, 0],
        [1, 0],
        [-1, 1],
        [0, 1],
        [1, 1]
    ])
    elements = np.array([
        [3, 0, 1],
        [0, 3, 2],
        [6, 2, 3],
        [7, 3, 4],
        [2, 6, 5],
        [3, 7, 6]
    ])
    dirichlet = np.array([
        [0, 1],
        [1, 3],
        [3, 4],
        [4, 7],
        [7, 6],
        [6, 5],
        [5, 2],
        [2, 0]
    ])
    boundaries = [dirichlet]

    return Mesh(
        coordinates=coordinates,
        elements=elements,
        boundaries=boundaries)


def get_problem_1() -> Problem:
    def f(r: CoordinatesType) -> float:
        """returns ones only"""
        return np.ones(r.shape[0], dtype=float)

    def a_11(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return - np.ones(n_vertices, dtype=float)

    def a_22(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return - np.ones(n_vertices, dtype=float)

    def a_12(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return np.zeros(n_vertices, dtype=float)

    def a_21(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return np.zeros(n_vertices, dtype=float)

    def phi(u: np.ndarray) -> np.ndarray:
        return u**3
    
    def Phi(u: np.ndarray) -> np.ndarray:
        return u**4 / 4.
    
    def phi_prime(u: np.ndarray) -> np.ndarray:
        return 3. * u**2

    return Problem(
        f=f, a_11=a_11, a_12=a_12,
        a_21=a_21, a_22=a_22,
        phi=phi, phi_prime=phi_prime, Phi=Phi,
        get_coarse_initial_mesh=get_coarse_L_shape_mesh)


def get_problem_2() -> Problem:
    def f(r: CoordinatesType) -> float:
        """returns ones only"""
        return np.ones(r.shape[0], dtype=float)

    def a_11(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return - np.ones(n_vertices, dtype=float)

    def a_22(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return - np.ones(n_vertices, dtype=float)

    def a_12(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return np.zeros(n_vertices, dtype=float)

    def a_21(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return np.zeros(n_vertices, dtype=float)

    def phi(u: np.ndarray) -> np.ndarray:
        return u * np.abs(u)

    def Phi(u: np.ndarray) -> np.ndarray:
        return np.abs(u) * u**2 / 3.
    
    def phi_prime(u: np.ndarray) -> np.ndarray:
        return 2. * np.abs(u)

    return Problem(
        f=f, a_11=a_11, a_12=a_12,
        a_21=a_21, a_22=a_22,
        phi=phi, phi_prime=phi_prime, Phi=Phi,
        get_coarse_initial_mesh=get_coarse_L_shape_mesh)


def get_problem_3() -> Problem:
    def f(r: CoordinatesType) -> float:
        """returns ones only"""
        return np.ones(r.shape[0], dtype=float)

    def a_11(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return - np.ones(n_vertices, dtype=float)

    def a_22(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return - np.ones(n_vertices, dtype=float)

    def a_12(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return np.zeros(n_vertices, dtype=float)

    def a_21(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return np.zeros(n_vertices, dtype=float)

    def phi(u: np.ndarray) -> np.ndarray:
        return np.exp(u) - 1

    def Phi(u: np.ndarray) -> np.ndarray:
        return np.exp(u) - u
    
    def phi_prime(u: np.ndarray) -> np.ndarray:
        return np.exp(u)

    return Problem(
        f=f, a_11=a_11, a_12=a_12,
        a_21=a_21, a_22=a_22,
        phi=phi, phi_prime=phi_prime, Phi=Phi,
        get_coarse_initial_mesh=get_coarse_L_shape_mesh)


def get_problem_4() -> Problem:
    def f(r: CoordinatesType) -> float:
        """
        returns the RHS function f of the PDE
        - laplace u(x, y) + u(x, y)^3 = f(x, y),
        if we impose
        u(x, y) = 2r^{-4/3}xy(1-x^2)(1-y^2),
        where r:= sqrt(x^2 + y^2),
        see [1, chapter 4.2].

        References
        ----------
        [1] https://arxiv.org/abs/2504.11292

        Notes
        -----
        the right hand side was computed
        (and left unchanged, that's why it looks horrible)
        using the accompanying script
        `compute_rhs.py`
        in this folder
        """
        x, y = r[:, 0], r[:, 1]
        return (x*y*(x**2 + y**2)**(-7.0)*(8.0*x**2*y**2*(x**2 - 1)**3*(x**2 + y**2)**5.0*(y**2 - 1)**3 - 8.88888888888889*(x**2 - 1)*(x**2 + y**2)**5.33333333333333*(y**2 - 1) + (x**2 + y**2)**5.33333333333333*(10.6666666666667*x**2*(y**2 - 1) + 10.6666666666667*y**2*(x**2 - 1) + 16.0*(x**2 - 1)*(y**2 - 1)) + 12.0*(x**2 + y**2)**6.33333333333333*(-x**2 - y**2 + 2)))

    def a_11(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return - np.ones(n_vertices, dtype=float)

    def a_22(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return - np.ones(n_vertices, dtype=float)

    def a_12(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return np.zeros(n_vertices, dtype=float)

    def a_21(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return np.zeros(n_vertices, dtype=float)

    def phi(u: np.ndarray) -> np.ndarray:
        return u**3

    def Phi(u: np.ndarray) -> np.ndarray:
        return u**4. / 4.
    
    def phi_prime(u: np.ndarray) -> np.ndarray:
        return 3. * u**2.

    return Problem(
        f=f, a_11=a_11, a_12=a_12,
        a_21=a_21, a_22=a_22,
        phi=phi, phi_prime=phi_prime, Phi=Phi,
        get_coarse_initial_mesh=get_coarse_L_shape_mesh)


def get_problem(number: int) -> Problem:
    if number == 1:
        return get_problem_1()
    if number == 2:
        return get_problem_2()
    if number == 3:
        return get_problem_3()
    if number == 4:
        return get_problem_4()
    raise RuntimeError(f'unknown problem number: {number}')
