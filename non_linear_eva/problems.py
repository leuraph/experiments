from p1afempy.data_structures import \
    BoundaryConditionType, CoordinatesType, ElementsType, BoundaryType
from typing import Callable
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from triangle_cubature.cubature_rule import CubatureRuleEnum
from p1afempy.solvers import get_general_stiffness_matrix, get_right_hand_side


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


def get_zero_initial_guess(mesh: Mesh) -> np.ndarray:
    return np.zeros(mesh.coordinates.shape[0], dtype=float)


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
    get_initial_guess_on_initial_mesh: Callable[[Mesh], np.ndarray]

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
            get_coarse_initial_mesh: Callable[[], Mesh],
            get_initial_guess_on_initial_mesh: Callable[[Mesh], np.ndarray] = get_zero_initial_guess):
        self.f = f
        self.a_11 = a_11
        self.a_12 = a_12
        self.a_21 = a_21
        self.a_22 = a_22
        self.phi = phi
        self.phi_prime = phi_prime
        self.Phi = Phi
        self.get_coarse_initial_mesh = get_coarse_initial_mesh
        self.get_initial_guess_on_initial_mesh = get_initial_guess_on_initial_mesh


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


def get_initial_guess_for_cubic_helmholtz(
        mesh: Mesh, lamba: float, sign: float) -> np.ndarray:
    """
    on the (initial) mesh, solve the auxiliary problem
    -Laplace psi = 1, with homogeneous boundary conditions,
    then normalize the solution such that, on the vertex
    where the solution is maximal, the equation
    -Laplace (c psi) - lamba (c psi) - (c psi)^3 = 0
    is fulfilled, i.e.,
    c = sign \sqrt((lamba*psi_max - 1)/(psi_max**3))

    parameters
    ----------
    mesh: Mesh
    lamba: float
    sign: float
        either (-1) or (+1)
    """
    
    # ------------
    # RHS = 1
    # ------------
    def f_aux(r: CoordinatesType) -> float:
        """returns zeros only"""
        return np.ones(r.shape[0], dtype=float)

    # ------------------
    # Negative Laplacian
    # ------------------
    def a_11_aux(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return - np.ones(n_vertices, dtype=float)

    def a_22_aux(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return - np.ones(n_vertices, dtype=float)

    def a_12_aux(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return np.zeros(n_vertices, dtype=float)

    def a_21_aux(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return np.zeros(n_vertices, dtype=float)

    stiffness_matrix_aux = csr_matrix(get_general_stiffness_matrix(
        coordinates=mesh.coordinates,
        elements=mesh.elements,
        a_11=a_11_aux, a_12=a_12_aux, a_21=a_21_aux, a_22=a_22_aux,
        cubature_rule=CubatureRuleEnum.MIDPOINT))

    right_hand_side_vector_aux = get_right_hand_side(
        coordinates=mesh.coordinates,
        elements=mesh.elements,
        f=f_aux,
        cubature_rule=CubatureRuleEnum.MIDPOINT)
    
    n_coordinates = mesh.coordinates.shape[0]
    auxiliary_solution = np.zeros(n_coordinates)

    n_vertices = mesh.coordinates.shape[0]
    indices_of_free_nodes = np.setdiff1d(
        ar1=np.arange(n_vertices),
        ar2=np.unique(mesh.boundaries[0].flatten()))
    free_nodes = np.zeros(n_vertices, dtype=bool)
    free_nodes[indices_of_free_nodes] = 1

    auxiliary_solution[free_nodes] = spsolve(
        stiffness_matrix_aux[free_nodes, :][:, free_nodes],
        right_hand_side_vector_aux[free_nodes],
        use_umfpack=True)
    
    psi_max = np.max(auxiliary_solution)

    c = np.sqrt((lamba*psi_max - 1)/(psi_max**3))
    # ------------------------------------------------------

    return sign * c*auxiliary_solution


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
