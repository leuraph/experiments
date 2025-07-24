from p1afempy.data_structures import \
    BoundaryConditionType, CoordinatesType, ElementsType, BoundaryType
from typing import Callable
import numpy as np


class Rectangle:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __init__(
            self,
            x_min: float, x_max: float,
            y_min: float, y_max: float):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def has_coordinates(self, coordinates: CoordinatesType) -> list[bool]:
        x, y = coordinates[:, 0], coordinates[:, 1]
        return (
            (self.x_min < x) &
            (x < self.x_max) &
            (self.y_min < y) &
            (y < self.y_max))


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
    nabla (A(x) nabla u(x)) + cu(x) = f(x),
    where A(x)_ij = a_ij(x) is a 2x2 Matrix
    """
    f: BoundaryConditionType
    a_11: BoundaryConditionType
    a_12: BoundaryConditionType
    a_21: BoundaryConditionType
    a_22: BoundaryConditionType
    c: float

    # a function that returns a coarse mesh of the problem's domain
    get_coarse_initial_mesh: Callable[[], Mesh]

    def __init__(
            self,
            f: BoundaryConditionType,
            a_11: BoundaryConditionType,
            a_12: BoundaryConditionType,
            a_21: BoundaryConditionType,
            a_22: BoundaryConditionType,
            c: float,
            get_coarse_initial_mesh: Callable[[], Mesh]):
        self.f = f
        self.a_11 = a_11
        self.a_12 = a_12
        self.a_21 = a_21
        self.a_22 = a_22
        self.c = c
        self.get_coarse_initial_mesh = get_coarse_initial_mesh


def get_coarse_square_mesh() -> Mesh:
    """
    returns a coarse mesh for the domain (0,1)^2 with
    homogeneous Dirichlet boundary conditions
    """

    coordinates = np.array([
        [0., 0.],
        [1., 0.],
        [1., 1.],
        [0., 1.]
    ])
    elements = np.array([
        [0, 1, 2],
        [2, 3, 0]
    ])
    dirichlet = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0]
    ])
    boundaries = [dirichlet]

    return Mesh(
        coordinates=coordinates,
        elements=elements,
        boundaries=boundaries)


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
        return - np.ones(n_vertices, dtype=float) * 1e-2

    def a_12(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return np.zeros(n_vertices, dtype=float)

    def a_21(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return np.zeros(n_vertices, dtype=float)

    c = 1.0

    return Problem(
        f=f, a_11=a_11, a_12=a_12,
        a_21=a_21, a_22=a_22, c=c,
        get_coarse_initial_mesh=get_coarse_square_mesh)


def get_problem_2() -> Problem:
    def f(r: CoordinatesType) -> float:
        """returns ones only"""
        return np.ones(r.shape[0], dtype=float)

    def a_11(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return - np.ones(n_vertices, dtype=float) * 1e-2

    def a_22(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return - np.ones(n_vertices, dtype=float) * 1e-2

    def a_12(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return np.zeros(n_vertices, dtype=float)

    def a_21(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return np.zeros(n_vertices, dtype=float)

    c = 1.0

    return Problem(
        f=f, a_11=a_11, a_12=a_12,
        a_21=a_21, a_22=a_22, c=c,
        get_coarse_initial_mesh=get_coarse_square_mesh)


def get_problem_3() -> Problem:

    def kappa(coordinates: CoordinatesType):
        omega_1 = Rectangle(0.1, 0.3, 0.1, 0.2)
        omega_2 = Rectangle(0.4, 0.7, 0.1, 0.3)
        omega_3 = Rectangle(0.4, 0.6, 0.5, 0.8)

        in_omega_1 = omega_1.has_coordinates(coordinates)
        in_omega_2 = omega_2.has_coordinates(coordinates)
        in_omega_3 = omega_3.has_coordinates(coordinates)

        # Values for each region
        values = [1e2, 1e4, 1e6]

        # Default value (like `else`)
        default_value = 1.0

        return np.select(
            [in_omega_1, in_omega_2, in_omega_3],
            values, default=default_value)

    def f(r: CoordinatesType) -> float:
        """returns ones only"""
        return np.ones(r.shape[0], dtype=float)

    def a_11(r: CoordinatesType) -> np.ndarray:
        return - kappa(coordinates=r)

    def a_22(r: CoordinatesType) -> np.ndarray:
        return - kappa(coordinates=r)

    def a_12(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return np.zeros(n_vertices, dtype=float)

    def a_21(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return np.zeros(n_vertices, dtype=float)

    c = 1.0

    return Problem(
        f=f, a_11=a_11, a_12=a_12,
        a_21=a_21, a_22=a_22, c=c,
        get_coarse_initial_mesh=get_coarse_square_mesh)


def get_problem_4() -> Problem:

    def kappa(coordinates: CoordinatesType):
        omega_1 = Rectangle(0.1, 0.3, 0.1, 0.2)
        omega_2 = Rectangle(0.4, 0.7, 0.1, 0.3)
        omega_3 = Rectangle(0.8, 1.0, 0.7, 1.0)

        in_omega_1 = omega_1.has_coordinates(coordinates)
        in_omega_2 = omega_2.has_coordinates(coordinates)
        in_omega_3 = omega_3.has_coordinates(coordinates)

        # Values for each region
        values = [10., 0.1, 0.05]

        # Default value (like `else`)
        default_value = 1.0

        return np.select(
            [in_omega_1, in_omega_2, in_omega_3],
            values, default=default_value)

    def f(r: CoordinatesType) -> float:
        """returns ones only"""
        return np.ones(r.shape[0], dtype=float)

    def a_11(r: CoordinatesType) -> np.ndarray:
        return - kappa(coordinates=r)

    def a_22(r: CoordinatesType) -> np.ndarray:
        return - kappa(coordinates=r)

    def a_12(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return np.zeros(n_vertices, dtype=float)

    def a_21(r: CoordinatesType) -> np.ndarray:
        n_vertices = r.shape[0]
        return np.zeros(n_vertices, dtype=float)

    c = 1.0

    return Problem(
        f=f, a_11=a_11, a_12=a_12,
        a_21=a_21, a_22=a_22, c=c,
        get_coarse_initial_mesh=get_coarse_square_mesh)


def get_problem_5() -> Problem:
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

    c = 0.0

    return Problem(
        f=f, a_11=a_11, a_12=a_12,
        a_21=a_21, a_22=a_22, c=c,
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
    if number == 5:
        return get_problem_5()
    raise RuntimeError(f'unknown problem number: {number}')
