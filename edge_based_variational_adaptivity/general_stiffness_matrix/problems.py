from p1afempy.data_structures import BoundaryConditionType, CoordinatesType
import numpy as np


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

    def __init__(
            self,
            f: BoundaryConditionType,
            a_11: BoundaryConditionType,
            a_12: BoundaryConditionType,
            a_21: BoundaryConditionType,
            a_22: BoundaryConditionType,
            c: float):
        self.f = f
        self.a_11 = a_11
        self.a_12 = a_12
        self.a_21 = a_21
        self.a_22 = a_22
        self.c = c


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
        a_21=a_21, a_22=a_22, c=c)


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
        a_21=a_21, a_22=a_22, c=c)


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
        a_21=a_21, a_22=a_22, c=c)


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
        a_21=a_21, a_22=a_22, c=c)


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
