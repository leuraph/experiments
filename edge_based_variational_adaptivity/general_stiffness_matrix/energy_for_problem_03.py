"""
This scripts approximates the energy for the solution of
nabla (A(x) nabla u(x)) + u(x) = 1, A(x) = -kappa(x) Id,
kappa(x) = 1e2, x in Omega1,
kappa(x) = 1e4, x in Omega2,
kappa(x) = 1e6, x in Omega3,
kappa(x) = 1, else,
on (0,1)^2 with homogeneous boundary conditions
and
Omega1 = (0.1, 0.3) x (0.1, 0.2),
Omega2 = (0.4, 0.7) x (0.1, 0.3),
Omega3 = (0.4, 0.6) x (0.5, 0.8),
"""
import numpy as np
from p1afempy.data_structures import CoordinatesType
from p1afempy import refinement
from p1afempy.solvers import \
    get_general_stiffness_matrix, get_mass_matrix, get_right_hand_side
from triangle_cubature.cubature_rule import CubatureRuleEnum
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from p1afempy.refinement import refineNVB
from scipy.sparse import csr_matrix


class Square:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
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


def kappa(coordinates: CoordinatesType):
    omega_1 = Square(0.1, 0.3, 0.1, 0.2)
    omega_2 = Square(0.4, 0.7, 0.1, 0.3)
    omega_3 = Square(0.4, 0.6, 0.5, 0.8)

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


def uD(r: CoordinatesType) -> np.ndarray:
    """returns homogeneous boundary conditions"""
    return np.zeros(r.shape[0], dtype=float)


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


def show_solution(coordinates, solution):
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 12

    x_coords, y_coords = zip(*coordinates)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points with scalar values as colors (adjust colormap as needed)
    _ = ax.plot_trisurf(x_coords, y_coords, solution, linewidth=0.2,
                        antialiased=True, cmap=cm.viridis)
    # Add labels to the axes
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    plt.show()


def main() -> None:

    n_max_dofs = 100e6
    n_initial_refinements = 5

    # mesh
    # ----
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

    # initial refinement
    # ------------------
    for _ in range(n_initial_refinements):
        marked = np.arange(elements.shape[0])
        coordinates, elements, boundaries, _ = \
            refinement.refineNVB(coordinates=coordinates,
                                 elements=elements,
                                 marked_elements=marked,
                                 boundary_conditions=boundaries)

    # initial exact galerkin solution
    # -------------------------------
    # calculating free nodes on the initial mesh
    # ------------------------------------------
    n_vertices = coordinates.shape[0]
    indices_of_free_nodes = np.setdiff1d(
        ar1=np.arange(n_vertices),
        ar2=np.unique(boundaries[0].flatten()))
    free_nodes = np.zeros(n_vertices, dtype=bool)
    free_nodes[indices_of_free_nodes] = 1
    n_dofs = np.sum(free_nodes)

    rhs_vector = get_right_hand_side(
        coordinates=coordinates,
        elements=elements,
        f=f,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR)
    stiffness_matrix = get_general_stiffness_matrix(
        coordinates=coordinates,
        elements=elements,
        a_11=a_11,
        a_12=a_12,
        a_21=a_21,
        a_22=a_22,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR)
    mass_matrix = get_mass_matrix(
        coordinates=coordinates,
        elements=elements)
    lhs_matrix = csr_matrix(mass_matrix + stiffness_matrix)

    galerkin_solution = np.zeros(n_vertices)
    galerkin_solution[free_nodes] = spsolve(
        A=lhs_matrix[free_nodes, :][:, free_nodes],
        b=rhs_vector[free_nodes])

    coordinates, elements, boundaries, _ = \
        refineNVB(
            coordinates=coordinates,
            elements=elements,
            marked_elements=marked,
            boundary_conditions=boundaries)
    # --------------------------------

    while True:
        # reset as the mesh has changed
        # -----------------------------
        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        n_dofs = np.sum(free_nodes)

        general_stiffness_matrix = get_general_stiffness_matrix(
            coordinates=coordinates,
            elements=elements,
            a_11=a_11,
            a_12=a_12,
            a_21=a_21,
            a_22=a_22,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)
        mass_matrix = get_mass_matrix(
            coordinates=coordinates,
            elements=elements)

        lhs_matrix = csr_matrix(general_stiffness_matrix + mass_matrix)
        rhs_vector = get_right_hand_side(
            coordinates=coordinates,
            elements=elements,
            f=f,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        # compute the Galerkin solution on current mesh
        # ---------------------------------------------
        galerkin_solution = np.zeros(n_vertices)
        galerkin_solution[free_nodes] = spsolve(
            A=lhs_matrix[free_nodes, :][:, free_nodes],
            b=rhs_vector[free_nodes])

        # show_solution(coordinates, galerkin_solution)

        energy_norm_squared = galerkin_solution.dot(
            lhs_matrix.dot(galerkin_solution))
        print(
            f'nDOF = {n_dofs}, '
            f'Galerkin solution energy norm squared = {energy_norm_squared}')

        # stop right before refining if maximum number of DOFs is reached
        if n_dofs >= n_max_dofs:
            print(
                f'Maximum number of DOFs ({n_max_dofs})'
                'reached, stopping iteration.')
            break

        marked = np.ones(elements.shape[0], dtype=bool)
        coordinates, elements, boundaries, _ = \
            refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked,
                boundary_conditions=boundaries)


if __name__ == '__main__':
    main()
