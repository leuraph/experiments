"""
This experiment considers the Problem
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
on (0,1)^2 with homogeneous boundary conditions.
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
from custom_callback import AriolisAdaptiveDelayCustomCallback
import argparse
from scipy.sparse.linalg import cg
from pathlib import Path
from load_save_dumps import dump_object
from p1afempy.mesh import provide_geometric_data
from ismember import is_row_in
from variational_adaptivity.edge_based_variational_adaptivity import \
    get_energy_gains
from variational_adaptivity.markers import doerfler_marking
from p1afempy.refinement import refineNVB_edge_based
from custom_callback import ConvergedException
from scipy.sparse import csr_matrix, diags


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, required=True,
                        help="value of theta used in the Dörfler marking")
    parser.add_argument("--fudge", type=float, required=True,
                        help="additional fudge parameter "
                        "in Ariolis stopping criterion")
    parser.add_argument("--miniter", type=int, required=True,
                        help="minimum number of iterations on each mesh")
    parser.add_argument("--initial_delay", type=int, required=True,
                        help="initial delay parameter for the "
                        "Hestenes-Stiefel Estimator")
    parser.add_argument("--delay_increase", type=int, required=True,
                        help="increase delay by this amount if needed")
    parser.add_argument("--tau", type=float, required=True,
                        help="tau parameter in the adaptive delay choice")
    args = parser.parse_args()

    MINITER = args.miniter
    DELAY = args.initial_delay
    DELAY_INCREASE = args.delay_increase
    TAU = args.tau
    THETA = args.theta
    FUDGE = args.fudge

    n_max_dofs = 1e6
    n_initial_refinements = 5

    # ------------------------------------------------
    # Setup
    # ------------------------------------------------
    np.random.seed(42)

    base_results_path = (
        Path('results/experiment_05') /
        Path(
            f'theta-{THETA}_fudge-{FUDGE}_'
            f'miniter-{MINITER}_initial_delay-{DELAY}_'
            f'tau-{TAU}_delay_increase-{DELAY_INCREASE}'))

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

    # initializing the iteration with Galerkin
    # solution on first mesh
    current_iterate = np.copy(galerkin_solution)

    # dump initial mesh and initial Galerkin solution
    # -----------------------------------------------
    dump_object(obj=elements, path_to_file=base_results_path /
                Path(f'{n_dofs}/elements.pkl'))
    dump_object(obj=coordinates, path_to_file=base_results_path /
                Path(f'{n_dofs}/coordinates.pkl'))
    dump_object(obj=boundaries, path_to_file=base_results_path /
                Path(f'{n_dofs}/boundaries.pkl'))
    dump_object(
        obj=galerkin_solution, path_to_file=base_results_path /
        Path(f'{n_dofs}/galerkin_solution.pkl'))
    dump_object(
        obj=current_iterate, path_to_file=base_results_path /
        Path(f'{n_dofs}/last_iterate.pkl'))
    dump_object(
        obj=int(0), path_to_file=base_results_path /
        Path(f'{n_dofs}/n_iterations_done.pkl'))
    dump_object(
        obj=int(0), path_to_file=base_results_path /
        Path(f'{n_dofs}/delay_at_convergence.pkl'))
    # -----------------------------------------------------

    # perform first refinement by hand
    # --------------------------------
    # (non-boundary) edges
    _, edge_to_nodes, _ = \
        provide_geometric_data(
            elements=elements,
            boundaries=boundaries)

    edge_to_nodes_flipped = np.column_stack(
        [edge_to_nodes[:, 1], edge_to_nodes[:, 0]])
    boundary = np.logical_or(
        is_row_in(edge_to_nodes, boundaries[0]),
        is_row_in(edge_to_nodes_flipped, boundaries[0])
    )
    non_boundary = np.logical_not(boundary)
    edges = edge_to_nodes
    non_boundary_edges = edge_to_nodes[non_boundary]

    # free nodes / edges
    n_vertices = coordinates.shape[0]
    indices_of_free_nodes = np.setdiff1d(
        ar1=np.arange(n_vertices),
        ar2=np.unique(boundaries[0].flatten()))
    free_nodes = np.zeros(n_vertices, dtype=bool)
    free_nodes[indices_of_free_nodes] = 1
    free_edges = non_boundary  # integer array (holding actual indices)
    free_nodes = free_nodes  # boolean array

    energy_gains = get_energy_gains(
        coordinates=coordinates,
        elements=elements,
        non_boundary_edges=non_boundary_edges,
        current_iterate=current_iterate,
        f=f,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR,
        verbose=False)

    # dörfler based on EVA
    marked_edges = np.zeros(edges.shape[0], dtype=int)
    marked_non_boundary_egdes = doerfler_marking(
        input=energy_gains, theta=THETA)
    marked_edges[free_edges] = marked_non_boundary_egdes

    element_to_edges, edge_to_nodes, boundaries_to_edges =\
        provide_geometric_data(elements=elements, boundaries=boundaries)

    coordinates, elements, boundaries, current_iterate = \
        refineNVB_edge_based(
            coordinates=coordinates,
            elements=elements,
            boundary_conditions=boundaries,
            element2edges=element_to_edges,
            edge_to_nodes=edge_to_nodes,
            boundaries_to_edges=boundaries_to_edges,
            edge2newNode=marked_edges,
            to_embed=current_iterate)
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

        # perform CG on the current mesh
        # ------------------------------
        custom_callback = AriolisAdaptiveDelayCustomCallback(
            batch_size=1,
            min_n_iterations_per_mesh=MINITER,
            elements=elements,
            coordinates=coordinates,
            boundaries=boundaries,
            initial_delay=DELAY,
            delay_increase=DELAY_INCREASE,
            tau=TAU,
            fudge=FUDGE,
            lhs_matrix=lhs_matrix,
            rhs_vector=rhs_vector)

        cg_converged = False

        try:
            lhs_reduced = lhs_matrix[free_nodes, :][:, free_nodes]
            diagonal = lhs_reduced.diagonal()
            M = diags(diagonals=1./diagonal)
            current_iterate[free_nodes], _ = cg(
                A=lhs_matrix[free_nodes, :][:, free_nodes],
                b=rhs_vector[free_nodes],
                x0=current_iterate[free_nodes],
                M=M,
                rtol=1e-100,
                callback=custom_callback)
        except ConvergedException as conv:
            cg_converged = True
            current_iterate = conv.last_iterate
            energy_history = np.array(conv.energy_history)
            n_iterations_done = conv.n_iterations_done
            delay_at_convergence = conv.delay
            print(f"CG stopped after {conv.n_iterations_done} iterations!")

        if not cg_converged:
            raise RuntimeError('CG failed to converge. Stopping immediately.')

        # dump the current state
        # ----------------------
        dump_object(obj=elements, path_to_file=base_results_path /
                    Path(f'{n_dofs}/elements.pkl'))
        dump_object(obj=coordinates, path_to_file=base_results_path /
                    Path(f'{n_dofs}/coordinates.pkl'))
        dump_object(obj=boundaries, path_to_file=base_results_path /
                    Path(f'{n_dofs}/boundaries.pkl'))
        dump_object(obj=energy_history, path_to_file=base_results_path /
                    Path(f'{n_dofs}/energy_history.pkl'))
        dump_object(obj=n_iterations_done, path_to_file=base_results_path /
                    Path(f'{n_dofs}/n_iterations_done.pkl'))
        dump_object(obj=delay_at_convergence, path_to_file=base_results_path /
                    Path(f'{n_dofs}/delay_at_convergence.pkl'))
        dump_object(
            obj=galerkin_solution, path_to_file=base_results_path /
            Path(f'{n_dofs}/galerkin_solution.pkl'))
        dump_object(obj=current_iterate, path_to_file=base_results_path /
                    Path(f'{n_dofs}/last_iterate.pkl'))

        # stop right before refining if maximum number of DOFs is reached
        if n_dofs >= n_max_dofs:
            print(
                f'Maximum number of DOFs ({n_max_dofs})'
                'reached, stopping iteration.')
            break

        # perform edge-based variational adaptivity
        # -----------------------------------------
        _, edge_to_nodes, _ = \
            provide_geometric_data(
                elements=elements,
                boundaries=boundaries)

        edge_to_nodes_flipped = np.column_stack(
            [edge_to_nodes[:, 1], edge_to_nodes[:, 0]])
        boundary = np.logical_or(
            is_row_in(edge_to_nodes, boundaries[0]),
            is_row_in(edge_to_nodes_flipped, boundaries[0])
        )
        non_boundary = np.logical_not(boundary)
        edges = edge_to_nodes
        non_boundary_edges = edge_to_nodes[non_boundary]

        # free nodes / edges
        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        free_edges = non_boundary  # integer array (holding actual indices)
        free_nodes = free_nodes  # boolean array

        energy_gains = get_energy_gains(
            coordinates=coordinates,
            elements=elements,
            non_boundary_edges=non_boundary_edges,
            current_iterate=current_iterate,
            f=f,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR,
            verbose=True,
            parallel=False)

        # dörfler based on EVA
        # --------------------
        marked_edges = np.zeros(edges.shape[0], dtype=int)
        marked_non_boundary_egdes = doerfler_marking(
            input=energy_gains, theta=THETA)
        marked_edges[free_edges] = marked_non_boundary_egdes

        element_to_edges, edge_to_nodes, boundaries_to_edges =\
            provide_geometric_data(elements=elements, boundaries=boundaries)

        coordinates, elements, boundaries, current_iterate = \
            refineNVB_edge_based(
                coordinates=coordinates,
                elements=elements,
                boundary_conditions=boundaries,
                element2edges=element_to_edges,
                edge_to_nodes=edge_to_nodes,
                boundaries_to_edges=boundaries_to_edges,
                edge2newNode=marked_edges,
                to_embed=current_iterate)


if __name__ == '__main__':
    main()
