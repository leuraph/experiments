import numpy as np
from p1afempy import refinement
from p1afempy.solvers import \
    get_general_stiffness_matrix, get_right_hand_side
from triangle_cubature.cubature_rule import CubatureRuleEnum
from scipy.sparse.linalg import spsolve
from pathlib import Path
from load_save_dumps import dump_object
from p1afempy.mesh import provide_geometric_data
from ismember import is_row_in
from variational_adaptivity.edge_based_variational_adaptivity import \
    get_energy_gains
from p1afempy.refinement import refineNVB_edge_based
from scipy.sparse import csr_matrix
from p1afempy.data_structures import CoordinatesType
from variational_adaptivity.markers import doerfler_marking
import argparse

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


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, required=True,
                        help="dörfler parameter")
    args = parser.parse_args()

    THETA = args.theta

    n_max_dofs = 1e6
    n_initial_refinements = 2

    # ------------------------------------------------
    # Setup
    # ------------------------------------------------
    np.random.seed(42)

    base_results_path = Path('results/experiment_02') / Path(f'theta-{THETA}')

    # mesh
    # ----
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

    # initial refinement
    # ------------------
    for _ in range(n_initial_refinements):
        marked = np.arange(elements.shape[0])
        coordinates, elements, boundaries, _ = \
            refinement.refineNVB(coordinates=coordinates,
                                 elements=elements,
                                 marked_elements=marked,
                                 boundary_conditions=boundaries)


    while True:

        # galerkin solution
        # -----------------
        # calculating free nodes on the initial mesh
        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        n_dofs = np.sum(free_nodes)

        print(f'n_dof = {n_dofs}')

        if n_dofs > n_max_dofs:
            break

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
        lhs_matrix = csr_matrix(stiffness_matrix)

        def energy(u: np.ndarray) -> float:
            return 0.5 * u.dot(lhs_matrix.dot(u)) - rhs_vector.dot(u)

        galerkin_solution = np.zeros(n_vertices)
        galerkin_solution[free_nodes] = spsolve(
            A=lhs_matrix[free_nodes, :][:, free_nodes],
            b=rhs_vector[free_nodes])

        # dump mesh and Galerkin solution
        # -------------------------------
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
            obj=energy(galerkin_solution),
            path_to_file=base_results_path / Path(f'{n_dofs}/galerkin_solution_energy.pkl'))
        # -----------------------------------------------------

        # EVA
        # ---
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
            current_iterate=galerkin_solution,
            f=f,
            a_11=a_11,
            a_12=a_12,
            a_21=a_21,
            a_22=a_22,
            c=0.0,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR,
            verbose=False)

        # dörfler based on EVA
        marked_edges = np.zeros(edges.shape[0], dtype=int)
        marked_non_boundary_egdes = doerfler_marking(
            input=energy_gains, theta=THETA)
        marked_edges[free_edges] = marked_non_boundary_egdes

        # dumping sums of (marked) energy decays
        sum_of_all_energy_decays = np.sum(energy_gains)
        sum_of_all_marked_energy_decays = np.sum(energy_gains[marked_non_boundary_egdes])
        dump_object(
            obj=sum_of_all_energy_decays,
            path_to_file=base_results_path / Path(f'{n_dofs}/sum_energy_decays.pkl'))
        dump_object(
            obj=sum_of_all_marked_energy_decays,
            path_to_file=base_results_path / Path(f'{n_dofs}/sum_marked_energy_decays.pkl'))


        element_to_edges, edge_to_nodes, boundaries_to_edges =\
            provide_geometric_data(elements=elements, boundaries=boundaries)

        coordinates, elements, boundaries, galerkin_solution = \
            refineNVB_edge_based(
                coordinates=coordinates,
                elements=elements,
                boundary_conditions=boundaries,
                element2edges=element_to_edges,
                edge_to_nodes=edge_to_nodes,
                boundaries_to_edges=boundaries_to_edges,
                edge2newNode=marked_edges,
                to_embed=galerkin_solution)
        # --------------------------------

if __name__ == '__main__':
    main()
