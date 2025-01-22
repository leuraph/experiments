import numpy as np
from p1afempy import io_helpers, refinement, solvers
from p1afempy.mesh import provide_geometric_data
from p1afempy.refinement import refineNVB_edge_based
from variational_adaptivity.markers import doerfler_marking
from variational_adaptivity.edge_based_variational_adaptivity import\
     get_energy_gains
from pathlib import Path
from configuration import uD, f
from ismember import is_row_in
import pickle
import argparse
from triangle_cubature.cubature_rule import CubatureRuleEnum


def main() -> None:
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, required=True,
                        help="value of theta used in the Dörfler marking")
    args = parser.parse_args()

    THETA = args.theta

    # ------------------------------------------------
    # Setup
    # ------------------------------------------------
    base_path = Path('data')
    path_to_elements = base_path / Path('elements.dat')
    path_to_coordinates = base_path / Path('coordinates.dat')
    path_to_dirichlet = base_path / Path('dirichlet.dat')

    base_results_path = Path('results/experiment_01') / Path(f'theta_{THETA}')

    coordinates, elements = io_helpers.read_mesh(
        path_to_coordinates=path_to_coordinates,
        path_to_elements=path_to_elements,
        shift_indices=False)
    boundaries = [io_helpers.read_boundary_condition(
        path_to_boundary=path_to_dirichlet,
        shift_indices=False)]

    # initial refinement
    # ------------------
    n_initial_refinements = 5
    for _ in range(n_initial_refinements):
        marked = np.arange(elements.shape[0])
        coordinates, elements, boundaries, _ = \
            refinement.refineNVB(coordinates=coordinates,
                                 elements=elements,
                                 marked_elements=marked,
                                 boundary_conditions=boundaries)

    # solve exactly on the initial mesh
    solution, _ = solvers.solve_laplace(
        coordinates=coordinates,
        elements=elements,
        dirichlet=boundaries[0],
        neumann=np.array([]),
        f=f, g=None, uD=uD,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR)

    n_vertices = coordinates.shape[0]
    indices_of_free_nodes = np.setdiff1d(
        ar1=np.arange(n_vertices),
        ar2=np.unique(boundaries[0].flatten()))
    free_nodes = np.zeros(n_vertices, dtype=bool)
    free_nodes[indices_of_free_nodes] = 1
    n_dofs = np.sum(free_nodes)

    dump_object(obj=solution,
                path_to_file=base_results_path / Path(
                    f'{n_dofs}/galerkin_solution.pkl'))
    dump_object(obj=elements,
                path_to_file=base_results_path / Path(
                    f'{n_dofs}/elements.pkl'))
    dump_object(obj=coordinates,
                path_to_file=base_results_path / Path(
                    f'{n_dofs}/coordinates.pkl'))
    dump_object(obj=boundaries,
                path_to_file=base_results_path / Path(
                    f'{n_dofs}/boundaries.pkl'))

    # ------------------------------------------------
    # variational adaptivity
    # ------------------------------------------------

    n_max_dofs = 1e6
    while True:

        element_to_edges, edge_to_nodes, boundaries_to_edges =\
            provide_geometric_data(elements=elements, boundaries=boundaries)

        # compute all local energy gains
        # ------------------------------
        edge_to_nodes_flipped = np.column_stack(
            [edge_to_nodes[:, 1], edge_to_nodes[:, 0]])
        boundary = np.logical_or(
            is_row_in(edge_to_nodes, boundaries[0]),
            is_row_in(edge_to_nodes_flipped, boundaries[0])
        )
        non_boundary = np.logical_not(boundary)
        n_non_boundary_edges = np.sum(non_boundary)
        marked_edges = np.zeros(edge_to_nodes.shape[0], dtype=int)
        energy_gains = np.zeros(n_non_boundary_edges, dtype=float)
        non_boundary_edges = edge_to_nodes[non_boundary]

        energy_gains = get_energy_gains(
            coordinates=coordinates,
            elements=elements,
            non_boundary_edges=non_boundary_edges,
            current_iterate=solution,
            f=f,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR,
            verbose=True)

        # mark elements to be refined, then refine
        # ---------------------------------------
        marked_non_boundary_egdes = doerfler_marking(
            input=energy_gains, theta=THETA)
        marked_edges[non_boundary] = marked_non_boundary_egdes

        coordinates, elements, boundaries, _ = refineNVB_edge_based(
            coordinates=coordinates,
            elements=elements,
            boundary_conditions=boundaries,
            element2edges=element_to_edges,
            edge_to_nodes=edge_to_nodes,
            boundaries_to_edges=boundaries_to_edges,
            edge2newNode=marked_edges)

        # solve linear problem exactly on current mesh
        # --------------------------------------------
        solution, _ = solvers.solve_laplace(
            coordinates=coordinates,
            elements=elements,
            dirichlet=boundaries[0],
            neumann=np.array([]),
            f=f, g=None, uD=uD,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        n_dofs = np.sum(free_nodes)

        dump_object(obj=solution, path_to_file=base_results_path /
                    Path(f'{n_dofs}/galerkin_solution.pkl'))
        dump_object(obj=elements, path_to_file=base_results_path /
                    Path(f'{n_dofs}/elements.pkl'))
        dump_object(obj=coordinates, path_to_file=base_results_path /
                    Path(f'{n_dofs}/coordinates.pkl'))
        dump_object(obj=boundaries, path_to_file=base_results_path /
                    Path(f'{n_dofs}/boundaries.pkl'))

        if n_dofs >= n_max_dofs:
            break


def dump_object(obj, path_to_file: Path) -> None:
    path_to_file.parent.mkdir(parents=True, exist_ok=True)
    # save result as a pickle dump of pd.dataframe
    with open(path_to_file, "wb") as file:
        # Dump the DataFrame into the file using pickle
        pickle.dump(obj, file)


if __name__ == '__main__':
    main()
