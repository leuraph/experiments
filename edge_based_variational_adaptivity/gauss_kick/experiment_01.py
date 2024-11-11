import numpy as np
from p1afempy import io_helpers, refinement, solvers
from p1afempy.mesh import provide_geometric_data
from p1afempy.refinement import refineNVB_edge_based
from pathlib import Path
from utils import distort_coordinates, shuffle_elements
from configuration import uD, f
import pickle
import argparse


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
    # TODO if this order is what we want, then remove the other one and rename
    path_to_elements = base_path / Path('elements_order1.dat')
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
    # marking only non-boundary coordinates for jiggling
    # all_coordinates_indices = np.arange(coordinates.shape[0])
    # coordinates_on_boundary = np.isin(all_coordinates_indices, boundaries[0])
    # marked_coordinates = np.logical_not(coordinates_on_boundary)
    # # jiggle the initial mesh's non-boundary coordinates
    # delta = 1./2**(n_initial_refinements+3)
    # coordinates = distort_coordinates(coordinates=coordinates,
    #                                   delta=delta, marked=marked_coordinates)
    # shuffle initial mesh's elements
    # elements = shuffle_elements(elements=elements)

    # solve exactly on the initial mesh
    solution, _ = solvers.solve_laplace(
        coordinates=coordinates,
        elements=elements,
        dirichlet=boundaries[0],
        neumann=np.array([]),
        f=f, g=None, uD=uD)

    n_vertices = coordinates.shape[0]
    indices_of_free_nodes = np.setdiff1d(
        ar1=np.arange(n_vertices),
        ar2=np.unique(boundaries[0].flatten()))
    free_nodes = np.zeros(n_vertices, dtype=bool)
    free_nodes[indices_of_free_nodes] = 1
    n_dofs = np.sum(free_nodes)

    dump_object(obj=solution,
                path_to_file=base_results_path / Path(
                    f'{n_dofs}/solution.pkl'))
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

    n_refinements = 4
    for _ in range(n_refinements):

        # compute all local energy gains
        # ------------------------------
        # TODO

        # mark elements to be refined, then refine
        # ---------------------------------------
        # TODO provide the correct logic when marking
        element_to_edges, edge_to_nodes, boundaries_to_edges = \
            provide_geometric_data(elements=elements, boundaries=boundaries)
        n_edges = edge_to_nodes.shape[0]
        marked_edges = np.ones(n_edges, dtype=int)

        coordinates, elements, boundaries, _ = refineNVB_edge_based(
            coordinates=coordinates,
            elements=elements,
            boundary_conditions=boundaries,
            element2edges=element_to_edges,
            edge_to_nodes=edge_to_nodes,
            boundaries_to_edges=boundaries_to_edges,
            edge2newNode=marked_edges)

        # shuffle refined mesh's elements
        elements = shuffle_elements(elements=elements)

        # solve linear problem exactly on current mesh
        # --------------------------------------------
        solution, _ = solvers.solve_laplace(
            coordinates=coordinates,
            elements=elements,
            dirichlet=boundaries[0],
            neumann=np.array([]),
            f=f, g=None, uD=uD)

        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        n_dofs = np.sum(free_nodes)

        dump_object(obj=solution, path_to_file=base_results_path /
                    Path(f'{n_dofs}/solution.pkl'))
        dump_object(obj=elements, path_to_file=base_results_path /
                    Path(f'{n_dofs}/elements.pkl'))
        dump_object(obj=coordinates, path_to_file=base_results_path /
                    Path(f'{n_dofs}/coordinates.pkl'))
        dump_object(obj=boundaries, path_to_file=base_results_path /
                    Path(f'{n_dofs}/boundaries.pkl'))


def dump_object(obj, path_to_file: Path) -> None:
    path_to_file.parent.mkdir(parents=True, exist_ok=True)
    # save result as a pickle dump of pd.dataframe
    with open(path_to_file, "wb") as file:
        # Dump the DataFrame into the file using pickle
        pickle.dump(obj, file)


if __name__ == '__main__':
    main()
