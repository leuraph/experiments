import numpy as np
from p1afempy import io_helpers, solvers
from triangle_cubature.cubature_rule import CubatureRuleEnum
from p1afempy.refinement import refineNVB

from pathlib import Path
from configuration import uD, f
import pickle
import argparse
from tqdm import tqdm


def main() -> None:
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--percentage", type=float, required=True,
                        help="what percentage of all elemets get "
                        "randomly selected for refinement")
    args = parser.parse_args()

    PERCENTAGE = args.percentage

    # ------------------------------------------------
    # Setup
    # ------------------------------------------------
    base_path = Path('data')
    path_to_elements = base_path / Path('elements.dat')
    path_to_coordinates = base_path / Path('coordinates.dat')
    path_to_dirichlet = base_path / Path('dirichlet.dat')

    base_results_path = Path(
        'results/experiment_01') / Path(f'percentage_{PERCENTAGE}')

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
            refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked,
                boundary_conditions=boundaries)

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

    # -----------------------------------
    # random refinements and exact solves
    # -----------------------------------

    max_n_dofs = 1e6
    while True:

        # random refinement
        # -----------------

        # Calculate the number of elements to mark
        n_elements = elements.shape[0]
        n_to_mark = int(np.ceil(PERCENTAGE * n_elements))

        # Randomly select elements to mark
        marked = np.random.choice(n_elements, size=n_to_mark, replace=False)

        # refine
        coordinates, elements, boundaries, _ = \
            refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked,
                boundary_conditions=boundaries)

        # calculate number of degrees of freedom
        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        n_dofs = np.sum(free_nodes)

        # solve linear problem exactly on current mesh
        # --------------------------------------------
        print(f'Solving Problem for {n_dofs} DOFs...')
        solution, _ = solvers.solve_laplace(
            coordinates=coordinates,
            elements=elements,
            dirichlet=boundaries[0],
            neumann=np.array([]),
            f=f, g=None, uD=uD,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)
        print('Done!')

        dump_object(obj=solution, path_to_file=base_results_path /
                    Path(f'{n_dofs}/galerkin_solution.pkl'))
        dump_object(obj=elements, path_to_file=base_results_path /
                    Path(f'{n_dofs}/elements.pkl'))
        dump_object(obj=coordinates, path_to_file=base_results_path /
                    Path(f'{n_dofs}/coordinates.pkl'))
        dump_object(obj=boundaries, path_to_file=base_results_path /
                    Path(f'{n_dofs}/boundaries.pkl'))

        if n_dofs >= max_n_dofs:
            break


def dump_object(obj, path_to_file: Path) -> None:
    path_to_file.parent.mkdir(parents=True, exist_ok=True)
    # save result as a pickle dump of pd.dataframe
    with open(path_to_file, "wb") as file:
        # Dump the DataFrame into the file using pickle
        pickle.dump(obj, file)


if __name__ == '__main__':
    main()
