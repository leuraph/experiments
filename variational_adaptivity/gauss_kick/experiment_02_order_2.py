import numpy as np
from p1afempy import io_helpers, refinement, solvers
from p1afempy.mesh import get_element_to_neighbours
from pathlib import Path
from variational_adaptivity import algo_4_1, markers
from configuration import uD, f, ExperimentConfig
import time
import pickle
import pandas as pd
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, required=True,
                        help="value of theta used in the Dörfler marking")
    args = parser.parse_args()

    THETA = args.theta

    # ------------------------------------------------
    # Setup
    # ------------------------------------------------
    base_path = Path('data')
    path_to_elements = base_path / Path('elements_order2.dat')
    path_to_coordinates = base_path / Path('coordinates.dat')
    path_to_dirichlet = base_path / Path('dirichlet.dat')

    base_results_path = Path('results/experiment_02') / Path(f'theta_{THETA}')

    coordinates, elements = io_helpers.read_mesh(
        path_to_coordinates=path_to_coordinates,
        path_to_elements=path_to_elements,
        shift_indices=False)
    boundaries = [io_helpers.read_boundary_condition(
        path_to_boundary=path_to_dirichlet,
        shift_indices=False)]

    energies = []
    n_elements = []
    dts = []

    # initial refinement
    # ------------------
    n_initial_refinements = ExperimentConfig.N_INITIAL_REFINEMENTS
    for _ in range(n_initial_refinements):
        marked = np.arange(elements.shape[0])
        coordinates, elements, boundaries, _ = \
            refinement.refineNVB(coordinates=coordinates,
                                 elements=elements,
                                 marked_elements=marked,
                                 boundary_conditions=boundaries)

    # solve exactly on the initial mesh
    solution, energy = solvers.solve_laplace(
        coordinates=coordinates,
        elements=elements,
        dirichlet=boundaries[0],
        neumann=np.array([]),
        f=f, g=None, uD=uD)
    energies.append(energy)
    n_elements.append(elements.shape[0])
    dts.append(0.)

    dump_object(obj=solution,
                path_to_file=base_results_path / Path('0/solution.pkl'))
    dump_object(obj=elements,
                path_to_file=base_results_path / Path('0/elements.pkl'))
    dump_object(obj=coordinates,
                path_to_file=base_results_path / Path('0/coordinates.pkl'))
    dump_object(obj=boundaries,
                path_to_file=base_results_path / Path('0/boundaries.pkl'))

    # ------------------------------------------------
    # variational adaptivity
    # ------------------------------------------------

    for n_refinement in np.arange(1, ExperimentConfig.N_REFINEMENTS + 1):
        start_time = time.process_time_ns()

        # compute all local energy gains
        # ------------------------------
        element_to_neighbours = get_element_to_neighbours(elements=elements)
        local_energy_gains = algo_4_1.get_all_local_enery_gains(
            coordinates=coordinates,
            elements=elements,
            boundaries=boundaries,
            current_iterate=solution,
            element_to_neighbours=element_to_neighbours,
            rhs_function=f, lamba_a=1)

        # mark elements to be refined, then refine
        # ---------------------------------------
        marked = markers.doerfler_marking(local_energy_gains, theta=THETA)
        coordinates, elements, boundaries, solution = refinement.refineNVB(
            coordinates=coordinates,
            elements=elements,
            marked_elements=(marked != 0),
            boundary_conditions=boundaries,
            to_embed=solution)

        # solve linear problem exactly on current mesh
        # --------------------------------------------
        solution, energy = solvers.solve_laplace(
            coordinates=coordinates,
            elements=elements,
            dirichlet=boundaries[0],
            neumann=np.array([]),
            f=f, g=None, uD=uD)

        end_time = time.process_time_ns()
        dts.append((end_time - start_time)*1e-9)

        dump_object(obj=solution, path_to_file=base_results_path /
                    Path(f'{n_refinement}/solution.pkl'))
        dump_object(obj=elements, path_to_file=base_results_path /
                    Path(f'{n_refinement}/elements.pkl'))
        dump_object(obj=coordinates, path_to_file=base_results_path /
                    Path(f'{n_refinement}/coordinates.pkl'))
        dump_object(obj=boundaries, path_to_file=base_results_path /
                    Path(f'{n_refinement}/boundaries.pkl'))

        energies.append(energy)
        n_elements.append(elements.shape[0])

    # ------------------------------------------------
    # wrap-up
    # ------------------------------------------------

    energies = np.array(energies)
    n_elements = np.array(n_elements)
    dts = np.array(dts)

    result = {
        'dt (s)': dts,
        'n_elements': n_elements,
        'energies': energies,
        'theta': THETA
    }

    dump_object(obj=pd.DataFrame(result),
                path_to_file=base_results_path / Path('results.pkl'))


# ----------------
# helper functions
# ----------------


def dump_object(obj, path_to_file: Path) -> None:
    path_to_file.parent.mkdir(parents=True, exist_ok=True)
    # save result as a pickle dump of pd.dataframe
    with open(path_to_file, "wb") as file:
        # Dump the DataFrame into the file using pickle
        pickle.dump(obj, file)


if __name__ == '__main__':
    main()
