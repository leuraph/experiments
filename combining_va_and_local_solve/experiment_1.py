import numpy as np
from p1afempy import io_helpers, refinement, solvers
from p1afempy.mesh import get_element_to_neighbours
from pathlib import Path
from variational_adaptivity import algo_4_1, markers
from experiment_setup import f
from load_save_dumps import dump_object
from iterative_methods.local_solvers \
    import LocalContextSolver
from scipy.sparse import csr_matrix
from variational_adaptivity.markers import doerfler_marking
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
    path_to_elements = base_path / Path('elements.dat')
    path_to_coordinates = base_path / Path('coordinates.dat')
    path_to_dirichlet = base_path / Path('dirichlet.dat')

    base_results_path = Path('results/experiment_1') / Path(f'theta_{THETA}')

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

    # ------------------------------------------------
    # variational adaptivity + Local Solvers
    # ------------------------------------------------

    # initializing the solution to zero
    current_iterate = np.zeros(coordinates.shape[0])

    n_full_sweeps = 10
    for _ in range(n_full_sweeps):
        # -------------------------------------
        # Adapting local Solver to current mesh
        # -------------------------------------
        n_vertices = coordinates.shape[0]
        n_elements = elements.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1

        # assembly of right hand side
        right_hand_side = solvers.get_right_hand_side(
            coordinates=coordinates,
            elements=elements,
            f=f)

        # assembly of the stiffness matrix
        stiffness_matrix = csr_matrix(solvers.get_stiffness_matrix(
            coordinates=coordinates,
            elements=elements))

        local_context_solver = LocalContextSolver(
            elements=elements,
            free_nodes=free_nodes,
            lhs_matrix=stiffness_matrix,
            rhs_vector=right_hand_side,
            simultaneous_solve=True)

        # ------------------------------------------------------------
        # compute all energy gains / local increments via local solver
        # ------------------------------------------------------------
        local_energy_differences_context = []
        local_increments = []

        for k in range(n_elements):
            local_increment, local_energy_difference = \
                local_context_solver.get_local_increment_and_energy_difference(
                    current_iterate=current_iterate,
                    element=k)
            local_energy_differences_context.append(local_energy_difference)
            local_increments.append(local_increment)

        local_energy_differences_context = np.array(
            local_energy_differences_context)
        local_increments = np.array(local_increments)

        # -------------------------------------
        # compute all local energy gains via VA
        # -------------------------------------
        element_to_neighbours = get_element_to_neighbours(elements=elements)
        local_energy_differences_va = algo_4_1.get_all_local_enery_gains(
            coordinates=coordinates,
            elements=elements,
            boundaries=boundaries,
            current_iterate=current_iterate,
            element_to_neighbours=element_to_neighbours,
            rhs_function=f, lamba_a=1)

        # ---------------------------------------------------
        # deciding where to refine and where to locally solve
        # ---------------------------------------------------
        # if the energy difference is equal, we prefer locally solving
        # instead of adding more expensive degrees of freedom
        solve = local_energy_differences_va <= local_energy_differences_context
        refine = local_energy_differences_va > local_energy_differences_context

        bigger_energy_differences = np.zeros(n_elements)
        bigger_energy_differences[solve] = local_energy_differences_context[solve]
        bigger_energy_differences[refine] = local_energy_differences_va[refine]

        marked = doerfler_marking(input=bigger_energy_differences, theta=THETA)
        solve = solve & marked
        refine = refine & marked

        # -----------------------------------------------------------------
        # performing a global increment for the elements marked for solving
        # -----------------------------------------------------------------
        global_increment = np.zeros_like(current_iterate)

        reduced_local_increments = local_increments[solve]
        reduced_elements = elements[solve]
        reduced_energy_differences_solve = local_energy_differences_context[solve]

        # sorting such that local increments corresponding
        # to biggest energy gain come last
        energy_based_sorting = np.argsort(reduced_energy_differences_solve)
        reduced_elements = reduced_elements[energy_based_sorting]
        reduced_local_increments = reduced_local_increments[
            energy_based_sorting]

        # collect all local increments in a single vector
        # in a way that local increments corresponding to the
        # same node are overwritten by the one corresponding
        # to the bigger change in energy
        for element, local_increment in zip(
                reduced_elements, reduced_local_increments):
            global_increment[element] = local_increment

        # performing the update
        current_iterate += global_increment

        # -------------------------------------
        # refine elements marked for refinement
        # -------------------------------------
        coordinates, elements, boundaries, solution = refinement.refineNVB(
            coordinates=coordinates,
            elements=elements,
            marked_elements=refine,
            boundary_conditions=boundaries,
            to_embed=current_iterate)

        # # dump the current iterate
        # dump_object(obj=solution, path_to_file=base_results_path /
        #             Path(f'{n_refinement}/solution.pkl'))
        # dump_object(obj=elements, path_to_file=base_results_path /
        #             Path(f'{n_refinement}/elements.pkl'))
        # dump_object(obj=coordinates, path_to_file=base_results_path /
        #             Path(f'{n_refinement}/coordinates.pkl'))
        # dump_object(obj=boundaries, path_to_file=base_results_path /
        #             Path(f'{n_refinement}/boundaries.pkl'))


if __name__ == '__main__':
    main()
