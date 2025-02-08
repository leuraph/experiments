import numpy as np
import p1afempy
import argparse
from pathlib import Path
from configuration import f, uD
from triangle_cubature.cubature_rule import CubatureRuleEnum
from variational_adaptivity.algo_4_1 import get_all_local_enery_gains
from variational_adaptivity.markers import doerfler_marking
from iterative_methods.local_solvers \
    import LocalContextSolver
from scipy.sparse import csr_matrix
from tqdm import tqdm
from load_save_dumps import dump_object


def main() -> None:
    np.random.seed(42)

    # command line arguments
    # ----------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, required=True,
                        help="value of theta used in the Dörfler marking")
    parser.add_argument("--fudge", type=float, required=True,
                        help="fudge parameter used when deciding"
                        " whether to refine an element or locally solve")
    parser.add_argument("--miniter", type=int, required=True,
                        help="minimum number of full sweeps performed"
                        " on each mesh")
    args = parser.parse_args()

    THETA = args.theta
    FUDGE_PARAMETER = args.fudge
    MINITER = args.miniter

    # hard-coded variables
    max_n_dofs: int = int(1e7)
    n_initial_refinement_steps: int = 3

    # read the initial data
    # ---------------------
    base_path = Path('data')
    path_to_elements = base_path / Path('elements.dat')
    path_to_coordinates = base_path / Path('coordinates.dat')
    path_to_dirichlet = base_path / Path('dirichlet.dat')

    base_results_path = (
        Path('results/experiment_01') /
        Path(f'theta-{THETA}_fudge-{FUDGE_PARAMETER}_miniter-{MINITER}'))

    coordinates, elements = p1afempy.io_helpers.read_mesh(
        path_to_coordinates=path_to_coordinates,
        path_to_elements=path_to_elements,
        shift_indices=False)
    boundaries = [p1afempy.io_helpers.read_boundary_condition(
        path_to_boundary=path_to_dirichlet,
        shift_indices=False)]

    # perform initial refinement to get a decent mesh
    # -----------------------------------------------
    for _ in range(n_initial_refinement_steps):
        marked = np.arange(elements.shape[0])
        coordinates, elements, boundaries, _ = \
            p1afempy.refinement.refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked,
                boundary_conditions=boundaries)

    # solve problem on initial mesh
    # -----------------------------
    galerkin_solution, _ = p1afempy.solvers.solve_laplace(
        coordinates=coordinates,
        elements=elements,
        dirichlet=boundaries[0],
        neumann=np.array([]),
        f=f, g=None, uD=uD,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR)

    # initialize the iteration with Galerkin solution on initial mesh
    # ---------------------------------------------------------------
    current_iterate = np.copy(galerkin_solution)

    # perform initial VA by hand
    # --------------------------
    element_to_neighbours = p1afempy.mesh.get_element_to_neighbours(
        elements=elements)

    energy_gains = get_all_local_enery_gains(
        coordinates=coordinates,
        elements=elements,
        boundaries=boundaries,
        current_iterate=current_iterate,
        rhs_function=f,
        element_to_neighbours=element_to_neighbours,
        uD=uD,
        lamba_a=1.,
        return_local_solutions=False,
        display_progress_bar=True,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR)

    marked = doerfler_marking(input=energy_gains, theta=THETA)

    coordinates, elements, boundaries, current_iterate = \
        p1afempy.refinement.refineNVB(
            coordinates=coordinates,
            elements=elements,
            marked_elements=marked,
            boundary_conditions=boundaries,
            to_embed=current_iterate)

    n_vertices = coordinates.shape[0]
    n_elements = elements.shape[0]
    indices_of_free_nodes = np.setdiff1d(
        ar1=np.arange(n_vertices),
        ar2=np.unique(boundaries[0].flatten()))
    free_nodes = np.zeros(n_vertices, dtype=bool)
    free_nodes[indices_of_free_nodes] = 1
    n_dofs = np.sum(free_nodes)

    # loop until maximum number of degrees of freedom is reached
    # ----------------------------------------------------------
    while True:
        # recalculate mesh specific objects / parameters
        # ----------------------------------------------
        stiffness_matrix = csr_matrix(p1afempy.solvers.get_stiffness_matrix(
            coordinates=coordinates,
            elements=elements))
        right_hand_side = p1afempy.solvers.get_right_hand_side(
            coordinates=coordinates,
            elements=elements,
            f=f,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        local_context_solver = LocalContextSolver(
            elements=elements,
            free_nodes=free_nodes,
            lhs_matrix=stiffness_matrix,
            rhs_vector=right_hand_side)

        # calculate the Galerkin solution on the current mesh
        # ---------------------------------------------------
        galerkin_solution, _ = p1afempy.solvers.solve_laplace(
            coordinates=coordinates,
            elements=elements,
            dirichlet=boundaries[0],
            neumann=np.array([]),
            f=f, g=None, uD=uD,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        # perform iterations until stopping criterion is met
        # --------------------------------------------------
        n_iterations_done = 0
        n_solves_done: int = 0
        accumulated_energy_gain = 0.
        old_energy: float = None  # gets initializied when miniter reached
        energy_history: list[float] = []
        while True:
            # solving locally on each element, separately
            # -------------------------------------------
            local_energy_differences_solve = []
            local_increments = []

            print(f'performing full sweep of local solve for {n_dofs} DOFs')
            for k in tqdm(range(n_elements)):
                local_increment, local_energy_difference = \
                    local_context_solver.get_local_increment_and_energy_difference(
                        current_iterate=current_iterate,
                        element=k)
                n_solves_done += 1
                local_energy_differences_solve.append(local_energy_difference)
                local_increments.append(local_increment)

            local_energy_differences_solve = np.array(
                local_energy_differences_solve)
            local_increments = np.array(local_increments)

            # performing a global increment for all elements
            # ----------------------------------------------
            global_increment = np.zeros_like(current_iterate)

            # sorting such that local increments corresponding
            # to biggest energy gain come last
            energy_based_sorting = np.argsort(local_energy_differences_solve)
            sorted_elements = elements[energy_based_sorting]
            local_increments = local_increments[energy_based_sorting]

            # collect all local increments in a single vector
            # in a way that local increments corresponding to the
            # same node are overwritten by the one corresponding
            # to the bigger change in energy
            for element, local_increment in zip(
                    sorted_elements, local_increments):
                global_increment[element] = local_increment

            # performing the update
            current_iterate += global_increment
            n_iterations_done += 1

            # flushing the cache asap as, in the next sweep,
            # after performing the global update,
            # the global contribution has changed
            local_context_solver.flush_cache()

            if n_iterations_done < MINITER:
                continue

            if n_iterations_done == MINITER:
                # initialize old energy
                old_energy = get_energy(
                    current_iterate=current_iterate,
                    stiffness_matrix=stiffness_matrix,
                    right_hand_side=right_hand_side)
                continue

            current_energy = get_energy(
                current_iterate=current_iterate,
                stiffness_matrix=stiffness_matrix,
                right_hand_side=right_hand_side)

            energy_history.append(current_energy)

            current_energy_gain = old_energy - current_energy
            accumulated_energy_gain += current_energy_gain

            avg_de = accumulated_energy_gain / (n_iterations_done - MINITER)

            print(f'current energy gain  = {current_energy_gain}')
            print(f'averaged energy gain = {avg_de}')

            stopping_criterion_met = (
                current_energy_gain <
                FUDGE_PARAMETER * avg_de)

            if stopping_criterion_met:
                print(
                    'stopping criterion met, stopping iteration after'
                    f' {n_iterations_done} iterations')
                energy_history = np.array(energy_history)
                break

            old_energy = current_energy

        # drop all the data accumulated in the corresponding results directory
        # --------------------------------------------------------------------

        dump_object(obj=n_iterations_done, path_to_file=base_results_path /
                    Path(f'{n_dofs}/n_iterations_done.pkl'))
        dump_object(obj=current_iterate, path_to_file=base_results_path /
                    Path(f'{n_dofs}/last_iterate.pkl'))
        dump_object(obj=galerkin_solution, path_to_file=base_results_path /
                    Path(f'{n_dofs}/galerkin_solution.pkl'))

        dump_object(obj=energy_history, path_to_file=base_results_path /
                    Path(f'{n_dofs}/energy_history.pkl'))
        dump_object(obj=n_solves_done, path_to_file=base_results_path /
                    Path(f'{n_dofs}/n_solves_done.pkl'))

        dump_object(obj=elements, path_to_file=base_results_path /
                    Path(f'{n_dofs}/elements.pkl'))
        dump_object(obj=coordinates, path_to_file=base_results_path /
                    Path(f'{n_dofs}/coordinates.pkl'))
        dump_object(obj=boundaries, path_to_file=base_results_path /
                    Path(f'{n_dofs}/boundaries.pkl'))

        # perform element-based VA with the last iterate
        # ----------------------------------------------
        element_to_neighbours = p1afempy.mesh.get_element_to_neighbours(
            elements=elements)

        energy_gains = get_all_local_enery_gains(
            coordinates=coordinates,
            elements=elements,
            boundaries=boundaries,
            current_iterate=current_iterate,
            rhs_function=f,
            element_to_neighbours=element_to_neighbours,
            uD=uD,
            lamba_a=1.,
            return_local_solutions=False,
            display_progress_bar=True,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        marked = doerfler_marking(input=energy_gains, theta=THETA)

        coordinates, elements, boundaries, current_iterate = \
            p1afempy.refinement.refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked,
                boundary_conditions=boundaries,
                to_embed=current_iterate)

        # calculate the number of degrees of freedom on the new mesh
        # ----------------------------------------------------------
        n_vertices = coordinates.shape[0]
        n_elements = elements.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        n_dofs = np.sum(free_nodes)

        if n_dofs > max_n_dofs:
            break


def get_energy(
        current_iterate: np.ndarray,
        stiffness_matrix: csr_matrix,
        right_hand_side: np.ndarray) -> float:
    return 0.5 * current_iterate.dot(stiffness_matrix.dot(current_iterate)) \
        - current_iterate.dot(right_hand_side)


def get_energy_per_element(
        current_iterate: np.ndarray,
        elements: p1afempy.data_structures.ElementsType,
        coordinates: p1afempy.data_structures.CoordinatesType,
        cubature_rule: CubatureRuleEnum = CubatureRuleEnum.DAYTAYLOR) -> np.ndarray:
    virtual_element = np.array([[0, 1, 2]])
    energy_per_element = np.zeros(elements.shape[0])
    print('calculating energy contribution of each element...')
    for k, element in tqdm(enumerate(elements)):
        local_iterate = current_iterate[element]
        local_coordinates = coordinates[element]
        A = p1afempy.solvers.get_stiffness_matrix(
            coordinates=local_coordinates, elements=virtual_element)
        rhs = p1afempy.solvers.get_right_hand_side(
            coordinates=local_coordinates,
            elements=virtual_element,
            f=f, cubature_rule=cubature_rule)
        energy_on_current_element = (
            0.5 * local_iterate.dot(A.dot(local_iterate))
            - local_iterate.dot(rhs))
        energy_per_element[k] = energy_on_current_element
    return energy_per_element


if __name__ == '__main__':
    main()
