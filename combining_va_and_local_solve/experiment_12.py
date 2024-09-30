import numpy as np
from p1afempy import io_helpers, refinement, solvers
from p1afempy.mesh import get_element_to_neighbours
from pathlib import Path
from variational_adaptivity import algo_4_1
from experiment_setup import f, uD
from load_save_dumps import dump_object
from scipy.sparse import csr_matrix
from variational_adaptivity.markers import doerfler_marking
import argparse
import tqdm
from scipy.sparse.linalg import cg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, required=True,
                        help="value of theta used in the Dörfler marking")
    parser.add_argument("--c", type=float, required=True,
                        help="if || u_h - u_h^n ||_a <= c dof^-1/2, then"
                        " VA kicks in")
    args = parser.parse_args()

    THETA = args.theta
    C = args.c

    max_n_updates = 500
    min_n_updates = 10

    # ------------------------------------------------
    # Setup
    # ------------------------------------------------
    np.random.seed(42)
    base_path = Path('data')
    path_to_elements = base_path / Path('elements.dat')
    path_to_coordinates = base_path / Path('coordinates.dat')
    path_to_dirichlet = base_path / Path('dirichlet.dat')

    base_results_path = (
        Path('results/experiment_9') /
        Path(f'theta-{THETA}_c-{C}'))

    coordinates, elements = io_helpers.read_mesh(
        path_to_coordinates=path_to_coordinates,
        path_to_elements=path_to_elements,
        shift_indices=False)
    boundaries = [io_helpers.read_boundary_condition(
        path_to_boundary=path_to_dirichlet,
        shift_indices=False)]

    # initial refinement
    # ------------------
    n_initial_refinements = 3
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

    # initializing the solution to random values
    current_iterate = np.random.rand(coordinates.shape[0])
    # forcing the boundary values to be zero, nevertheless
    current_iterate[np.unique(boundaries[0].flatten())] = 0.

    # number of refinement steps using variational adaptivity
    n_va_refinement_steps = 8
    for _ in range(n_va_refinement_steps):
        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        n_dofs = np.sum(free_nodes)

        # assembly of right hand side
        right_hand_side = solvers.get_right_hand_side(
            coordinates=coordinates,
            elements=elements,
            f=f)

        # assembly of the stiffness matrix
        stiffness_matrix = csr_matrix(solvers.get_stiffness_matrix(
            coordinates=coordinates,
            elements=elements))

        # -----------------------------------
        # compute and drop the exact solution
        # -----------------------------------
        solution, _ = solvers.solve_laplace(
            coordinates=coordinates,
            elements=elements,
            dirichlet=boundaries[0],
            neumann=np.array([]),
            f=f,
            g=None,
            uD=uD)

        dump_object(
            obj=solution, path_to_file=base_results_path /
            Path(f'{n_dofs}/exact_solution.pkl'))

        # ------------------------------------------------------------
        # compute all energy gains / local increments via local solver
        # ------------------------------------------------------------
        print('performing global update steps')
        for n_update in tqdm.tqdm(range(max_n_updates)):

            # ----------------------------------------------------
            # Perform a global line search in gradient's direction
            # ----------------------------------------------------
            residual_free_nodes = (
                right_hand_side[free_nodes]
                - stiffness_matrix[free_nodes, :][:, free_nodes].dot(
                    current_iterate[free_nodes]))

            # step size calculation
            numerator = residual_free_nodes.dot(residual_free_nodes)
            denominator = residual_free_nodes.dot(
                stiffness_matrix[free_nodes, :][:, free_nodes].dot(
                    residual_free_nodes))
            step_size = numerator / denominator

            # perform update
            current_iterate[free_nodes] += step_size * residual_free_nodes

            # dump snapshot of current current state
            dump_object(obj=current_iterate, path_to_file=base_results_path /
                        Path(f'{n_dofs}/{n_update+1}/solution.pkl'))
            dump_object(obj=elements, path_to_file=base_results_path /
                        Path(f'{n_dofs}/elements.pkl'))
            dump_object(obj=coordinates, path_to_file=base_results_path /
                        Path(f'{n_dofs}/coordinates.pkl'))
            dump_object(obj=boundaries, path_to_file=base_results_path /
                        Path(f'{n_dofs}/boundaries.pkl'))

            if n_update + 1 < min_n_updates:
                continue

            def energy_norm(u):
                return np.sqrt(u.dot(stiffness_matrix.dot(u)))

            energy_error_to_galerkin_solution = (
                energy_norm(current_iterate - solution)
            )

            if energy_error_to_galerkin_solution <= C / np.sqrt(n_dofs):
                break

        # --------------------------------------------------------------
        # compute all local energy gains via VA, based on exact solution
        # --------------------------------------------------------------
        element_to_neighbours = get_element_to_neighbours(elements=elements)
        print('computing all local energy gains with variational adaptivity')
        local_energy_differences_refine = algo_4_1.get_all_local_enery_gains(
            coordinates=coordinates,
            elements=elements,
            boundaries=boundaries,
            current_iterate=solution,
            element_to_neighbours=element_to_neighbours,
            uD=uD,
            rhs_function=f, lamba_a=1)

        # -------------------------------------
        # refine elements marked for refinement
        # -------------------------------------
        marked = doerfler_marking(
            input=local_energy_differences_refine, theta=THETA)

        coordinates, elements, boundaries, current_iterate = \
            refinement.refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked,
                boundary_conditions=boundaries,
                to_embed=current_iterate)


if __name__ == '__main__':
    main()
