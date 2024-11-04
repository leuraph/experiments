import numpy as np
from p1afempy import io_helpers, refinement, solvers
from p1afempy.mesh import get_element_to_neighbours
from p1afempy.refinement import refineNVB
from pathlib import Path
from variational_adaptivity import algo_4_1
from experiment_setup import f, uD
from load_save_dumps import dump_object
from scipy.sparse import csr_matrix
from variational_adaptivity.markers import doerfler_marking
import argparse
from scipy.sparse.linalg import cg
from copy import copy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, required=True,
                        help="value of theta used in the Dörfler marking")
    parser.add_argument("--fudge", type=float, required=True,
                        help="if dE_n < fudge * sum_{k=1}^n dE_k/n, then"
                        " CG on current mesh is stopped and VA kicks in")
    args = parser.parse_args()

    THETA = args.theta
    FUDGE = args.fudge

    max_n_sweeps = 20
    n_cg_steps = 5

    # ------------------------------------------------
    # Setup
    # ------------------------------------------------
    np.random.seed(42)
    base_path = Path('data')
    path_to_elements = base_path / Path('elements.dat')
    path_to_coordinates = base_path / Path('coordinates.dat')
    path_to_dirichlet = base_path / Path('dirichlet.dat')

    base_results_path = (
        Path('results/experiment_14') /
        Path(f'theta-{THETA}_fudge-{FUDGE}'))

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

    # initializing the solution to random values
    current_iterate = np.random.rand(coordinates.shape[0])
    # forcing the boundary values to be zero, nevertheless
    current_iterate[np.unique(boundaries[0].flatten())] = 0.

    # calculating free nodes on the initial mesh
    # ------------------------------------------
    n_vertices = coordinates.shape[0]
    indices_of_free_nodes = np.setdiff1d(
        ar1=np.arange(n_vertices),
        ar2=np.unique(boundaries[0].flatten()))
    free_nodes = np.zeros(n_vertices, dtype=bool)
    free_nodes[indices_of_free_nodes] = 1
    n_dofs = np.sum(free_nodes)

    # initial exact galerkin solution
    # -------------------------------
    # assembly of right hand side
    right_hand_side = solvers.get_right_hand_side(
        coordinates=coordinates,
        elements=elements,
        f=f)

    # assembly of the stiffness matrix
    stiffness_matrix = csr_matrix(solvers.get_stiffness_matrix(
        coordinates=coordinates,
        elements=elements))

    # compute exact galerkin solution
    solution, _ = solvers.solve_laplace(
        coordinates=coordinates,
        elements=elements,
        dirichlet=boundaries[0],
        neumann=np.array([]),
        f=f,
        g=None,
        uD=uD)

    # dump initial mesh and initial exact galerkin solution
    # -----------------------------------------------------
    dump_object(obj=elements, path_to_file=base_results_path /
                Path(f'{n_dofs}/elements.pkl'))
    dump_object(obj=coordinates, path_to_file=base_results_path /
                Path(f'{n_dofs}/coordinates.pkl'))
    dump_object(obj=boundaries, path_to_file=base_results_path /
                Path(f'{n_dofs}/boundaries.pkl'))
    dump_object(
        obj=solution, path_to_file=base_results_path /
        Path(f'{n_dofs}/exact_solution.pkl'))
    # -----------------------------------------------------

    old_iterate = copy(current_iterate)
    for n_sweep in range(max_n_sweeps):
        # re-setup as the mesh might have changed
        # ------------------------------------------
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

        # compute exact galerkin solution on current mesh
        solution, _ = solvers.solve_laplace(
            coordinates=coordinates,
            elements=elements,
            dirichlet=boundaries[0],
            neumann=np.array([]),
            f=f,
            g=None,
            uD=uD)

        # ------------------------------
        # Perform CG on the current mesh
        # ------------------------------
        print(f'performing {n_cg_steps} global CG steps on current mesh')
        current_iterate[free_nodes], _ = cg(
            A=stiffness_matrix[free_nodes, :][:, free_nodes],
            b=right_hand_side[free_nodes],
            x0=current_iterate[free_nodes],
            maxiter=n_cg_steps,
            rtol=1e-100)

        # dump the current state
        # ----------------------
        dump_object(obj=elements, path_to_file=base_results_path /
                    Path(f'{n_dofs}/elements.pkl'))
        dump_object(obj=coordinates, path_to_file=base_results_path /
                    Path(f'{n_dofs}/coordinates.pkl'))
        dump_object(obj=boundaries, path_to_file=base_results_path /
                    Path(f'{n_dofs}/boundaries.pkl'))
        dump_object(
            obj=solution, path_to_file=base_results_path /
            Path(f'{n_dofs}/exact_solution.pkl'))
        dump_object(obj=current_iterate, path_to_file=base_results_path /
                    Path(f'{n_dofs}/{n_sweep+1}/solution.pkl'))

        if n_sweep == max_n_sweeps - 1:
            break

        old_iterate = copy(current_iterate)

        current_iterate[free_nodes], _ = cg(
            A=stiffness_matrix[free_nodes, :][:, free_nodes],
            b=right_hand_side[free_nodes],
            x0=current_iterate[free_nodes],
            maxiter=1,
            rtol=1e-100)

        # compute energy drop of cg per element -> dE_cg
        n_elements = elements.shape[0]
        dE_cg = np.zeros(n_elements)
        for k in range(n_elements):
            local_nodes = elements[k]

            # indices of non-local nodes
            non_local_nodes = np.setdiff1d(
                ar1=np.arange(n_vertices),
                ar2=np.unique(local_nodes.flatten()))

            local_cg_solution = current_iterate[local_nodes]
            non_local_old_iterate = old_iterate[non_local_nodes]

            numerator = (
                right_hand_side[non_local_nodes].dot(non_local_old_iterate)
                - local_cg_solution.dot(
                    stiffness_matrix[local_nodes, :][:, non_local_nodes].dot(
                        non_local_old_iterate
                    )))
            denominator = non_local_old_iterate.dot(
                stiffness_matrix[non_local_nodes, :][:, non_local_nodes].dot(
                    non_local_old_iterate
                ))
            alpha = numerator / denominator

            locally_updated_u = copy(old_iterate)
            locally_updated_u[local_nodes] = local_cg_solution
            locally_updated_u[non_local_nodes] *= alpha
            E_old = calculate_energy(
                u=old_iterate,
                lhs_matrix=stiffness_matrix, rhs_vector=right_hand_side)
            E_current = calculate_energy(
                u=locally_updated_u,
                lhs_matrix=stiffness_matrix, rhs_vector=right_hand_side)
            dE_local = E_old - E_current  # positive, if old energy was higher
            dE_cg[k] = dE_local

        # perform va with old_iterate -> dE_va
        dE_va = algo_4_1.get_all_local_enery_gains(
            coordinates=coordinates,
            elements=elements,
            boundaries=boundaries,
            current_iterate=old_iterate,
            rhs_function=f,
            element_to_neighbours=get_element_to_neighbours(elements),
            uD=uD, lamba_a=1)

        # deciding where to refine based on local energy gains
        # ----------------------------------------------------
        # if the energy difference is equal, we prefer locally solving
        # instead of adding more "expensive" degrees of freedom
        refine = (
            dE_va
            > (FUDGE * dE_cg))
        solve = np.logical_not(refine)

        bigger_energy_drops = np.zeros_like(dE_cg)
        bigger_energy_drops[solve] = FUDGE * dE_cg[solve]
        bigger_energy_drops[refine] = dE_va[refine]
        marked = doerfler_marking(
            input=bigger_energy_drops,
            theta=THETA)
        refine = marked & refine

        coordinates, elements, boundaries, current_iterate = refineNVB(
            coordinates=coordinates,
            elements=elements,
            marked_elements=marked,
            boundary_conditions=boundaries,
            to_embed=current_iterate)


def calculate_energy(u: np.ndarray, lhs_matrix: np.ndarray, rhs_vector: np.ndarray) -> float:
    return 0.5 * u.dot(lhs_matrix.dot(u)) - rhs_vector.dot(u)


if __name__ == '__main__':
    main()
