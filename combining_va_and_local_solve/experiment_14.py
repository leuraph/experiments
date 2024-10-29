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

    max_n_updates = 1000
    max_n_loops = 10
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
    for k in range(max_n_loops):
        # --------------------
        # Split into two paths
        # --------------------

        # ------------------------------
        # Perform CG on the current mesh
        # ------------------------------

        # create a copy of the current_iterate

        print('performing n global CG steps on current mesh')
        current_iterate[free_nodes], _ = cg(
            A=stiffness_matrix[free_nodes, :][:, free_nodes],
            b=right_hand_side[free_nodes],
            x0=current_iterate[free_nodes],
            maxiter=n_cg_steps,
            rtol=1e-100)

        # dump current iterate

        if k == max_n_loops - 1:
            break

        # compute energy drop of cg per element -> dE_cg

        # perform va with current_iterate -> dE_va

        old_iterate = copy(current_iterate)

        # compare dE_cg with dE_va and decide whether to refine an element


if __name__ == '__main__':
    main()
