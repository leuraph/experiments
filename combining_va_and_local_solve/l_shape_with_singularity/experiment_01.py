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
    n_initial_refinement_steps: int = 5

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

    # loop until maximum number of degrees of freedom is reached
    # ----------------------------------------------------------
    while True:
        # recalculate mesh specific objects / parameters
        # ----------------------------------------------
        n_vertices = coordinates.shape[0]
        n_elements = elements.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        n_dofs = np.sum(free_nodes)

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

        # perform iterations until stopping criterion is met
        # --------------------------------------------------

        # drop all the data accumulated in the corresponding results directory
        # --------------------------------------------------------------------

        # perform EVA with the last iterate
        # ---------------------------------

        # calculate the number of degrees of freedom on the new mesh
        # ----------------------------------------------------------
        n_dofs: int = int(1e8)
        if n_dofs > max_n_dofs:
            break


if __name__ == '__main__':
    main()
