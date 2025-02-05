import numpy as np
from pathlib import Path
from p1afempy import io_helpers
from p1afempy.refinement import refineNVB
from p1afempy.solvers import solve_laplace
import argparse
from configuration import f, uD
from triangle_cubature.cubature_rule import CubatureRuleEnum


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

    coordinates, elements = io_helpers.read_mesh(
        path_to_coordinates=path_to_coordinates,
        path_to_elements=path_to_elements,
        shift_indices=False)
    boundaries = [io_helpers.read_boundary_condition(
        path_to_boundary=path_to_dirichlet,
        shift_indices=False)]

    # perform initial refinement to get a decent mesh
    # -----------------------------------------------
    for _ in range(n_initial_refinement_steps):
        marked = np.arange(elements.shape[0])
        coordinates, elements, boundaries, _ = \
            refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked,
                boundary_conditions=boundaries)

    # solve problem on initial mesh
    # -----------------------------
    galerkin_solution, _ = solve_laplace(
        coordinates=coordinates,
        elements=elements,
        dirichlet=boundaries[0],
        neumann=np.array([]),
        f=f, g=None, uD=uD,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR)

    # initialize the iteration with Galerkin solution on initial mesh
    # ---------------------------------------------------------------

    # perform initial EVA by hand
    # ---------------------------

    # loop until maximum number of degrees of freedom is reached
    # ----------------------------------------------------------
    while True:
        # recalculate mesh specific objects / parameters
        # ----------------------------------------------

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
