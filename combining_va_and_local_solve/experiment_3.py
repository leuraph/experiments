import numpy as np
from p1afempy import io_helpers, refinement, solvers
from p1afempy.mesh import get_element_to_neighbours
from pathlib import Path
from variational_adaptivity import algo_4_1
from experiment_setup import f, uD
from load_save_dumps import dump_object
from iterative_methods.local_solvers \
    import LocalContextSolver
from scipy.sparse import csr_matrix
from variational_adaptivity.markers import doerfler_marking
import argparse
from p1afempy.mesh import show_mesh
import matplotlib.pyplot as plt
from matplotlib import cm
import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, required=True,
                        help="value of theta used in the Dörfler marking")
    parser.add_argument("--sweeps", type=int, required=True,
                        help="number of full sweeps before any refinement"
                        " takes place")
    args = parser.parse_args()

    THETA = args.theta
    N_FULL_SWEEPS = args.sweeps

    # ------------------------------------------------
    # Setup
    # ------------------------------------------------
    np.random.seed(42)
    base_path = Path('data')
    path_to_elements = base_path / Path('elements.dat')
    path_to_coordinates = base_path / Path('coordinates.dat')
    path_to_dirichlet = base_path / Path('dirichlet.dat')

    base_results_path = (
        Path('results/experiment_3') /
        Path(f'theta-{THETA}'))

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
            rhs_vector=right_hand_side)

        # ------------------------------------------------------------
        # compute all energy gains / local increments via local solver
        # ------------------------------------------------------------
        print('performing full sweeps of local solves')
        for n_sweep in tqdm.tqdm(range(N_FULL_SWEEPS)):
            local_energy_differences_solve = []
            local_increments = []

            # solving locally on each element, separately
            for k in range(n_elements):
                local_increment, local_energy_difference = \
                    local_context_solver.get_local_increment_and_energy_difference(
                        current_iterate=current_iterate,
                        element=k)
                local_energy_differences_solve.append(local_energy_difference)
                local_increments.append(local_increment)

            # transforming lists to np.ndarray's
            local_energy_differences_solve = np.array(
                local_energy_differences_solve)
            local_increments = np.array(local_increments)

            # ----------------------------------------------
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

            # flushing the cache asap as, in the next full sweep,
            # after performing the global update,
            # the global contribution has changed
            local_context_solver.flush_cache()

            # dump snapshot of current current state
            n_dofs = np.sum(free_nodes)

            dump_object(obj=current_iterate, path_to_file=base_results_path /
                        Path(f'{n_dofs}/{n_sweep+1}/solution.pkl'))
            dump_object(obj=elements, path_to_file=base_results_path /
                        Path(f'{n_dofs}/elements.pkl'))
            dump_object(obj=coordinates, path_to_file=base_results_path /
                        Path(f'{n_dofs}/coordinates.pkl'))
            dump_object(obj=boundaries, path_to_file=base_results_path /
                        Path(f'{n_dofs}/boundaries.pkl'))

        # compute and drop the exact solution
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

        # --------------------------------------------------------------
        # compute all local energy gains via VA, based on exact solution
        # --------------------------------------------------------------
        element_to_neighbours = get_element_to_neighbours(elements=elements)
        print('computing all local energy gains with variational adaptivity')
        local_energy_differences_va = algo_4_1.get_all_local_enery_gains(
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
            input=local_energy_differences_va, theta=THETA)

        coordinates, elements, boundaries, current_iterate = \
            refinement.refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked,
                boundary_conditions=boundaries,
                to_embed=current_iterate)


def show_solution(coordinates, solution):
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 12

    x_coords, y_coords = zip(*coordinates)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points with scalar values as colors (adjust colormap as needed)
    _ = ax.plot_trisurf(x_coords, y_coords, solution, linewidth=0.2,
                        antialiased=True, cmap=cm.viridis)
    # Add labels to the axes
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    #ax.set_zticks([0, 0.02, 0.04, 0.06])

    # Show and save the plot
    # fig.savefig(out_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
