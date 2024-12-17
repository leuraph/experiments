import numpy as np
from p1afempy import io_helpers, refinement, solvers
from p1afempy.refinement import refineNVB_edge_based
from p1afempy.mesh import provide_geometric_data
from pathlib import Path
from configuration import f, uD
from load_save_dumps import dump_object
from scipy.sparse import csr_matrix
from variational_adaptivity.markers import doerfler_marking
import argparse
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
from matplotlib import cm
from custom_callback import ConvergedException, CustomCallBack


def calculate_energy(
        u: np.ndarray,
        lhs_matrix: np.ndarray,
        rhs_vector: np.ndarray) -> float:
    return 0.5 * u.dot(lhs_matrix.dot(u)) - rhs_vector.dot(u)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, required=True,
                        help="value of theta used in the Dörfler marking")
    parser.add_argument("--fudge", type=float, required=True,
                        help="if fudge * dE_CG < dE_EVA, then"
                        " CG on current mesh is stopped and EVA kicks in")
    args = parser.parse_args()

    THETA = args.theta
    FUDGE = args.fudge

    max_n_refinements = 2
    n_cg_steps = 5
    n_initial_refinements = 6

    # ------------------------------------------------
    # Setup
    # ------------------------------------------------
    np.random.seed(42)
    base_path = Path('data')
    path_to_elements = base_path / Path('elements_order1.dat')
    path_to_coordinates = base_path / Path('coordinates.dat')
    path_to_dirichlet = base_path / Path('dirichlet.dat')

    base_results_path = (
        Path('results/experiment_03_improved') /
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

    for n_refinement in range(max_n_refinements):
        # re-setup as the mesh has changed
        # --------------------------------
        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        n_dofs = np.sum(free_nodes)

        # compute exact galerkin solution on current mesh
        solution, _ = solvers.solve_laplace(
            coordinates=coordinates,
            elements=elements,
            dirichlet=boundaries[0],
            neumann=np.array([]),
            f=f,
            g=None,
            uD=uD)

        right_hand_side = solvers.get_right_hand_side(
            coordinates=coordinates,
            elements=elements,
            f=f)

        # assembly of the stiffness matrix
        stiffness_matrix = csr_matrix(solvers.get_stiffness_matrix(
            coordinates=coordinates,
            elements=elements))

        # ------------------------------
        # Perform CG on the current mesh
        # ------------------------------
        # assembly of right hand side
        custom_callback = CustomCallBack(
            batch_size=n_cg_steps,
            elements=elements,
            coordinates=coordinates,
            boundaries=boundaries,
            energy_of_initial_guess=calculate_energy(
                u=current_iterate,
                lhs_matrix=stiffness_matrix,
                rhs_vector=right_hand_side),
            eva_energy_gain_of_initial_guess=0.0,
            energy_gains_of_initial_guess=np.zeros(np.sum(free_nodes)),
            fudge=FUDGE)

        try:
            current_iterate[free_nodes], _ = cg(
                A=stiffness_matrix[free_nodes, :][:, free_nodes],
                b=right_hand_side[free_nodes],
                x0=current_iterate[free_nodes],
                rtol=1e-100,
                callback=custom_callback)
        except ConvergedException as conv:
            current_iterate = conv.last_iterate
            energy_gains = conv.energy_gains
            print("CG stopped!")

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
                    Path(f'{n_dofs}/solution.pkl'))

        # in the last iteration, do not consider the possibility of refinement
        if n_refinement == max_n_refinements - 1:
            break

        # dörfler based on EVA
        # --------------------
        marked_edges = np.zeros(custom_callback.edges.shape[0], dtype=int)
        marked_non_boundary_egdes = doerfler_marking(
            input=energy_gains, theta=THETA)
        marked_edges[custom_callback.free_edges] = marked_non_boundary_egdes

        element_to_edges, edge_to_nodes, boundaries_to_edges =\
            provide_geometric_data(elements=elements, boundaries=boundaries)

        coordinates, elements, boundaries, current_iterate = \
            refineNVB_edge_based(
                coordinates=coordinates,
                elements=elements,
                boundary_conditions=boundaries,
                element2edges=element_to_edges,
                edge_to_nodes=edge_to_nodes,
                boundaries_to_edges=boundaries_to_edges,
                edge2newNode=marked_edges,
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
    # ax.set_zticks([0, 0.02, 0.04, 0.06])

    # Show and save the plot
    # fig.savefig(out_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
