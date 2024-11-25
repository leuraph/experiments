import numpy as np
from p1afempy import io_helpers, refinement, solvers
from p1afempy.mesh import provide_geometric_data, show_mesh, get_local_patch_edge_based
from p1afempy.solvers import get_right_hand_side, get_stiffness_matrix
from p1afempy.refinement import refineNVB_edge_based, refine_single_edge
from variational_adaptivity.markers import doerfler_marking
from pathlib import Path
from utils import shuffle_elements, distort_coordinates
from configuration import uD, f
from ismember import is_row_in
import pickle
import argparse
from scipy.sparse import csr_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm


def main() -> None:
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, required=True,
                        help="value of theta used in the Dörfler marking")
    args = parser.parse_args()

    THETA = args.theta

    # ------------------------------------------------
    # Setup
    # ------------------------------------------------
    base_path = Path('data')
    # TODO if this order is what we want, then remove the other one and rename
    path_to_elements = base_path / Path('elements_order1.dat')
    path_to_coordinates = base_path / Path('coordinates.dat')
    path_to_dirichlet = base_path / Path('dirichlet.dat')

    base_results_path = Path('results/experiment_03') / Path(f'theta_{THETA}')

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
    # marking only non-boundary coordinates for jiggling
    # all_coordinates_indices = np.arange(coordinates.shape[0])
    # coordinates_on_boundary = np.isin(all_coordinates_indices, boundaries[0])
    # marked_coordinates = np.logical_not(coordinates_on_boundary)
    # # jiggle the initial mesh's non-boundary coordinates
    # delta = 1./2**(n_initial_refinements+1)
    # coordinates = distort_coordinates(coordinates=coordinates,
    #                                   delta=delta, marked=marked_coordinates)
    # # shuffle initial mesh's elements
    # elements = shuffle_elements(elements=elements)

    # solve exactly on the initial mesh
    solution, _ = solvers.solve_laplace(
        coordinates=coordinates,
        elements=elements,
        dirichlet=boundaries[0],
        neumann=np.array([]),
        f=f, g=None, uD=uD)

    n_vertices = coordinates.shape[0]
    indices_of_free_nodes = np.setdiff1d(
        ar1=np.arange(n_vertices),
        ar2=np.unique(boundaries[0].flatten()))
    free_nodes = np.zeros(n_vertices, dtype=bool)
    free_nodes[indices_of_free_nodes] = 1
    n_dofs = np.sum(free_nodes)

    dump_object(obj=solution,
                path_to_file=base_results_path / Path(
                    f'{n_dofs}/solution.pkl'))
    dump_object(obj=elements,
                path_to_file=base_results_path / Path(
                    f'{n_dofs}/elements.pkl'))
    dump_object(obj=coordinates,
                path_to_file=base_results_path / Path(
                    f'{n_dofs}/coordinates.pkl'))
    dump_object(obj=boundaries,
                path_to_file=base_results_path / Path(
                    f'{n_dofs}/boundaries.pkl'))

    # ------------------------------------------------
    # variational adaptivity
    # ------------------------------------------------

    n_refinements = 20
    for _ in range(n_refinements):

        element_to_edges, edge_to_nodes, boundaries_to_edges =\
            provide_geometric_data(elements=elements, boundaries=boundaries)

        # computing global terms before loop
        stiffness_matrix = get_stiffness_matrix(
            coordinates=coordinates, elements=elements)
        rhs_vector = get_right_hand_side(
            coordinates=coordinates, elements=elements, f=f)
        L_1 = rhs_vector.dot(solution)
        A_11 = solution.dot(stiffness_matrix.dot(solution))

        # compute all local energy gains
        # ------------------------------
        edge_to_nodes_flipped = np.column_stack(
            [edge_to_nodes[:, 1], edge_to_nodes[:, 0]])
        boundary = np.logical_or(
            is_row_in(edge_to_nodes, boundaries[0]),
            is_row_in(edge_to_nodes_flipped, boundaries[0])
        )
        non_boundary = np.logical_not(boundary)
        n_non_boundary_edges = np.sum(non_boundary)
        marked_edges = np.zeros(edge_to_nodes.shape[0], dtype=int)
        energy_gains = np.zeros(n_non_boundary_edges, dtype=float)
        non_boundary_edges = edge_to_nodes[non_boundary]

        print(f'Calculating all energy gains for {n_dofs} DOFs...')
        for k, non_boundary_edge in enumerate(tqdm(non_boundary_edges)):
            local_elements, local_coordinates, \
                local_iterate, local_edge_indices = get_local_patch_edge_based(
                    elements=elements,
                    coordinates=coordinates,
                    current_iterate=solution,
                    edge=non_boundary_edge)
            tmp_local_coordinates, tmp_local_elements, tmp_local_solution =\
                refine_single_edge(
                    coordinates=local_coordinates,
                    elements=local_elements,
                    edge=local_edge_indices,
                    to_embed=local_iterate)
            tmp_stiffness_matrix = csr_matrix(get_stiffness_matrix(
                coordinates=tmp_local_coordinates,
                elements=tmp_local_elements))
            tmp_rhs_vector = get_right_hand_side(
                coordinates=tmp_local_coordinates,
                elements=tmp_local_elements, f=f)

            # building the local 2x2 system
            A_12 = tmp_stiffness_matrix.dot(tmp_local_solution)[-1]
            A_22 = tmp_stiffness_matrix[-1, -1]

            L_2 = tmp_rhs_vector[-1]

            detA = (A_11 * A_22 - A_12 * A_12)

            alpha = (A_22 * L_1 - A_12 * L_2)/detA
            beta = (-A_12 * L_2 + A_11 * L_2)/detA

            dE = 0.5*(
                (alpha-1)**2 * A_11
                + 2.*(alpha-1)*beta*A_12
                + beta**2 * A_22)

            energy_gains[k] = dE

        # mark elements to be refined, then refine
        # ---------------------------------------
        marked_non_boundary_egdes = doerfler_marking(
            input=energy_gains, theta=THETA)
        marked_edges[non_boundary] = marked_non_boundary_egdes

        coordinates, elements, boundaries, _ = refineNVB_edge_based(
            coordinates=coordinates,
            elements=elements,
            boundary_conditions=boundaries,
            element2edges=element_to_edges,
            edge_to_nodes=edge_to_nodes,
            boundaries_to_edges=boundaries_to_edges,
            edge2newNode=marked_edges)

        # shuffle refined mesh's elements
        # elements = shuffle_elements(elements=elements)

        # solve linear problem exactly on current mesh
        # --------------------------------------------
        solution, _ = solvers.solve_laplace(
            coordinates=coordinates,
            elements=elements,
            dirichlet=boundaries[0],
            neumann=np.array([]),
            f=f, g=None, uD=uD)

        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        n_dofs = np.sum(free_nodes)

        dump_object(obj=solution, path_to_file=base_results_path /
                    Path(f'{n_dofs}/solution.pkl'))
        dump_object(obj=elements, path_to_file=base_results_path /
                    Path(f'{n_dofs}/elements.pkl'))
        dump_object(obj=coordinates, path_to_file=base_results_path /
                    Path(f'{n_dofs}/coordinates.pkl'))
        dump_object(obj=boundaries, path_to_file=base_results_path /
                    Path(f'{n_dofs}/boundaries.pkl'))


def dump_object(obj, path_to_file: Path) -> None:
    path_to_file.parent.mkdir(parents=True, exist_ok=True)
    # save result as a pickle dump of pd.dataframe
    with open(path_to_file, "wb") as file:
        # Dump the DataFrame into the file using pickle
        pickle.dump(obj, file)


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
