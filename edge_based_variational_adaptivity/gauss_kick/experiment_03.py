import numpy as np
from p1afempy import io_helpers, refinement, solvers
from p1afempy.refinement import refineNVB, refine_single_edge, refineNVB_edge_based
from p1afempy.mesh import provide_geometric_data, get_local_patch_edge_based, show_mesh
from p1afempy.solvers import get_stiffness_matrix, get_right_hand_side
from pathlib import Path
from configuration import f, uD
from load_save_dumps import dump_object
from scipy.sparse import csr_matrix
from variational_adaptivity.markers import doerfler_marking
import argparse
from scipy.sparse.linalg import cg
from copy import copy
from ismember import is_row_in
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm


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

    max_n_sweeps = 20
    n_cg_steps = 5

    # ------------------------------------------------
    # Setup
    # ------------------------------------------------
    np.random.seed(42)
    base_path = Path('data')
    path_to_elements = base_path / Path('elements_order1.dat')
    path_to_coordinates = base_path / Path('coordinates.dat')
    path_to_dirichlet = base_path / Path('dirichlet.dat')

    base_results_path = (
        Path('results/experiment_03') /
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
    n_initial_refinements = 4
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
        # assembly of right hand side
        right_hand_side = solvers.get_right_hand_side(
            coordinates=coordinates,
            elements=elements,
            f=f)

        # assembly of the stiffness matrix
        stiffness_matrix = csr_matrix(solvers.get_stiffness_matrix(
            coordinates=coordinates,
            elements=elements))

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

        # in the last iteration, do not consider the possibility of refinement
        if n_sweep == max_n_sweeps - 1:
            break

        old_iterate = copy(current_iterate)

        current_iterate[free_nodes], _ = cg(
            A=stiffness_matrix[free_nodes, :][:, free_nodes],
            b=right_hand_side[free_nodes],
            x0=current_iterate[free_nodes],
            maxiter=1,
            rtol=1e-100)

        # compute energy drop of one global cg step  -> dE_cg
        old_energy = calculate_energy(
            u=old_iterate,
            lhs_matrix=stiffness_matrix,
            rhs_vector=right_hand_side)
        current_energy = calculate_energy(
            u=current_iterate,
            lhs_matrix=stiffness_matrix,
            rhs_vector=right_hand_side)
        dE_cg = old_energy - current_energy

        # --------------------------------------
        # perform EVA with old_iterate -> dE_EVA
        # --------------------------------------
        element_to_edges, edge_to_nodes, boundaries_to_edges = \
            provide_geometric_data(
                elements=elements,
                boundaries=boundaries)

        # computing global terms before loop
        L_1 = right_hand_side.dot(old_iterate)
        A_11 = old_iterate.dot(stiffness_matrix.dot(old_iterate))

        n_boundaries = edge_to_nodes.shape[0]

        edge_to_nodes_flipped = np.column_stack(
            [edge_to_nodes[:, 1], edge_to_nodes[:, 0]])
        boundary = np.logical_or(
            is_row_in(edge_to_nodes, boundaries[0]),
            is_row_in(edge_to_nodes_flipped, boundaries[0])
        )
        non_boundary = np.logical_not(boundary)
        non_boundary_edges = edge_to_nodes[non_boundary]
        n_non_boundary_edges = non_boundary_edges.shape[0]
        marked_edges = np.zeros(edge_to_nodes.shape[0], dtype=int)
        energy_gains = np.zeros(n_non_boundary_edges, dtype=float)

        # we get a new value for each new edge
        values_on_new_edges = np.zeros(n_boundaries)
        values_on_new_edges_non_boundary = np.zeros(n_non_boundary_edges)

        for k, non_boundary_edge in enumerate(tqdm(non_boundary_edges)):

            local_elements, local_coordinates, \
                local_iterate, local_edge_indices = get_local_patch_edge_based(
                    elements=elements,
                    coordinates=coordinates,
                    current_iterate=old_iterate,
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
            print(detA)
            beta = (-A_12 * L_1 + A_11 * L_2)/detA

            print(f'A_11 = {A_11}')
            print(f'A_12 = {A_12}')
            print(f'A_22 = {A_22}')
            print(f'L_1 = {L_1}')
            print(f'L_2 = {L_2}')

            dE = 0.5*(
                (alpha-1)**2 * A_11
                + 2.*(alpha-1)*beta*A_12
                + beta**2 * A_22)

            energy_gains[k] = dE
            values_on_new_edges_non_boundary[k] = beta

        values_on_new_edges[non_boundary] = \
            values_on_new_edges_non_boundary
        print(np.max(values_on_new_edges))

        old_iterate_after_eva = np.hstack(
            [old_iterate, values_on_new_edges])

        # mark all elements for refinement
        marked_elements = np.arange(elements.shape[0])
        new_coordinates, new_elements, _, _ =\
            refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked_elements,
                boundary_conditions=boundaries)

        new_stiffness_matrix = csr_matrix(
            get_stiffness_matrix(
                coordinates=new_coordinates,
                elements=new_elements))
        new_right_hand_side = get_right_hand_side(
            coordinates=new_coordinates, elements=new_elements, f=f)

        show_solution(
            coordinates=new_coordinates,
            solution=old_iterate_after_eva)
        show_solution(
            coordinates=coordinates, solution=current_iterate)

        eva_energy = calculate_energy(
            u=old_iterate_after_eva,
            lhs_matrix=new_stiffness_matrix,
            rhs_vector=new_right_hand_side)
        dE_eva = old_energy - eva_energy

        # deciding whether to refine based on global energy gains
        # -------------------------------------------------------
        # if the energy difference is equal, we prefer locally solving
        # instead of adding more "expensive" degrees of freedom
        print(f'dE_eva = {dE_eva}')
        print(f'dE_cg = {dE_cg}')
        refine = FUDGE * dE_cg < dE_eva

        if not refine:
            continue

        # dörfler based on EVA
        # ---------------------------------------
        marked_non_boundary_egdes = doerfler_marking(
            input=energy_gains, theta=THETA)
        marked_edges[non_boundary] = marked_non_boundary_egdes

        coordinates, elements, boundaries, current_iterate = \
            refineNVB_edge_based(
                coordinates=coordinates,
                elements=elements,
                boundary_conditions=boundaries,
                element2edges=element_to_edges,
                edge_to_nodes=edge_to_nodes,
                boundaries_to_edges=boundaries_to_edges,
                edge2newNode=marked_edges)


def calculate_energy(u: np.ndarray, lhs_matrix: np.ndarray, rhs_vector: np.ndarray) -> float:
    return 0.5 * u.dot(lhs_matrix.dot(u)) - rhs_vector.dot(u)


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
