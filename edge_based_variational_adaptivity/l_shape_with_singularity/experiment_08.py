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
from custom_callback import ConvergedException, AriolisCustomCallback
from ismember import is_row_in
from variational_adaptivity.edge_based_variational_adaptivity import \
    get_energy_gains
from triangle_cubature.cubature_rule import CubatureRuleEnum


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
                        help="additional fudge parameter "
                        "in Ariolis stopping criterion")
    parser.add_argument("--miniter", type=int, required=True,
                        help="minimum number of iterations on each mesh")
    parser.add_argument("--delay", type=int, required=True,
                        help="delay parameter in the "
                        "Hestenes-Stiefel Estimator")
    args = parser.parse_args()

    MINITER = args.miniter
    DELAY = args.delay
    THETA = args.theta
    FUDGE = args.fudge

    n_max_dofs = 1e6
    n_initial_refinements = 5

    # ------------------------------------------------
    # Setup
    # ------------------------------------------------
    np.random.seed(42)
    base_path = Path('data')
    path_to_elements = base_path / Path('elements.dat')
    path_to_coordinates = base_path / Path('coordinates.dat')
    path_to_dirichlet = base_path / Path('dirichlet.dat')

    base_results_path = (
        Path('results/experiment_08') /
        Path(
            f'theta-{THETA}_fudge-{FUDGE}_'
            f'miniter-{MINITER}_delay-{DELAY}'))

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

    # initial exact galerkin solution
    # -------------------------------
    galerkin_solution, _ = solvers.solve_laplace(
        coordinates=coordinates,
        elements=elements,
        dirichlet=boundaries[0],
        neumann=np.array([]),
        f=f,
        g=None,
        uD=uD,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR)

    # initializing the iteration with Galerkin
    # solution on first mesh
    current_iterate = np.copy(galerkin_solution)

    # calculating free nodes on the initial mesh
    # ------------------------------------------
    n_vertices = coordinates.shape[0]
    indices_of_free_nodes = np.setdiff1d(
        ar1=np.arange(n_vertices),
        ar2=np.unique(boundaries[0].flatten()))
    free_nodes = np.zeros(n_vertices, dtype=bool)
    free_nodes[indices_of_free_nodes] = 1
    n_dofs = np.sum(free_nodes)

    # dump initial mesh and initial Galerkin solution
    # -----------------------------------------------
    dump_object(obj=elements, path_to_file=base_results_path /
                Path(f'{n_dofs}/elements.pkl'))
    dump_object(obj=coordinates, path_to_file=base_results_path /
                Path(f'{n_dofs}/coordinates.pkl'))
    dump_object(obj=boundaries, path_to_file=base_results_path /
                Path(f'{n_dofs}/boundaries.pkl'))
    dump_object(
        obj=galerkin_solution, path_to_file=base_results_path /
        Path(f'{n_dofs}/galerkin_solution.pkl'))
    dump_object(
        obj=current_iterate, path_to_file=base_results_path /
        Path(f'{n_dofs}/last_iterate.pkl'))
    dump_object(
        obj=int(0), path_to_file=base_results_path /
        Path(f'{n_dofs}/n_iterations_done.pkl'))
    # -----------------------------------------------------

    # perform first refinement by hand
    # --------------------------------
    # (non-boundary) edges
    _, edge_to_nodes, _ = \
        provide_geometric_data(
            elements=elements,
            boundaries=boundaries)

    edge_to_nodes_flipped = np.column_stack(
        [edge_to_nodes[:, 1], edge_to_nodes[:, 0]])
    boundary = np.logical_or(
        is_row_in(edge_to_nodes, boundaries[0]),
        is_row_in(edge_to_nodes_flipped, boundaries[0])
    )
    non_boundary = np.logical_not(boundary)
    edges = edge_to_nodes
    non_boundary_edges = edge_to_nodes[non_boundary]

    # free nodes / edges
    n_vertices = coordinates.shape[0]
    indices_of_free_nodes = np.setdiff1d(
        ar1=np.arange(n_vertices),
        ar2=np.unique(boundaries[0].flatten()))
    free_nodes = np.zeros(n_vertices, dtype=bool)
    free_nodes[indices_of_free_nodes] = 1
    free_edges = non_boundary  # integer array (holding actual indices)
    free_nodes = free_nodes  # boolean array

    energy_gains = get_energy_gains(
        coordinates=coordinates,
        elements=elements,
        non_boundary_edges=non_boundary_edges,
        current_iterate=current_iterate,
        f=f,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR,
        verbose=False)

    # dörfler based on EVA
    marked_edges = np.zeros(edges.shape[0], dtype=int)
    marked_non_boundary_egdes = doerfler_marking(
        input=energy_gains, theta=THETA)
    marked_edges[free_edges] = marked_non_boundary_egdes

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
    # --------------------------------

    while True:
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
        galerkin_solution, _ = solvers.solve_laplace(
            coordinates=coordinates,
            elements=elements,
            dirichlet=boundaries[0],
            neumann=np.array([]),
            f=f,
            g=None,
            uD=uD,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        right_hand_side = solvers.get_right_hand_side(
            coordinates=coordinates,
            elements=elements,
            f=f,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        # assembly of the stiffness matrix
        stiffness_matrix = csr_matrix(solvers.get_stiffness_matrix(
            coordinates=coordinates,
            elements=elements))

        # ------------------------------
        # Perform CG on the current mesh
        # ------------------------------
        # assembly of right hand side
        custom_callback = AriolisCustomCallback(
            batch_size=1,
            min_n_iterations_per_mesh=MINITER,
            elements=elements,
            coordinates=coordinates,
            boundaries=boundaries,
            delay=DELAY,
            fudge=FUDGE,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        cg_converged = False
        try:
            current_iterate[free_nodes], _ = cg(
                A=stiffness_matrix[free_nodes, :][:, free_nodes],
                b=right_hand_side[free_nodes],
                x0=current_iterate[free_nodes],
                rtol=1e-100,
                callback=custom_callback)
        except ConvergedException as conv:
            cg_converged = True
            current_iterate = conv.last_iterate
            energy_history = np.array(conv.energy_history)
            n_iterations_done = conv.n_iterations_done
            print(f"CG stopped after {conv.n_iterations_done} iterations!")

        if not cg_converged:
            raise RuntimeError('CG failed to converge. Stopping immediately.')

        # dump the current state
        # ----------------------
        dump_object(obj=elements, path_to_file=base_results_path /
                    Path(f'{n_dofs}/elements.pkl'))
        dump_object(obj=coordinates, path_to_file=base_results_path /
                    Path(f'{n_dofs}/coordinates.pkl'))
        dump_object(obj=boundaries, path_to_file=base_results_path /
                    Path(f'{n_dofs}/boundaries.pkl'))
        dump_object(obj=energy_history, path_to_file=base_results_path /
                    Path(f'{n_dofs}/energy_history.pkl'))
        dump_object(obj=n_iterations_done, path_to_file=base_results_path /
                    Path(f'{n_dofs}/n_iterations_done.pkl'))
        dump_object(
            obj=galerkin_solution, path_to_file=base_results_path /
            Path(f'{n_dofs}/galerkin_solution.pkl'))
        dump_object(obj=current_iterate, path_to_file=base_results_path /
                    Path(f'{n_dofs}/last_iterate.pkl'))

        # stop right before refining if maximum number of DOFs is reached
        if n_dofs >= n_max_dofs:
            print(
                f'Maximum number of DOFs ({n_max_dofs})'
                'reached, stopping iteration.')
            break

        _, edge_to_nodes, _ = \
            provide_geometric_data(
                elements=elements,
                boundaries=boundaries)

        edge_to_nodes_flipped = np.column_stack(
            [edge_to_nodes[:, 1], edge_to_nodes[:, 0]])
        boundary = np.logical_or(
            is_row_in(edge_to_nodes, boundaries[0]),
            is_row_in(edge_to_nodes_flipped, boundaries[0])
        )
        non_boundary = np.logical_not(boundary)
        edges = edge_to_nodes
        non_boundary_edges = edge_to_nodes[non_boundary]

        # free nodes / edges
        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        free_edges = non_boundary  # integer array (holding actual indices)
        free_nodes = free_nodes  # boolean array

        energy_gains = get_energy_gains(
            coordinates=coordinates,
            elements=elements,
            non_boundary_edges=non_boundary_edges,
            current_iterate=current_iterate,
            f=f,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR,
            verbose=True,
            parallel=False)

        # dörfler based on EVA
        # --------------------
        marked_edges = np.zeros(edges.shape[0], dtype=int)
        marked_non_boundary_egdes = doerfler_marking(
            input=energy_gains, theta=THETA)
        marked_edges[free_edges] = marked_non_boundary_egdes

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


if __name__ == '__main__':
    main()
