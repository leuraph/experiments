"""
This experiment considers the Problem
-\Delta u(x) = 1,
on L-shape with homogeneous boundary conditions.
"""
import numpy as np
from p1afempy import refinement
from p1afempy.solvers import \
    get_general_stiffness_matrix, get_mass_matrix, get_right_hand_side
from triangle_cubature.cubature_rule import CubatureRuleEnum
from scipy.sparse.linalg import spsolve
from custom_callback import EnergyTailOffAveragedCustomCallback
import argparse
from scipy.sparse.linalg import cg
from pathlib import Path
from load_save_dumps import dump_object
from p1afempy.mesh import provide_geometric_data
from ismember import is_row_in
from variational_adaptivity.edge_based_variational_adaptivity import \
    get_energy_gains
from variational_adaptivity.markers import doerfler_marking
from p1afempy.refinement import refineNVB_edge_based
from custom_callback import ConvergedException
from scipy.sparse import csr_matrix
from problems import get_problem_5


def main() -> None:

    MINITER = 10
    BATCHSIZE = 2
    THETA = 0.5
    FUDGE = 0.1

    n_max_dofs = 1e5
    n_initial_refinements = 2

    # ------------------------------------------------
    # Setup
    # ------------------------------------------------
    np.random.seed(42)

    base_results_path = (
        Path('results/experiment_13_mesh') /
        Path(
            f'theta-{THETA}_fudge-{FUDGE}_'
            f'miniter-{MINITER}_batchsize-{BATCHSIZE}'))

    # mesh
    # ----
    coordinates = np.array([
        [-1, -1],
        [0, -1],
        [-1, 0],
        [0, 0],
        [1, 0],
        [-1, 1],
        [0, 1],
        [1, 1]
    ])
    elements = np.array([
        [3, 0, 1],
        [0, 3, 2],
        [6, 2, 3],
        [7, 3, 4],
        [2, 6, 5],
        [3, 7, 6]
    ])
    dirichlet = np.array([
        [0, 1],
        [1, 3],
        [3, 4],
        [4, 7],
        [7, 6],
        [6, 5],
        [5, 2],
        [2, 0]
    ])
    boundaries = [dirichlet]

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
    # calculating free nodes on the initial mesh
    # ------------------------------------------
    n_vertices = coordinates.shape[0]
    indices_of_free_nodes = np.setdiff1d(
        ar1=np.arange(n_vertices),
        ar2=np.unique(boundaries[0].flatten()))
    free_nodes = np.zeros(n_vertices, dtype=bool)
    free_nodes[indices_of_free_nodes] = 1
    n_dofs = np.sum(free_nodes)

    rhs_vector = get_right_hand_side(
        coordinates=coordinates,
        elements=elements,
        f=get_problem_5().f,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR)
    stiffness_matrix = get_general_stiffness_matrix(
        coordinates=coordinates,
        elements=elements,
        a_11=get_problem_5().a_11,
        a_12=get_problem_5().a_12,
        a_21=get_problem_5().a_21,
        a_22=get_problem_5().a_22,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR)
    mass_matrix = get_mass_matrix(
        coordinates=coordinates,
        elements=elements)
    c: float = get_problem_5().c
    lhs_matrix = csr_matrix(c * mass_matrix + stiffness_matrix)

    galerkin_solution = np.zeros(n_vertices)
    galerkin_solution[free_nodes] = spsolve(
        A=lhs_matrix[free_nodes, :][:, free_nodes],
        b=rhs_vector[free_nodes])

    # initializing the iteration with Galerkin
    # solution on first mesh
    current_iterate = np.copy(galerkin_solution)

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
        f=get_problem_5().f,
        a_11=get_problem_5().a_11,
        a_12=get_problem_5().a_12,
        a_21=get_problem_5().a_21,
        a_22=get_problem_5().a_22,
        c=get_problem_5().c,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR,
        verbose=True)

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
        # reset as the mesh has changed
        # -----------------------------
        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        n_dofs = np.sum(free_nodes)

        general_stiffness_matrix = get_general_stiffness_matrix(
            coordinates=coordinates,
            elements=elements,
            a_11=get_problem_5().a_11,
            a_12=get_problem_5().a_12,
            a_21=get_problem_5().a_21,
            a_22=get_problem_5().a_22,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)
        mass_matrix = get_mass_matrix(
            coordinates=coordinates,
            elements=elements)
        c: float = get_problem_5().c
        lhs_matrix = csr_matrix(general_stiffness_matrix + c*mass_matrix)
        rhs_vector = get_right_hand_side(
            coordinates=coordinates,
            elements=elements,
            f=get_problem_5().f,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        # compute the Galerkin solution on current mesh
        # ---------------------------------------------
        galerkin_solution = np.zeros(n_vertices)
        galerkin_solution[free_nodes] = spsolve(
            A=lhs_matrix[free_nodes, :][:, free_nodes],
            b=rhs_vector[free_nodes])

        # perform CG on the current mesh
        # ------------------------------
        custom_callback = EnergyTailOffAveragedCustomCallback(
            batch_size=BATCHSIZE,
            min_n_iterations_per_mesh=MINITER,
            elements=elements,
            coordinates=coordinates,
            boundaries=boundaries,
            fudge=FUDGE,
            lhs_matrix=lhs_matrix,
            rhs_vector=rhs_vector)

        cg_converged = False

        try:
            current_iterate[free_nodes], _ = cg(
                A=lhs_matrix[free_nodes, :][:, free_nodes],
                b=rhs_vector[free_nodes],
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

        # perform edge-based variational adaptivity
        # -----------------------------------------
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
            f=get_problem_5().f,
            a_11=get_problem_5().a_11,
            a_12=get_problem_5().a_12,
            a_21=get_problem_5().a_21,
            a_22=get_problem_5().a_22,
            c=get_problem_5().c,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR,
            verbose=True)

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
