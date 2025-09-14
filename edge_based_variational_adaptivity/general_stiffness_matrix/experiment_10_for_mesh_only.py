import numpy as np
from p1afempy import refinement
from p1afempy.solvers import \
    get_general_stiffness_matrix, get_mass_matrix, get_right_hand_side
from triangle_cubature.cubature_rule import CubatureRuleEnum
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
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
from scipy.sparse import csr_matrix, diags
from problems import get_problem_4
from p1afempy.data_structures import CoordinatesType, ElementsType, BoundaryType


def get_initial_mesh() -> tuple[CoordinatesType, ElementsType, BoundaryType]:
    # mesh
    # ----
    coordinates = np.array(
        [[0.9,0. ],
        [1. ,0. ],
        [1. ,0.1],
        [0.9,0.1],
        [0.8,0. ],
        [1. ,0.2],
        [0.8,0.1],
        [0.9,0.2],
        [0.7,0. ],
        [1. ,0.3],
        [0.7,0.1],
        [0.8,0.2],
        [0.9,0.3],
        [0.6,0. ],
        [1. ,0.4],
        [0.6,0.1],
        [0.7,0.2],
        [0.8,0.3],
        [0.9,0.4],
        [0.5,0. ],
        [1. ,0.5],
        [0.5,0.1],
        [0.6,0.2],
        [0.7,0.3],
        [0.8,0.4],
        [0.9,0.5],
        [0.4,0. ],
        [1. ,0.6],
        [0.4,0.1],
        [0.5,0.2],
        [0.6,0.3],
        [0.7,0.4],
        [0.8,0.5],
        [0.9,0.6],
        [0.3,0. ],
        [1. ,0.7],
        [0.3,0.1],
        [0.4,0.2],
        [0.5,0.3],
        [0.6,0.4],
        [0.7,0.5],
        [0.8,0.6],
        [0.9,0.7],
        [0.2,0. ],
        [1. ,0.8],
        [0.2,0.1],
        [0.3,0.2],
        [0.4,0.3],
        [0.5,0.4],
        [0.6,0.5],
        [0.7,0.6],
        [0.8,0.7],
        [0.9,0.8],
        [0.1,0. ],
        [1. ,0.9],
        [0.1,0.1],
        [0.2,0.2],
        [0.3,0.3],
        [0.4,0.4],
        [0.5,0.5],
        [0.6,0.6],
        [0.7,0.7],
        [0.8,0.8],
        [0.9,0.9],
        [0. ,0. ],
        [1. ,1. ],
        [0. ,0.1],
        [0.1,0.2],
        [0.2,0.3],
        [0.3,0.4],
        [0.4,0.5],
        [0.5,0.6],
        [0.6,0.7],
        [0.7,0.8],
        [0.8,0.9],
        [0.9,1. ],
        [0. ,0.2],
        [0.1,0.3],
        [0.2,0.4],
        [0.3,0.5],
        [0.4,0.6],
        [0.5,0.7],
        [0.6,0.8],
        [0.7,0.9],
        [0.8,1. ],
        [0. ,0.3],
        [0.1,0.4],
        [0.2,0.5],
        [0.3,0.6],
        [0.4,0.7],
        [0.5,0.8],
        [0.6,0.9],
        [0.7,1. ],
        [0. ,0.4],
        [0.1,0.5],
        [0.2,0.6],
        [0.3,0.7],
        [0.4,0.8],
        [0.5,0.9],
        [0.6,1. ],
        [0. ,0.5],
        [0.1,0.6],
        [0.2,0.7],
        [0.3,0.8],
        [0.4,0.9],
        [0.5,1. ],
        [0. ,0.6],
        [0.1,0.7],
        [0.2,0.8],
        [0.3,0.9],
        [0.4,1. ],
        [0. ,0.7],
        [0.1,0.8],
        [0.2,0.9],
        [0.3,1. ],
        [0. ,0.8],
        [0.1,0.9],
        [0.2,1. ],
        [0. ,0.9],
        [0.1,1. ],
        [0. ,1. ]])
    elements = np.array(
        [[  0,  1,  2],
        [  0,  3,  2],
        [  4,  0,  3],
        [  3,  2,  5],
        [  4,  6,  3],
        [  3,  7,  5],
        [  8,  4,  6],
        [  6,  3,  7],
        [  7,  5,  9],
        [  8, 10,  6],
        [  6, 11,  7],
        [  7, 12,  9],
        [ 13,  8, 10],
        [ 10,  6, 11],
        [ 11,  7, 12],
        [ 12,  9, 14],
        [ 13, 15, 10],
        [ 10, 16, 11],
        [ 11, 17, 12],
        [ 12, 18, 14],
        [ 19, 13, 15],
        [ 15, 10, 16],
        [ 16, 11, 17],
        [ 17, 12, 18],
        [ 18, 14, 20],
        [ 19, 21, 15],
        [ 15, 22, 16],
        [ 16, 23, 17],
        [ 17, 24, 18],
        [ 18, 25, 20],
        [ 26, 19, 21],
        [ 21, 15, 22],
        [ 22, 16, 23],
        [ 23, 17, 24],
        [ 24, 18, 25],
        [ 25, 20, 27],
        [ 26, 28, 21],
        [ 21, 29, 22],
        [ 22, 30, 23],
        [ 23, 31, 24],
        [ 24, 32, 25],
        [ 25, 33, 27],
        [ 34, 26, 28],
        [ 28, 21, 29],
        [ 29, 22, 30],
        [ 30, 23, 31],
        [ 31, 24, 32],
        [ 32, 25, 33],
        [ 33, 27, 35],
        [ 34, 36, 28],
        [ 28, 37, 29],
        [ 29, 38, 30],
        [ 30, 39, 31],
        [ 31, 40, 32],
        [ 32, 41, 33],
        [ 33, 42, 35],
        [ 43, 34, 36],
        [ 36, 28, 37],
        [ 37, 29, 38],
        [ 38, 30, 39],
        [ 39, 31, 40],
        [ 40, 32, 41],
        [ 41, 33, 42],
        [ 42, 35, 44],
        [ 43, 45, 36],
        [ 36, 46, 37],
        [ 37, 47, 38],
        [ 38, 48, 39],
        [ 39, 49, 40],
        [ 40, 50, 41],
        [ 41, 51, 42],
        [ 42, 52, 44],
        [ 53, 43, 45],
        [ 45, 36, 46],
        [ 46, 37, 47],
        [ 47, 38, 48],
        [ 48, 39, 49],
        [ 49, 40, 50],
        [ 50, 41, 51],
        [ 51, 42, 52],
        [ 52, 44, 54],
        [ 53, 55, 45],
        [ 45, 56, 46],
        [ 46, 57, 47],
        [ 47, 58, 48],
        [ 48, 59, 49],
        [ 49, 60, 50],
        [ 50, 61, 51],
        [ 51, 62, 52],
        [ 52, 63, 54],
        [ 64, 53, 55],
        [ 55, 45, 56],
        [ 56, 46, 57],
        [ 57, 47, 58],
        [ 58, 48, 59],
        [ 59, 49, 60],
        [ 60, 50, 61],
        [ 61, 51, 62],
        [ 62, 52, 63],
        [ 63, 54, 65],
        [ 64, 66, 55],
        [ 55, 67, 56],
        [ 56, 68, 57],
        [ 57, 69, 58],
        [ 58, 70, 59],
        [ 59, 71, 60],
        [ 60, 72, 61],
        [ 61, 73, 62],
        [ 62, 74, 63],
        [ 63, 75, 65],
        [ 66, 55, 67],
        [ 67, 56, 68],
        [ 68, 57, 69],
        [ 69, 58, 70],
        [ 70, 59, 71],
        [ 71, 60, 72],
        [ 72, 61, 73],
        [ 73, 62, 74],
        [ 74, 63, 75],
        [ 66, 76, 67],
        [ 67, 77, 68],
        [ 68, 78, 69],
        [ 69, 79, 70],
        [ 70, 80, 71],
        [ 71, 81, 72],
        [ 72, 82, 73],
        [ 73, 83, 74],
        [ 74, 84, 75],
        [ 76, 67, 77],
        [ 77, 68, 78],
        [ 78, 69, 79],
        [ 79, 70, 80],
        [ 80, 71, 81],
        [ 81, 72, 82],
        [ 82, 73, 83],
        [ 83, 74, 84],
        [ 76, 85, 77],
        [ 77, 86, 78],
        [ 78, 87, 79],
        [ 79, 88, 80],
        [ 80, 89, 81],
        [ 81, 90, 82],
        [ 82, 91, 83],
        [ 83, 92, 84],
        [ 85, 77, 86],
        [ 86, 78, 87],
        [ 87, 79, 88],
        [ 88, 80, 89],
        [ 89, 81, 90],
        [ 90, 82, 91],
        [ 91, 83, 92],
        [ 85, 93, 86],
        [ 86, 94, 87],
        [ 87, 95, 88],
        [ 88, 96, 89],
        [ 89, 97, 90],
        [ 90, 98, 91],
        [ 91, 99, 92],
        [ 93, 86, 94],
        [ 94, 87, 95],
        [ 95, 88, 96],
        [ 96, 89, 97],
        [ 97, 90, 98],
        [ 98, 91, 99],
        [ 93,100, 94],
        [ 94,101, 95],
        [ 95,102, 96],
        [ 96,103, 97],
        [ 97,104, 98],
        [ 98,105, 99],
        [100, 94,101],
        [101, 95,102],
        [102, 96,103],
        [103, 97,104],
        [104, 98,105],
        [100,106,101],
        [101,107,102],
        [102,108,103],
        [103,109,104],
        [104,110,105],
        [106,101,107],
        [107,102,108],
        [108,103,109],
        [109,104,110],
        [106,111,107],
        [107,112,108],
        [108,113,109],
        [109,114,110],
        [111,107,112],
        [112,108,113],
        [113,109,114],
        [111,115,112],
        [112,116,113],
        [113,117,114],
        [115,112,116],
        [116,113,117],
        [115,118,116],
        [116,119,117],
        [118,116,119],
        [118,120,119]], dtype=int)
    dirichlet = np.array(
        [[  0,  1],
        [  0,  4],
        [  1,  2],
        [  2,  5],
        [  4,  8],
        [  5,  9],
        [  8, 13],
        [  9, 14],
        [ 13, 19],
        [ 14, 20],
        [ 19, 26],
        [ 20, 27],
        [ 26, 34],
        [ 27, 35],
        [ 34, 43],
        [ 35, 44],
        [ 43, 53],
        [ 44, 54],
        [ 53, 64],
        [ 54, 65],
        [ 64, 66],
        [ 65, 75],
        [ 66, 76],
        [ 75, 84],
        [ 76, 85],
        [ 84, 92],
        [ 85, 93],
        [ 92, 99],
        [ 93,100],
        [ 99,105],
        [100,106],
        [105,110],
        [106,111],
        [110,114],
        [111,115],
        [114,117],
        [115,118],
        [117,119],
        [118,120],
        [119,120]], dtype=int)
    boundaries = [dirichlet]

    return coordinates, elements, boundaries


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

    plt.show()


def main() -> None:

    MINITER = 10
    BATCHSIZE = 2
    THETA = 0.5
    FUDGE = 0.1

    n_max_dofs = 1e5

    # ------------------------------------------------
    # Setup
    # ------------------------------------------------
    np.random.seed(42)

    base_results_path = (
        Path('results/experiment_10_mesh') /
        Path(
            f'theta-{THETA}_fudge-{FUDGE}_'
            f'miniter-{MINITER}_batchsize-{BATCHSIZE}'))

    coordinates, elements, oundaries = get_initial_mesh()

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
        f=get_problem_4().f,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR)
    stiffness_matrix = get_general_stiffness_matrix(
        coordinates=coordinates,
        elements=elements,
        a_11=get_problem_4().a_11,
        a_12=get_problem_4().a_12,
        a_21=get_problem_4().a_21,
        a_22=get_problem_4().a_22,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR)
    mass_matrix = get_mass_matrix(
        coordinates=coordinates,
        elements=elements)
    lhs_matrix = csr_matrix(mass_matrix + stiffness_matrix)

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
        f=get_problem_4().f,
        a_11=get_problem_4().a_11,
        a_12=get_problem_4().a_12,
        a_21=get_problem_4().a_21,
        a_22=get_problem_4().a_22,
        c=get_problem_4().c,
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
            a_11=get_problem_4().a_11,
            a_12=get_problem_4().a_12,
            a_21=get_problem_4().a_21,
            a_22=get_problem_4().a_22,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)
        mass_matrix = get_mass_matrix(
            coordinates=coordinates,
            elements=elements)

        lhs_matrix = csr_matrix(general_stiffness_matrix + mass_matrix)
        rhs_vector = get_right_hand_side(
            coordinates=coordinates,
            elements=elements,
            f=get_problem_4().f,
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
            lhs_reduced = lhs_matrix[free_nodes, :][:, free_nodes]
            diagonal = lhs_reduced.diagonal()
            M = diags(diagonals=1./diagonal)
            current_iterate[free_nodes], _ = cg(
                A=lhs_matrix[free_nodes, :][:, free_nodes],
                b=rhs_vector[free_nodes],
                x0=current_iterate[free_nodes],
                M=M,
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
            f=get_problem_4().f,
            a_11=get_problem_4().a_11,
            a_12=get_problem_4().a_12,
            a_21=get_problem_4().a_21,
            a_22=get_problem_4().a_22,
            c=get_problem_4().c,
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
