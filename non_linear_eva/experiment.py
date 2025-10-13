from p1afempy.solvers import \
    integrate_composition_nonlinear_with_fem, \
        get_load_vector_of_composition_nonlinear_with_fem, \
            get_general_stiffness_matrix, get_right_hand_side
from p1afempy import refinement
from p1afempy.mesh import provide_geometric_data
from ismember import is_row_in
from p1afempy.data_structures import \
    BoundaryConditionType, CoordinatesType, ElementsType, BoundaryType
from triangle_cubature.cubature_rule import CubatureRuleEnum
import numpy as np
from scipy.optimize import fmin_cg
from scipy.sparse import csr_matrix
from show_solution import show_solution
from custom_callback import ConvergedException, EnergyTailOffAveragedCustomCallback
import argparse
from variational_adaptivity.edge_based_variational_adaptivity import get_energy_gains_nonlinear
from variational_adaptivity.markers import doerfler_marking
from p1afempy.refinement import refineNVB_edge_based

def a_11(r: CoordinatesType) -> np.ndarray:
    n_vertices = r.shape[0]
    return - np.ones(n_vertices, dtype=float)

def a_22(r: CoordinatesType) -> np.ndarray:
    n_vertices = r.shape[0]
    return - np.ones(n_vertices, dtype=float)

def a_12(r: CoordinatesType) -> np.ndarray:
    n_vertices = r.shape[0]
    return np.zeros(n_vertices, dtype=float)

def a_21(r: CoordinatesType) -> np.ndarray:
    n_vertices = r.shape[0]
    return np.zeros(n_vertices, dtype=float)

def phi(tau: float) -> float:
    return np.exp(tau)

def Phi(u: float) -> float:
    return np.exp(u)

def right_hand_side(r: CoordinatesType) -> np.ndarray:
    x, y = r[:, 0], r[:, 1]
    return -2.0*x*(x - 1) - 2.0*y*(y - 1) + np.exp(x*y*(x - 1)*(y - 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fudge", type=float, required=True)
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--miniter", type=int, required=True,
                        help="minimum number of iterations on each mesh")
    parser.add_argument("--batchsize", type=int, required=True,
                        help="minimum number of iterations on each mesh")
    args = parser.parse_args()

    MINITER = args.miniter
    FUDGE = args.fudge
    BATCHSIZE = args.batchsize
    THETA = args.theta

    # generating a reasonable mesh
    # ----------------------------
    coordinates = np.array([
        [0., 0.],
        [1., 0.],
        [1., 1.],
        [0., 1.]
    ])
    elements = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])
    dirichlet = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0]
    ])
    boundaries = [dirichlet]

    # initial refinement
    n_refinements = 6
    for _ in range(n_refinements):
        marked_elements = np.arange(elements.shape[0])
        coordinates, elements, boundaries, _ = refinement.refineNVB(
            coordinates=coordinates,
            elements=elements,
            marked_elements=marked_elements,
            boundary_conditions=boundaries)
    n_coordinates = coordinates.shape[0]
    indices_of_free_nodes = np.setdiff1d(
        ar1=np.arange(n_coordinates),
        ar2=np.unique(boundaries[0].flatten()))
    free_nodes = np.zeros(n_coordinates, dtype=bool)
    free_nodes[indices_of_free_nodes] = 1
    n_dof = np.sum(free_nodes)
    print(f'DOF = {n_dof}')

    # initializing with random values
    # NOTE that [Glowinski, 2013, remark 2.10] suggests u=0 as initial guess,
    # when there is no information available
    np.random.seed(42)
    random_values = np.random.rand(n_dof)
    initial_guess = np.zeros(n_coordinates, dtype=float)
    initial_guess[free_nodes] = random_values

    # midpoint suffices as we consider laplace operator
    stiffness_matrix = csr_matrix(get_general_stiffness_matrix(
        coordinates=coordinates,
        elements=elements,
        a_11=a_11, a_12=a_12, a_21=a_21, a_22=a_22,
        cubature_rule=CubatureRuleEnum.MIDPOINT))

    right_hand_side_vector = get_right_hand_side(
        coordinates=coordinates,
        elements=elements,
        f=right_hand_side,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR)

    def DJ(current_iterate: np.ndarray) -> np.ndarray:

        load_vector_phi = get_load_vector_of_composition_nonlinear_with_fem(
            f=phi,
            u=current_iterate,
            coordinates=coordinates,
            elements=elements,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        grad_J = np.zeros(n_coordinates, dtype=float)
        grad_J_on_free_nodes = (
            stiffness_matrix[free_nodes, :][:, free_nodes].dot(current_iterate[free_nodes])
            +
            load_vector_phi[free_nodes]
            -
            right_hand_side_vector[free_nodes]
        )
        grad_J[free_nodes] = grad_J_on_free_nodes
        return grad_J

    def J(current_iterate: np.ndarray) -> float:
        J = (
            0.5 * current_iterate.dot(stiffness_matrix.dot(current_iterate))
            +
            integrate_composition_nonlinear_with_fem(
                f=Phi,
                u=current_iterate,
                coordinates=coordinates,
                elements=elements,
                cubature_rule=CubatureRuleEnum.DAYTAYLOR)
            -
            right_hand_side_vector.dot(current_iterate)
        )
        return J

    # Custom Stopping Criterion: Energy Tail-Off
    # ------------------------------------------
    custom_callback = EnergyTailOffAveragedCustomCallback(
        batch_size=BATCHSIZE,
        min_n_iterations_per_mesh=MINITER,
        fudge=FUDGE,
        compute_energy=J)
    try:
        current_iterate, f_opt, func_calls, grad_calls, _ = \
            fmin_cg(f=J, x0=initial_guess, fprime=DJ, full_output=True, callback=custom_callback, gtol=1e-12)
    except ConvergedException as conv:
        current_iterate = conv.last_iterate
        n_iterations = conv.n_iterations_done
    print(f'\tIterations: {n_iterations}')
    show_solution(coordinates=coordinates, solution=current_iterate)

    # nonlinear EVA
    # -------------
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

    energy_gains = get_energy_gains_nonlinear(
        coordinates=coordinates,
        elements=elements,
        non_boundary_edges=non_boundary_edges,
        current_iterate=current_iterate,
        f=right_hand_side,
        a_11=a_11,
        a_12=a_12,
        a_21=a_21,
        a_22=a_22,
        phi=phi,
        eta=0.0,
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

    # Default usage of scipy's `fmin_cg`
    # ----------------------------------
    x_opt, f_opt, func_calls, grad_calls, _ = \
            fmin_cg(f=J, x0=initial_guess, fprime=DJ, full_output=True)
    show_solution(coordinates=coordinates, solution=x_opt)


if __name__ == '__main__':
    main()
