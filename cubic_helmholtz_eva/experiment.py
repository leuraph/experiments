import numpy as np
from problems import get_problem
from p1afempy import refinement
from p1afempy.solvers import \
    integrate_composition_nonlinear_with_fem, \
        get_load_vector_of_composition_nonlinear_with_fem, \
            get_general_stiffness_matrix, get_right_hand_side
from scipy.sparse import csr_matrix
from triangle_cubature.cubature_rule import CubatureRuleEnum
from ismember import is_row_in
from p1afempy.mesh import provide_geometric_data
from scipy.sparse.linalg import spsolve
from scipy.optimize import fmin_cg
from utils import show_solution
from variational_adaptivity.edge_based_variational_adaptivity import get_energy_gains_nonlinear
from variational_adaptivity.markers import doerfler_marking
from p1afempy.refinement import refineNVB_edge_based
from p1afempy.mesh import show_mesh
from custom_callback import AriolisAdaptiveDelayCustomCallback, ConvergedException, EnergyTailOffAveragedCustomCallback

def get_custom_callback(
        stopping_criterion: str,
        args: argparse.Namespace,
        n_dofs: int,
        compute_energy: Callable[[np.ndarray], float]) -> CustomCallBack:
    """
    based on the arguments passed,
    returns the corresponding custom callback
    """
    if stopping_criterion == "energy-tail-off":
        callback = EnergyTailOffAveragedCustomCallback(
            batch_size=args.batchsize,
            min_n_iterations_per_mesh=args.miniter,
            fudge=args.fudge,
            compute_energy=compute_energy
        )
        return callback
    elif stopping_criterion == "relative-energy-decay":
        callback = AriolisAdaptiveDelayCustomCallback(
            batch_size=1,
            min_n_iterations_per_mesh=args.miniter,
            initial_delay=args.initial_delay,
            delay_increase=args.delay_increase,
            tau=args.tau,
            fudge=args.fudge,
            n_dofs=n_dofs,
            compute_energy=compute_energy
        )
        return callback
    elif stopping_criterion == "default":
        callback = CustomCallBack(
            batch_size=1,
            min_n_iterations_per_mesh=1,
            compute_energy=compute_energy
        )
        return callback
    else:
        raise NotImplementedError(
            'The custom callback corresponding to the stopping criterion'
            f'{stopping_criterion} is not implemented.')


def get_results_path(args: argparse.Namespace) -> Path:
    """
    Returns a Path object for the results directory, based on the stopping criterion and arguments.
    The path string starts with the problem number and includes the stopping criterion name.
    """
    base = f"problem-{args.problem}_{args.stopping_criterion}_"
    if args.stopping_criterion == "energy-tail-off":
        path_str = (
            base +
            f"theta-{args.theta}_eta-{args.eta}_fudge-{args.fudge}_miniter-{args.miniter}_batchsize-{args.batchsize}"
        )
    elif args.stopping_criterion == "relative-energy-decay":
        path_str = (
            base +
            f"theta-{args.theta}_eta-{args.eta}_fudge-{args.fudge}_miniter-{args.miniter}_tau-{args.tau}_initial_delay-{args.initial_delay}_delay_increase-{args.delay_increase}"
        )
    elif args.stopping_criterion == "default":
        path_str = (
            base +
            f"theta-{args.theta}_eta-{args.eta}_miniter-{args.miniter}_gtol-{args.gtol}"
        )
    else:
        raise ValueError(f"Unknown stopping criterion: {args.stopping_criterion}")
    return Path("results/") / Path(path_str)


def main() -> None:
    n_initial_refinements = 2
    max_dof = 1e6

    ETA = 0.5
    THETA = 0.5

    problem = get_problem(number=1)

    # extracting the initial mesh
    # ---------------------------
    initial_coarse_mesh = problem.get_coarse_initial_mesh()
    coordinates = initial_coarse_mesh.coordinates
    elements = initial_coarse_mesh.elements
    boundaries = initial_coarse_mesh.boundaries

    # extracting the corresponding PDE's data
    # ---------------------------------------
    f = problem.f
    phi = problem.phi
    phi_prime = problem.phi_prime
    Phi = problem.Phi
    a_11 = problem.a_11
    a_12 = problem.a_12
    a_21 = problem.a_21
    a_22 = problem.a_22
    # ---------------------------------------

    # initial refinement
    # ---------------------------------------
    for _ in range(n_initial_refinements):
        marked_elements = np.arange(elements.shape[0])
        coordinates, elements, boundaries, _ = refinement.refineNVB(
            coordinates=coordinates,
            elements=elements,
            marked_elements=marked_elements,
            boundary_conditions=boundaries)
    n_vertices = coordinates.shape[0]
    indices_of_free_nodes = np.setdiff1d(
        ar1=np.arange(n_vertices),
        ar2=np.unique(boundaries[0].flatten()))
    free_nodes = np.zeros(n_vertices, dtype=bool)
    free_nodes[indices_of_free_nodes] = 1
    n_dofs = np.sum(free_nodes)
    print(f'DOF = {n_dofs}')
    # ---------------------------------------

    # on the initial mesh, solve the auxiliary problem
    # -Laplace phi = 1, with homogeneous boundary conditions
    # ------------------------------------------------------
    
    auxiliary_problem = get_problem(number=2)

    f_aux = auxiliary_problem.f
    a_11_aux = auxiliary_problem.a_11
    a_12_aux = auxiliary_problem.a_12
    a_21_aux = auxiliary_problem.a_21
    a_22_aux = auxiliary_problem.a_22

    stiffness_matrix_aux = csr_matrix(get_general_stiffness_matrix(
        coordinates=coordinates,
        elements=elements,
        a_11=a_11_aux, a_12=a_12_aux, a_21=a_21_aux, a_22=a_22_aux,
        cubature_rule=CubatureRuleEnum.MIDPOINT))

    right_hand_side_vector_aux = get_right_hand_side(
        coordinates=coordinates,
        elements=elements,
        f=f_aux,
        cubature_rule=CubatureRuleEnum.MIDPOINT)
    
    n_coordinates = coordinates.shape[0]
    auxiliary_solution = np.zeros(n_coordinates)

    auxiliary_solution[free_nodes] = spsolve(
        stiffness_matrix_aux[free_nodes, :][:, free_nodes],
        right_hand_side_vector_aux[free_nodes],
        use_umfpack=True)
    
    psi_max = np.max(auxiliary_solution)

    lamba = 12.
    c = np.sqrt((lamba*psi_max - 1)/(psi_max**3))
    # ------------------------------------------------------

    initial_gues_plus = c*auxiliary_solution
    initial_gues_minus = -c*auxiliary_solution

    stiffness_matrix = csr_matrix(get_general_stiffness_matrix(
        coordinates=coordinates,
        elements=elements,
        a_11=a_11, a_12=a_12, a_21=a_21, a_22=a_22,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR))

    right_hand_side_vector = get_right_hand_side(
        coordinates=coordinates,
        elements=elements,
        f=f,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR)

    def DJ(current_iterate: np.ndarray) -> np.ndarray:

        load_vector_phi = get_load_vector_of_composition_nonlinear_with_fem(
            f=phi,
            u=current_iterate,
            coordinates=coordinates,
            elements=elements,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        grad_J = np.zeros(n_vertices, dtype=float)
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
        energy = (
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
        return energy
    
    current_iterate, f_opt, func_calls, grad_calls, warnflag = \
        fmin_cg(
            f=J,
            x0=initial_gues_minus,
            fprime=DJ,
            full_output=True)
    
    show_solution(
        coordinates=coordinates,
        solution=current_iterate)
    
    # fmin_cg with default stopping criterion on initial mesh
    # -------------------------------------------------------
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
    n_dofs = np.sum(free_nodes)

    # nonlinear EVA
    # -------------
    energy_gains = get_energy_gains_nonlinear(
        coordinates=coordinates,
        elements=elements,
        non_boundary_edges=non_boundary_edges,
        current_iterate=current_iterate,
        f=f,
        a_11=a_11,
        a_12=a_12,
        a_21=a_21,
        a_22=a_22,
        phi=phi,
        phi_prime=phi_prime,
        eta=ETA,
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
    
    # main loop of the experiment, i.e.,
    # approximate -> mark -> refine
    # ----------------------------------
    while True:
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
        n_dofs = np.sum(free_nodes)

        # midpoint suffices as we consider laplace operator
        stiffness_matrix = csr_matrix(get_general_stiffness_matrix(
            coordinates=coordinates,
            elements=elements,
            a_11=a_11, a_12=a_12, a_21=a_21, a_22=a_22,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR))

        right_hand_side_vector = get_right_hand_side(
            coordinates=coordinates,
            elements=elements,
            f=f,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        def DJ(current_iterate: np.ndarray) -> np.ndarray:

            load_vector_phi = get_load_vector_of_composition_nonlinear_with_fem(
                f=phi,
                u=current_iterate,
                coordinates=coordinates,
                elements=elements,
                cubature_rule=CubatureRuleEnum.DAYTAYLOR)

            grad_J = np.zeros(n_vertices, dtype=float)
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
            energy = (
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
            return energy

        current_iterate, fopt, func_calls, grad_calls, warnflag = \
            fmin_cg(
                f=J,
                x0=current_iterate,
                fprime=DJ,
                full_output=True)


        # break after we have solved for the first mesh that
        # exceeds the maximum number of degrees of freedom
        if n_dofs >= max_dof:
            print("maximum number of degrees of freedom exceeded, stopping iteration")
            break

        # nonlinear EVA
        # -------------
        energy_gains = get_energy_gains_nonlinear(
            coordinates=coordinates,
            elements=elements,
            non_boundary_edges=non_boundary_edges,
            current_iterate=current_iterate,
            f=f,
            a_11=a_11,
            a_12=a_12,
            a_21=a_21,
            a_22=a_22,
            phi=phi,
            phi_prime=phi_prime,
            eta=ETA,
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

        show_mesh(coordinates, elements)
        # show_solution(coordinates, current_iterate)


if __name__ == '__main__':
    main()
