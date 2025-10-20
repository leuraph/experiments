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
from custom_callback import AriolisAdaptiveDelayCustomCallback, ConvergedException, EnergyTailOffAveragedCustomCallback
import argparse
from variational_adaptivity.edge_based_variational_adaptivity import get_energy_gains_nonlinear
from variational_adaptivity.markers import doerfler_marking
from p1afempy.refinement import refineNVB_edge_based
from problems import get_problem
from custom_callback import CustomCallBack, EnergyTailOffAveragedCustomCallback, \
    energ


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.stopping_criterion == "energy-tail-off":
        if args.miniter is None:
            parser.error("--miniter is required for 'energy-tail-off' stopping criterion.")
        if args.batchsize is None:
            parser.error("--batchsize is required for 'energy-tail-off' stopping criterion.")
        if args.fudge is None:
            parser.error("--fudge is required for 'energy-tail-off' stopping criterion.")
    if args.stopping_criterion == "relative-energy-decay":
        if args.miniter is None:
            parser.error("--miniter is required for 'relative-energy-decay' stopping criterion.")
        if args.tau is None:
            parser.error("--tau is required for 'relative-energy-decay' stopping criterion.")
        if args.fudge is None:
            parser.error("--fudge is required for 'relative-energy-decay' stopping criterion.")
        if args.initial_delay is None:
            parser.error("--initial-delay is required for 'relative-energy-decay' stopping criterion.")
        if args.delay_increase is None:
            parser.error("--delay-increase is required for 'relative-energy-decay' stopping criterion.")
    if args.stopping_criterion == "default":
        if args.miniter is None:
            parser.error("--miniter is required for 'default' stopping criterion.")
        if args.gtol is None:
            parser.error("--gtol is required for 'default' stopping criterion.")


def get_custom_callback(
        stopping_criterion: str,
        args: argparse.Namespace) -> CustomCallBack:
    """
    based on the arguments passed,
    returns the corresponding custom callback
    """
    if stopping_criterion == "energy-tail-off":
        callback = EnergyTailOffAveragedCustomCallback(
            batch_size=args.batchsize,
            min_n_iterations_per_mesh=args.miniter,
            fudge=args.fudge,
            compute_energy=None #TODO add
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
            compute_energy=None #TODO add
        )
        return callback
    elif stopping_criterion == "default":
        def callback() -> None:
            pass
        return callback
    else:
        raise NotImplementedError(
            'The custom callback corresponding to the stopping criterion'
            f'{stopping_criterion} is not implemented.')


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--problem", type=int, required=True,
                        help="problem number to be considered")
    parser.add_argument("--theta", type=float, required=True,
                        help="Dörfler marking parameter")
    parser.add_argument("--eta", type=float, required=True,
                        help="control parameter for computation of locally imporved approximation")
    parser.add_argument("--stopping-criterion", type=str, required=True,
                        choices=["energy-tail-off", "relative-energy-decay", "default"],
                        help="stopping criterion to be used, "
                        "choices are: [`energy-tail-off`, `relative-energy-decay`, `default`]")

    # arguments used for all stopping criteria
    # ----------------------------------------
    parser.add_argument("--miniter", type=int, required=False,
                        help="minimum number of iterations on each mesh")

    # arguments not used for all criteria
    # -----------------------------------
    parser.add_argument("--gtol", type=float, required=False,
                        help="stop when the norm of the gradient is less than `gtol`")
    parser.add_argument("--fudge", type=float, required=False,
                        help="fudge factor")
    parser.add_argument("--batchsize", type=int, required=False)
    parser.add_argument("--tau", type=float, required=False,
                        help="relative energy decay parameter")
    parser.add_argument("--initial-delay", type=int, required=False)
    parser.add_argument("--delay-increase", type=int, required=False)
    args = parser.parse_args()

    # validation depending on the choice of stopping criterion
    validate_args(args=args, parser=parser)

    # control parameter extraction
    # ----------------------------
    PROBLEM_N = args.problem
    THETA = args.theta
    ETA = args.eta
    STOPPING_CRITERION = args.stopping_criterion
    MINITER = args.miniter
    GTOL = args.gtol
    FUDGE = args.fudge
    BATCHSIZE = args.batchsize
    TAU = args.tau
    INITIAL_DELAY = args.initial_delay
    DELAY_INCREASE = args.delay_increase
    # ----------------------------

    problem = get_problem(number=PROBLEM_N)

    # extracting the initial mesh
    # ---------------------------
    initial_coarse_mesh = problem.get_coarse_initial_mesh()
    coordinates = initial_coarse_mesh.coordinates
    elements = initial_coarse_mesh.elements
    boundaries = initial_coarse_mesh.boundaries
    # ---------------------------

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
    n_refinements = 5
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

    current_iterate = np.zeros(n_coordinates, dtype=float)

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

    # Custom Stopping Criterion: Energy Tail-Off or Others
    # ----------------------------------------------------
    custom_callback = get_custom_callback(
        stopping_criterion=args.stopping_criterion, args=args)

    # distinguish two cases; default and non-default stopping criteria
    # as default criteria does not implement a custom callback and therefore
    # does not raise on convergence
    if args.stopping_criterion == "default":
        gtol = GTOL  # could be None, fmin_cg will use its default
        current_iterate, f_opt, func_calls, grad_calls, _ = \
            fmin_cg(
                f=J,
                x0=current_iterate,
                fprime=DJ,
                full_output=True,
                callback=custom_callback,
                gtol=gtol)
        n_iterations = func_calls
    else:
        gtol = 1e-12  # or smaller to ensure custom stopping criterion is used
        try:
            _, _, _, _, _ = \
            fmin_cg(
                f=J,
                x0=current_iterate,
                fprime=DJ,
                full_output=True,
                callback=custom_callback,
                gtol=gtol
            )
            raise RuntimeError("fmin_cg failed to converge, stopping immediately")
        except ConvergedException as conv:
            current_iterate = conv.last_iterate
            n_iterations = conv.n_iterations_done


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


if __name__ == '__main__':
    main()
