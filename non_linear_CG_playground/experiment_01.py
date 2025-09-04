from exceptiongroup import catch
from matplotlib.pylab import f
from p1afempy.solvers import \
    integrate_composition_nonlinear_with_fem, \
        get_load_vector_of_composition_nonlinear_with_fem, \
            get_general_stiffness_matrix, get_right_hand_side
from p1afempy import refinement
from p1afempy.data_structures import \
    BoundaryConditionType, CoordinatesType, ElementsType, BoundaryType
from triangle_cubature.cubature_rule import CubatureRuleEnum
import numpy as np
from scipy.optimize import fmin_cg
from scipy.sparse import csr_matrix
from show_solution import show_solution
from custom_callback import ConvergedException, EnergyTailOffAveragedCustomCallback

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
        batch_size=1,
        min_n_iterations_per_mesh=10,
        fudge=0.01,
        compute_energy=J)
    try:
        x_opt, f_opt, func_calls, grad_calls, _ = \
            fmin_cg(f=J, x0=initial_guess, fprime=DJ, full_output=True, callback=custom_callback, gtol=1e-12)
    except ConvergedException as conv:
        x_opt = conv.last_iterate
        n_iterations = conv.n_iterations_done
    print(f'n iterations done for custom energy tail-off callback: {n_iterations}')
    show_solution(coordinates=coordinates, solution=x_opt)

    # Default usage of scipy's `fmin_cg`
    # ----------------------------------
    print('default usage')
    x_opt, f_opt, func_calls, grad_calls, _ = \
            fmin_cg(f=J, x0=initial_guess, fprime=DJ, full_output=True)
    show_solution(coordinates=coordinates, solution=x_opt)


if __name__ == '__main__':
    main()
