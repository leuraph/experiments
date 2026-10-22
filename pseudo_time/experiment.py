from cProfile import label
from matplotlib import pyplot as plt
from p1afempy.solvers import \
    get_weighted_stiffness_matrix, get_stiffness_matrix, \
        get_right_hand_side, CubatureRuleEnum
from p1afempy import refinement
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from show_solution import show_solution
from copy import copy
import tqdm


# Manufactured solution
def analytical_solution(r):
    x, y = r[:, 0], r[:, 1]
    return 2.*x*y*(1.-x*x)*(1.-y*y)/(x*x + y*y)**(2./3.)


def analytical_solution_safe(x, y):
    if x==0. and y==0.:
        return 0.
    return 2.*x*y*(1-x*x)*(1-y*y)/(x*x + y*y)**(2/3)


# ------------------------------------------------------------------------------
# PROBLEM 1
# ------------------------------------------------------------------------------
def mu_1(t):
    return 1. + np.exp(-t)

def rhs_function_1(r):
    x, y = r[:, 0], r[:, 1]
    return (np.exp(-2) + 1)*(80*x*y**3*(1 - x**2)*(1 - y**2)/(9*(x**2 + y**2)**(8/3)) + 32*x*y**3*(1 - x**2)/(3*(x**2 + y**2)**(5/3)) - 8*x*y*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(5/3) - 12*x*y*(1 - x**2)/(x**2 + y**2)**(2/3)) + (np.exp(-2) + 1)*(80*x**3*y*(1 - x**2)*(1 - y**2)/(9*(x**2 + y**2)**(8/3)) + 32*x**3*y*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 8*x*y*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(5/3) - 12*x*y*(1 - y**2)/(x**2 + y**2)**(2/3))

m_mu_1 = 1. - 2.*np.exp(-3./2.)
M_mu_1 = 2.

nu_1 = m_mu_1
L_1  = 3*M_mu_1
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# PROBLEM 2
# ------------------------------------------------------------------------------
def mu_2(t):
    return 1./(1+t) + 0.5

def rhs_function_2(r):
    x, y = r[:, 0], r[:, 1]
    return (0.5 + 1/((-8*x*y**2*(1 - x**2)*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 4*x*y**2*(1 - x**2)/(x**2 + y**2)**(2/3) + 2*x*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(2/3))**2 + (-8*x**2*y*(1 - x**2)*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 4*x**2*y*(1 - y**2)/(x**2 + y**2)**(2/3) + 2*y*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(2/3))**2 + 1))*(80*x*y**3*(1 - x**2)*(1 - y**2)/(9*(x**2 + y**2)**(8/3)) + 32*x*y**3*(1 - x**2)/(3*(x**2 + y**2)**(5/3)) - 8*x*y*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(5/3) - 12*x*y*(1 - x**2)/(x**2 + y**2)**(2/3)) + (0.5 + 1/((-8*x*y**2*(1 - x**2)*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 4*x*y**2*(1 - x**2)/(x**2 + y**2)**(2/3) + 2*x*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(2/3))**2 + (-8*x**2*y*(1 - x**2)*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 4*x**2*y*(1 - y**2)/(x**2 + y**2)**(2/3) + 2*y*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(2/3))**2 + 1))*(80*x**3*y*(1 - x**2)*(1 - y**2)/(9*(x**2 + y**2)**(8/3)) + 32*x**3*y*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 8*x*y*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(5/3) - 12*x*y*(1 - y**2)/(x**2 + y**2)**(2/3)) + (-(-8*x*y**2*(1 - x**2)*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 4*x*y**2*(1 - x**2)/(x**2 + y**2)**(2/3) + 2*x*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(2/3))*(160*x*y**3*(1 - x**2)*(1 - y**2)/(9*(x**2 + y**2)**(8/3)) + 64*x*y**3*(1 - x**2)/(3*(x**2 + y**2)**(5/3)) - 16*x*y*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(5/3) - 24*x*y*(1 - x**2)/(x**2 + y**2)**(2/3)) - (-8*x**2*y*(1 - x**2)*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 4*x**2*y*(1 - y**2)/(x**2 + y**2)**(2/3) + 2*y*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(2/3))*(160*x**2*y**2*(1 - x**2)*(1 - y**2)/(9*(x**2 + y**2)**(8/3)) + 32*x**2*y**2*(1 - x**2)/(3*(x**2 + y**2)**(5/3)) + 32*x**2*y**2*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) + 16*x**2*y**2/(x**2 + y**2)**(2/3) - 16*x**2*(1 - x**2)*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 8*x**2*(1 - y**2)/(x**2 + y**2)**(2/3) - 16*y**2*(1 - x**2)*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 8*y**2*(1 - x**2)/(x**2 + y**2)**(2/3) + 4*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(2/3)))*(-8*x*y**2*(1 - x**2)*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 4*x*y**2*(1 - x**2)/(x**2 + y**2)**(2/3) + 2*x*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(2/3))/((-8*x*y**2*(1 - x**2)*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 4*x*y**2*(1 - x**2)/(x**2 + y**2)**(2/3) + 2*x*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(2/3))**2 + (-8*x**2*y*(1 - x**2)*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 4*x**2*y*(1 - y**2)/(x**2 + y**2)**(2/3) + 2*y*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(2/3))**2 + 1)**2 + (-(-8*x*y**2*(1 - x**2)*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 4*x*y**2*(1 - x**2)/(x**2 + y**2)**(2/3) + 2*x*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(2/3))*(160*x**2*y**2*(1 - x**2)*(1 - y**2)/(9*(x**2 + y**2)**(8/3)) + 32*x**2*y**2*(1 - x**2)/(3*(x**2 + y**2)**(5/3)) + 32*x**2*y**2*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) + 16*x**2*y**2/(x**2 + y**2)**(2/3) - 16*x**2*(1 - x**2)*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 8*x**2*(1 - y**2)/(x**2 + y**2)**(2/3) - 16*y**2*(1 - x**2)*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 8*y**2*(1 - x**2)/(x**2 + y**2)**(2/3) + 4*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(2/3)) - (-8*x**2*y*(1 - x**2)*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 4*x**2*y*(1 - y**2)/(x**2 + y**2)**(2/3) + 2*y*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(2/3))*(160*x**3*y*(1 - x**2)*(1 - y**2)/(9*(x**2 + y**2)**(8/3)) + 64*x**3*y*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 16*x*y*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(5/3) - 24*x*y*(1 - y**2)/(x**2 + y**2)**(2/3)))*(-8*x**2*y*(1 - x**2)*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 4*x**2*y*(1 - y**2)/(x**2 + y**2)**(2/3) + 2*y*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(2/3))/((-8*x*y**2*(1 - x**2)*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 4*x*y**2*(1 - x**2)/(x**2 + y**2)**(2/3) + 2*x*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(2/3))**2 + (-8*x**2*y*(1 - x**2)*(1 - y**2)/(3*(x**2 + y**2)**(5/3)) - 4*x**2*y*(1 - y**2)/(x**2 + y**2)**(2/3) + 2*y*(1 - x**2)*(1 - y**2)/(x**2 + y**2)**(2/3))**2 + 1)**2

m_mu_2 = 3/8
M_mu_2 = 3/2

nu_2 = m_mu_2
L_2  = 3*M_mu_2
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Preconditioner A
# ------------------------------------------------------------------------------
# We choose <Au, v> := (u, v)_X.
# Hence, we have <Au, v> \leq |u|_X |v|_X due to Cauchy-Schwarz,
# and |u|_X^2 = <Au, u> >= |u|_X^2 by choice.
beta = 1.
eta_c = 1

def main() -> None:
    n_initial_iterations = 10
    n_initial_refinements = 7
    max_n_iteration_steps = 10
    dt = 0.1

    # Coarse triangulation of L-shaped domain
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

    n_vertices = coordinates.shape[0]

    current_iterate = np.zeros(n_vertices)

    for _ in range(n_initial_refinements):
        marked_elements = np.arange(elements.shape[0])
        coordinates, elements, boundaries, current_iterate = refinement.refineNVB(
            coordinates=coordinates,
            elements=elements,
            marked_elements=marked_elements,
            boundary_conditions=boundaries,
            to_embed=current_iterate)

    n_vertices = coordinates.shape[0]
    indices_of_free_nodes = np.setdiff1d(
        ar1=np.arange(n_vertices),
        ar2=np.unique(boundaries[0].flatten()))
    free_nodes = np.zeros(n_vertices, dtype=bool)
    free_nodes[indices_of_free_nodes] = 1
    n_dofs = np.sum(free_nodes)
    print(f'DOF = {n_dofs}')

    stiffness_matrix = csr_matrix(get_stiffness_matrix(
        coordinates=coordinates,
        elements=elements))
    
    def norm(u: np.ndarray) -> float:
        """
        Discrete version of the energy-norm, i.e.,
        \|u\|_X := \|nabla u\|_{L2}.
        """
        sqrt_arg = u[free_nodes].dot(
            stiffness_matrix[free_nodes,:][:, free_nodes].dot(
                u[free_nodes]))
        return np.sqrt(sqrt_arg)
    
    def direct_error_estimator(
            u_new: np.ndarray,
            u_old: np.ndarray,
            dt: float) -> float:
        return (1. + (beta)/(dt * nu_1))*norm(u=u_new - u_old)
    
    def splitted_error_estimator_part_1(
            u_init: np.ndarray,
            t: float):
        M = csr_matrix(get_weighted_stiffness_matrix(
            coordinates=coordinates,
            elements=elements,
            u=u_init,
            mu=mu_1))
        F = get_right_hand_side(
            coordinates=coordinates,
            elements=elements,
            f=rhs_function_1,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)
        As = csr_matrix(get_stiffness_matrix(
            coordinates=coordinates,
            elements=elements))
        E = np.zeros(n_vertices)
        E[free_nodes] = spsolve(
            As[free_nodes,:][:,free_nodes],
            M[free_nodes,:][:, free_nodes].dot(
                u_init[free_nodes])
            - F[free_nodes])
        
        return (norm(E)/nu_1) * np.exp(-((nu_1 * eta_c)/(beta**2))*t)
        

    # load vector does not change during iterations
    load_vector = get_right_hand_side(
        coordinates=coordinates,
        elements=elements,
        f=rhs_function_1,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR)


    print('initial iterations...')
    for _ in tqdm.tqdm(range(n_initial_iterations)):
        weighted_stiffness_matrix = csr_matrix(get_weighted_stiffness_matrix(
            coordinates=coordinates,
            elements=elements,
            u=current_iterate,
            mu=mu_1))

        b = (
            stiffness_matrix.dot(current_iterate)
            -
            dt * weighted_stiffness_matrix.dot(current_iterate)
            +
            dt * load_vector
        )

        current_iterate[free_nodes] = spsolve(
            stiffness_matrix[free_nodes, :][:, free_nodes],
            b[free_nodes],
            use_umfpack=True)

    u_initial = copy(current_iterate)
    old_iterate = copy(current_iterate)

    direct_error_estimations = []
    splitted_error_estimations_pt1 = []

    print('solving the problem...')
    for k in tqdm.tqdm(range(max_n_iteration_steps)):

        weighted_stiffness_matrix = csr_matrix(get_weighted_stiffness_matrix(
            coordinates=coordinates,
            elements=elements,
            u=current_iterate,
            mu=mu_1))

        b = (
            stiffness_matrix.dot(current_iterate)
            -
            dt * weighted_stiffness_matrix.dot(current_iterate)
            +
            dt * load_vector
        )

        current_iterate[free_nodes] = spsolve(
            stiffness_matrix[free_nodes, :][:, free_nodes],
            b[free_nodes],
            use_umfpack=True)
        
        direct_error_estimations.append(
            direct_error_estimator(
                u_new=current_iterate,
                u_old=old_iterate,
                dt=dt))
        
        splitted_error_estimations_pt1.append(
            splitted_error_estimator_part_1(
                u_init=u_initial,
                t=(k+1)*dt))

        old_iterate = copy(current_iterate)


    plt.semilogy(direct_error_estimations, label='direct')
    plt.semilogy(splitted_error_estimations_pt1, label='split pt.1')
    plt.legend()
    plt.show()
    # show_solution(coordinates=coordinates, solution=current_iterate)


if __name__ == '__main__':
    main()
