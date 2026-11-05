from matplotlib import pyplot as plt
from p1afempy.solvers import \
    get_weighted_stiffness_matrix, get_stiffness_matrix, \
        get_right_hand_side, CubatureRuleEnum
from p1afempy import refinement
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from copy import copy
import tqdm
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from pathlib import Path


def get_restarted_euler_error_estimator(
        grad_energy: np.ndarray,
        eta_m: float,
        alpha: float,
        beta: float,
        dt: float,
        norm
) -> float:
    """
    Returns an error estimate for the euler error
    for one (restarted) step of size dt, i.e.,

    parameters
    ----------
    grad_energy: np.ndarray
        Riesz Representative J_X E'(u) in the P1FEM Lagrange basis
    eta_m: float
        monotonicity constant of E'
    alpha: float
        coercivity constant of A
    beta: float
        boundedness constant of A
    dt: float
        step-size
    norm
        Hilbert space norm, i.e. norm(u) := \|u\|_X
    """
    return norm(grad_energy)/eta_m * np.exp(-(alpha*eta_m)/(beta**2) * dt)


def get_restarted_relaxation_estimate(
        u_old: np.ndarray,
        u_new: np.ndarray,
        eta_m: float,
        alpha: float,
        beta: float,
        L: float,
        dt: float,
        norm
) -> float:
    """
    Returns an error estimate for the euler error
    for one (restarted) step of size dt, i.e.,

    parameters
    ----------
    u_old: np.ndarray
        u^{n}_N
    u_new: np.ndarray
        u^{n+1}_N
    eta_m: float
        monotonicity constant of E'
    alpha: float
        coercivity constant of A
    beta: float
        boundedness constant of A
    L: float
        Lipschitz constant of E
    dt: float
        step-size
    norm
        Hilbert space norm, i.e. norm(u) := \|u\|_X
    """
    return (
        np.sqrt((beta)/(2.*alpha)) 
        * (L / eta_m) 
        * norm(u_new - u_old) 
        * np.sqrt(1. - np.exp(-0.5 * eta_m * dt))
    )

def get_restarted_estimator(
        grad_energy_norm: np.ndarray,
        eta_m: float,
        alpha: float,
        beta: float,
        L: float,
        dt: float
) -> float:
    return grad_energy_norm * (
        (
            (1./eta_m)
            * np.exp(-(alpha*eta_m)/(beta**2) * dt)
        )
        +
        (
            np.sqrt(beta/alpha)
            * (L / eta_m)
            * dt
            * np.sqrt(1. - np.exp(- eta_m/beta * dt))
        )
    )


def get_restarted_estimator_derivative(
        grad_energy_norm: np.ndarray,
        eta_m: float,
        alpha: float,
        beta: float,
        L: float,
        dt: float
) -> float:
    alpha_tilde = (alpha*eta_m)/(beta**2)
    beta_tilde =  np.sqrt((beta)/(2.*alpha)) * L / eta_m

    return grad_energy_norm * (
        -alpha_tilde*np.exp(-alpha_tilde*dt)/eta_m 
        + 
        0.25*beta_tilde*eta_m*dt*np.exp(-0.5*eta_m*dt)/np.sqrt(1 - np.exp(-0.5*eta_m*dt)) 
        + 
        beta_tilde*np.sqrt(1 - np.exp(-0.5*eta_m*dt)))


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
    n_initial_iterations = 5
    n_initial_refinements = 6
    max_n_iteration_steps = 400
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
    
    restarted_euler_error_estimations = []
    restarted_relaxation_error_estimations = []
    dts = []

    print('solving the problem...')
    for k in tqdm.tqdm(range(max_n_iteration_steps)):

        weighted_stiffness_matrix = csr_matrix(get_weighted_stiffness_matrix(
            coordinates=coordinates,
            elements=elements,
            u=current_iterate,
            mu=mu_1))
        
        # # Computing J_X E'(u^n_N)
        b = (
            weighted_stiffness_matrix[free_nodes,:][:, free_nodes].dot(
                current_iterate[free_nodes])
            -
            load_vector[free_nodes]
        )

        grad_energy = np.zeros(n_vertices)
        grad_energy[free_nodes] = spsolve(
            stiffness_matrix[free_nodes, :][:, free_nodes],
            b,
            use_umfpack=True
        )
        grad_energy_norm = norm(grad_energy)
        estimator = lambda s: get_restarted_estimator(
            grad_energy_norm=grad_energy_norm,
            eta_m=nu_1,
            alpha=eta_c,
            beta=beta,
            L=L_1,
            dt=s
        )
        estimator_diff = lambda s: get_restarted_estimator_derivative(
            grad_energy_norm=grad_energy_norm,
            eta_m=nu_1,
            alpha=eta_c,
            beta=beta,
            L=L_1,
            dt=s
        )
        dt = root_scalar(f=estimator_diff, x0=dt).root
        dts.append(dt)
        s_array = np.linspace(0., 0.5, 100)
        # plt.plot(s_array, [estimator(s) for s in s_array])
        # plt.show()

        current_iterate[free_nodes] = (
            old_iterate[free_nodes] - dt * grad_energy[free_nodes]
        )

        restarted_euler_error_estimations.append(
            get_restarted_euler_error_estimator(
                grad_energy=grad_energy,
                eta_m=nu_1,
                alpha=eta_c,
                beta=beta,
                dt=dt,
                norm=norm
            )
        )
        restarted_relaxation_error_estimations.append(
            get_restarted_relaxation_estimate(
                u_old=old_iterate,
                u_new=current_iterate,
                eta_m=nu_1,
                alpha=eta_c,
                beta=beta,
                L=L_1,
                dt=dt,
                norm=norm
            )
        )
        
        direct_error_estimations.append(
            direct_error_estimator(
                u_new=current_iterate,
                u_old=old_iterate,
                dt=dt))

        old_iterate = copy(current_iterate)

    
    restarted_euler_error_estimations = np.array(
        restarted_euler_error_estimations)
    restarted_relaxation_error_estimations = np.array(
        restarted_relaxation_error_estimations
    )
    restarted_estimations = (
        restarted_euler_error_estimations
        + restarted_relaxation_error_estimations
    )

    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['text.usetex'] = True

    fig, ax = plt.subplots()
    ax.set_xlabel("$\mathrm{number}~\mathrm{of}~\mathrm{iterations}$")
    ax.set_ylabel(r'$\mathrm{estimator}$')
    ax.grid(True)

    ax.semilogy(direct_error_estimations, label='direct')
    ax.semilogy(restarted_estimations, label='restarted')
    ax.legend(loc='best')

    output_path = Path('plots') / Path('restarted_estimator.pdf')
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    main()
