from p1afempy.solvers import \
    get_weighted_stiffness_matrix, get_stiffness_matrix, \
        get_right_hand_side, CubatureRuleEnum
from p1afempy import refinement
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from matplotlib.pylab import cm


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
    #ax.set_zticks([0, 0.02, 0.04, 0.06])

    # Show and save the plot
    # fig.savefig(out_path, dpi=300)
    plt.show()


# Manufactured solution
def analytical_solution(r):
    x, y = r[:, 0], r[:, 1]
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

def main() -> None:
    n_initial_refinements = 5
    max_n_iteration_steps = 10

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

    # load vector does not change during iterations
    load_vector = get_right_hand_side(
        coordinates=coordinates,
        elements=elements,
        f=rhs_function_1,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR)

    for _ in range(max_n_iteration_steps):
        dt = 0.1

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


    show_solution(coordinates=coordinates, solution=current_iterate)

    show_solution(
        coordinates=coordinates,
        solution=analytical_solution(coordinates))


if __name__ == '__main__':
    main()
