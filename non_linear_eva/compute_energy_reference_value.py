"""
This scripts approximates the energy
for the specified problem using nonlinear CG iterations
on uniformly refined meshes
"""
import numpy as np
from p1afempy import refinement
from p1afempy.solvers import \
    get_general_stiffness_matrix, get_right_hand_side
from triangle_cubature.cubature_rule import CubatureRuleEnum
from p1afempy.refinement import refineNVB
from problems import get_problem
import argparse
from p1afempy.solvers import get_load_vector_of_composition_nonlinear_with_fem, \
    integrate_composition_nonlinear_with_fem
from scipy.optimize import fmin_cg
from show_solution import show_solution


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=int, required=True,
                        help="problem number to be considered")
    parser.add_argument("--gtol", type=float, required=True,
                        help="stop when the norm of the gradient is less than `gtol`")
    parser.add_argument("--debug", type=bool, required=False,
                        default=False,
                        help="if True, show solution after convergence")
    args = parser.parse_args()

    n_max_dofs = 100e6
    n_initial_refinements = 5
    GTOL_CG = args.gtol
    PROBLEM_N = args.problem
    DEBUG = args.debug

    problem = get_problem(PROBLEM_N)

    # extracting function handles from Problem object
    a_11 = problem.a_11
    a_12 = problem.a_12
    a_21 = problem.a_21
    a_22 = problem.a_22
    f = problem.f
    phi = problem.phi
    Phi = problem.Phi

    # Printing meta information first
    print(f'Calculating CG approximations and their energy...')
    print(f'')
    print(f'parameters')
    print(f'----------')
    print(f'problem number = {PROBLEM_N}')
    print(f'stop when the norm of the gradient is less than {GTOL_CG}')
    print(f'')

    # initial coarse mesh
    # -------------------
    coarse_mesh = problem.get_coarse_initial_mesh()

    coordinates = coarse_mesh.coordinates
    elements = coarse_mesh.elements
    boundaries = coarse_mesh.boundaries

    # initial refinement
    # ------------------
    for _ in range(n_initial_refinements):
        marked = np.arange(elements.shape[0])
        coordinates, elements, boundaries, _ = \
            refinement.refineNVB(coordinates=coordinates,
                                 elements=elements,
                                 marked_elements=marked,
                                 boundary_conditions=boundaries)

    n_vertices = coordinates.shape[0]
    indices_of_free_nodes = np.setdiff1d(
        ar1=np.arange(n_vertices),
        ar2=np.unique(boundaries[0].flatten()))
    free_nodes = np.zeros(n_vertices, dtype=bool)
    free_nodes[indices_of_free_nodes] = 1
    n_dofs = np.sum(free_nodes)

    # initial guess on initial mesh
    # -----------------------------
    current_iterate = np.zeros(n_vertices)

    while True:

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

        # approximate Galerkin solution on current mesh
        # ---------------------------------------------
        current_iterate, J_opt, func_calls, grad_calls, _ = fmin_cg(
            f=J, x0=current_iterate, fprime=DJ, full_output=True, gtol=GTOL_CG)

        print(
            f'nDOF = {n_dofs}, \t'
            f'converged CG approximation energy = {J_opt}, '
            f'n function calls = {func_calls}, '
            f'n gradient calls = {grad_calls}')

        marked = np.ones(elements.shape[0], dtype=bool)
        coordinates, elements, boundaries, current_iterate = \
            refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked,
                boundary_conditions=boundaries,
                to_embed=current_iterate)
        
        if DEBUG:
            show_solution(coordinates=coordinates, solution=current_iterate)
        
        # reset as the mesh has changed
        # -----------------------------
        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        n_dofs = np.sum(free_nodes)

        if n_dofs >= n_max_dofs:
            print(
                f'Maximum number of DOFs ({n_max_dofs})'
                'reached, stopping iteration.')
            break


if __name__ == '__main__':
    main()
