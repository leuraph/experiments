"""
This scripts approximates the energy norm squared
for the specified problem using CG iterations
on uniformly refined meshes
"""
import numpy as np
from p1afempy import refinement
from p1afempy.solvers import \
    get_general_stiffness_matrix, get_mass_matrix, get_right_hand_side
from triangle_cubature.cubature_rule import CubatureRuleEnum
from scipy.sparse.linalg import spsolve
from p1afempy.refinement import refineNVB
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg
from problems import get_problem
import argparse


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=int, required=True,
                        help="problem number to be considered")
    parser.add_argument("--rtol", type=float, required=True,
                        help="relative tolerance for CG iterations")
    args = parser.parse_args()

    n_max_dofs = 100e6
    n_initial_refinements = 5
    RTOL_CG = args.rtol
    PROBLEM_N = args.problem

    problem = get_problem(PROBLEM_N)

    # Printing meta information first
    print(f'Calculating CG approximations and their energy norm squared...')
    print(f'')
    print(f'parameters')
    print(f'----------')
    print(f'problem number = {PROBLEM_N}')
    print(f'relative tolerance for CG = {RTOL_CG}')
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
        f=problem.f,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR)
    stiffness_matrix = get_general_stiffness_matrix(
        coordinates=coordinates,
        elements=elements,
        a_11=problem.a_11,
        a_12=problem.a_12,
        a_21=problem.a_21,
        a_22=problem.a_22,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR)
    mass_matrix = get_mass_matrix(
        coordinates=coordinates,
        elements=elements)
    lhs_matrix = csr_matrix(
        mass_matrix + problem.c * stiffness_matrix)

    galerkin_solution = np.zeros(n_vertices)
    galerkin_solution[free_nodes] = spsolve(
        A=lhs_matrix[free_nodes, :][:, free_nodes],
        b=rhs_vector[free_nodes])

    current_iterate = np.copy(galerkin_solution)

    coordinates, elements, boundaries, current_iterate = \
        refineNVB(
            coordinates=coordinates,
            elements=elements,
            marked_elements=marked,
            boundary_conditions=boundaries,
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
            a_11=problem.a_11,
            a_12=problem.a_12,
            a_21=problem.a_21,
            a_22=problem.a_22,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)
        mass_matrix = get_mass_matrix(
            coordinates=coordinates,
            elements=elements)

        lhs_matrix = csr_matrix(
            general_stiffness_matrix + problem.c * mass_matrix)
        rhs_vector = get_right_hand_side(
            coordinates=coordinates,
            elements=elements,
            f=problem.f,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        # compute the Galerkin solution on current mesh
        # ---------------------------------------------
        current_iterate = np.zeros(n_vertices)

        lhs_reduced = lhs_matrix[free_nodes, :][:, free_nodes]
        diagonal = lhs_reduced.diagonal()
        M = diags(diagonals=1./diagonal)

        iteration_counter = IterationCounterCallback()

        current_iterate_on_free_nodes, _ = cg(
            A=lhs_reduced,
            b=rhs_vector[free_nodes],
            x0=current_iterate[free_nodes],
            M=M,
            rtol=RTOL_CG,
            callback=iteration_counter)

        current_iterate[free_nodes] = current_iterate_on_free_nodes

        energy_norm_squared = current_iterate.dot(
            lhs_matrix.dot(current_iterate))
        print(
            f'nDOF = {n_dofs}, '
            f'converged CG approximation energy norm squared = {energy_norm_squared}, '
            f'n_iterations = {iteration_counter.n_iterations}')

        # stop right before refining if maximum number of DOFs is reached
        if n_dofs >= n_max_dofs:
            print(
                f'Maximum number of DOFs ({n_max_dofs})'
                'reached, stopping iteration.')
            break

        marked = np.ones(elements.shape[0], dtype=bool)
        coordinates, elements, boundaries, current_iterate = \
            refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked,
                boundary_conditions=boundaries)


class IterationCounterCallback:
    n_iterations: int

    def __init__(self):
        self.n_iterations = 0

    def __call__(self, *args, **kwds):
        self.n_iterations += 1


if __name__ == '__main__':
    main()
