"""
an experiment setup for
u(x, y) = sin(2 pi x) sin(3 pi y)
on (0, 1)^2
"""
from iterative_methods.local_solvers \
    import LocalJacobiSolver
from iterative_methods.local_solvers \
     import LocalBlockJacobiSolver
from iterative_methods.local_solvers \
    import LocalGaussSeidelSolver
from iterative_methods.local_solvers \
    import LocalContextSolver
from scipy.sparse import csr_matrix
from pathlib import Path
from p1afempy.refinement import refineNVB
from p1afempy import solvers
from p1afempy.io_helpers import read_boundary_condition, read_mesh
import numpy as np
import tqdm
from experiment_setup import f
from load_save_dumps import dump_object
from variational_adaptivity.markers import doerfler_marking
from iterative_methods.energy_norm import calculate_energy_norm_error
from triangle_cubature.cubature_rule import CubatureRuleEnum
from experiment_setup import grad_u
import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, required=True,
                        help="dörfler parameter to be used")
    args = parser.parse_args()
    THETA = args.theta
    # -------------------
    # generating the mesh
    # -------------------
    base_path = Path('data')
    path_to_coordinates = base_path / Path('coordinates.dat')
    path_to_elements = base_path / Path('elements.dat')
    path_to_dirichlet = base_path / Path('dirichlet.dat')
    coordinates, elements = read_mesh(path_to_coordinates=path_to_coordinates,
                                      path_to_elements=path_to_elements)
    dirichlet = read_boundary_condition(path_to_boundary=path_to_dirichlet)
    boundaries = [dirichlet]

    # -------------------
    # initiaal refinement
    # -------------------
    n_refinements = 5
    for _ in range(n_refinements):
        coordinates, elements, boundaries, _ = refineNVB(
            coordinates=coordinates,
            elements=elements,
            marked_elements=np.arange(elements.shape[0]),
            boundary_conditions=boundaries)
    n_elements = elements.shape[0]

    # dumping the mesh as it is fixed throughout the experiment
    base_path_to_mesh = Path('results/3/mesh')
    path_to_elements = base_path_to_mesh / Path('elements.pkl')
    path_to_coordinates = base_path_to_mesh / Path('coordinates.pkl')
    path_to_dirichlet = base_path_to_mesh / Path('dirichlet.pkl')
    dump_object(obj=coordinates, path_to_file=path_to_coordinates)
    dump_object(obj=elements, path_to_file=path_to_elements)
    dump_object(obj=boundaries[0], path_to_file=path_to_dirichlet)

    # --------------
    # setup assembly
    # --------------
    n_vertices = coordinates.shape[0]
    indices_of_free_nodes = np.setdiff1d(ar1=np.arange(n_vertices),
                                         ar2=np.unique(boundaries[0].flatten()))
    free_nodes = np.zeros(n_vertices, dtype=bool)
    free_nodes[indices_of_free_nodes] = 1

    # assembly of right hand side
    right_hand_side = solvers.get_right_hand_side(
        coordinates=coordinates,
        elements=elements,
        f=f)

    # assembly of the stiffness matrix
    stiffness_matrix = csr_matrix(solvers.get_stiffness_matrix(
        coordinates=coordinates,
        elements=elements))

    # ------------------
    # iterative solution
    # ------------------

    solvers_to_test = [
        LocalJacobiSolver(
            elements=elements,
            free_nodes=free_nodes,
            lhs_matrix=stiffness_matrix,
            rhs_vector=right_hand_side),
        LocalBlockJacobiSolver(
            elements=elements,
            free_nodes=free_nodes,
            lhs_matrix=stiffness_matrix,
            rhs_vector=right_hand_side),
        LocalGaussSeidelSolver(
            elements=elements,
            free_nodes=free_nodes,
            lhs_matrix=stiffness_matrix,
            rhs_vector=right_hand_side),
        LocalContextSolver(
            elements=elements,
            free_nodes=free_nodes,
            lhs_matrix=stiffness_matrix,
            rhs_vector=right_hand_side,
            simultaneous_solve=False),
        LocalContextSolver(
            elements=elements,
            free_nodes=free_nodes,
            lhs_matrix=stiffness_matrix,
            rhs_vector=right_hand_side,
            simultaneous_solve=True)
    ]

    base_result_paths = [
        Path(f'results/3/{THETA}/local_jacobi/'),
        Path(f'results/3/{THETA}/local_block_jacobi/'),
        Path(f'results/3/{THETA}/local_gauss_seidel/'),
        Path(
            f'results/3/{THETA}'
            '/local_context_solver_non_simultaneous/'),
        Path(f'results/3/{THETA}/local_context_solver_simultaneous/')
    ]

    n_full_sweeps = 100

    for base_result_path, solver in \
            zip(base_result_paths, solvers_to_test):

        current_iterate = np.zeros(n_vertices)
        n_total_local_solves = 0
        energy_norm_errors_squared = []
        start: int = time.process_time_ns()

        for _ in tqdm.tqdm(range(n_full_sweeps)):
            local_energy_differences = []

            for k in range(n_elements):
                _, local_energy_difference = \
                    solver.get_local_increment_and_energy_difference(
                        current_iterate=current_iterate,
                        element=k)
                local_energy_differences.append(local_energy_difference)
            local_energy_differences = np.array(local_energy_differences)

            # ---------------------------------
            # Dörfler marking and global update
            # ---------------------------------
            marked = doerfler_marking(
                input=local_energy_differences,
                theta=THETA)

            reduced_global_element_indices = np.arange(n_elements)[marked]
            reduced_energy_differences = local_energy_differences[marked]

            # sorting such that biggest energy gains come last
            energy_based_sorting = np.argsort(reduced_energy_differences)
            reduced_global_element_indices = reduced_global_element_indices[
                energy_based_sorting]

            for k in reduced_global_element_indices:
                current_iterate = solver.get_next_iterate(
                    current_iterate=current_iterate,
                    element=k)
                n_total_local_solves += 1
            elapsed_time_s = (time.process_time_ns() - start) * 1e-9

            energy_norm_errors_squared.append(
                calculate_energy_norm_error(
                    current_iterate=current_iterate,
                    gradient_u=grad_u,
                    elements=elements,
                    coordinates=coordinates,
                    cubature_rule=CubatureRuleEnum.SMPLX1))

            path_to_solution_dump = base_result_path \
                / Path(f'solutions/{n_total_local_solves}.pkl')
            path_to_time_dump = base_result_path \
                / Path(f'elapsed_times/{n_total_local_solves}.pkl')

            dump_object(
                obj=current_iterate, path_to_file=path_to_solution_dump)
            dump_object(
                obj=elapsed_time_s, path_to_file=path_to_time_dump)


if __name__ == "__main__":
    main()
