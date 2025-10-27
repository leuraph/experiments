from p1afempy.io_helpers import \
    read_boundary_condition, read_elements, read_coordinates
from pathlib import Path
from p1afempy.mesh import show_mesh
import numpy as np
from math import ceil
from tqdm import tqdm
from problems import get_problem
from p1afempy.solvers import \
    get_general_stiffness_matrix, get_right_hand_side, \
        get_load_vector_of_composition_nonlinear_with_fem
from scipy.sparse import csr_matrix
from triangle_cubature.cubature_rule import CubatureRuleEnum
from scipy.sparse.linalg import spsolve
import argparse
import re
from load_save_dumps import dump_object


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--problem", type=int, required=True)
    parser.add_argument("--mesh-path", type=str, required=True,
                        help="path to the folder holding the graded mesh to be used")
    args = parser.parse_args()

    problem_number = args.problem
    alpha = args.alpha
    gamma = args.gamma
    mesh_path = args.mesh_path

    match = re.search(r"hmax-([0-9]*\.?[0-9]+)", mesh_path)
    hmax = float(match.group(1))

    output_path = Path('reference_solutions') / Path(f"hmax-{hmax}_alpha-{alpha}_gamma-{gamma}.pkl")

    problem = get_problem(number=problem_number)
    a_11 = problem.a_11
    a_12 = problem.a_12
    a_21 = problem.a_21
    a_22 = problem.a_22
    phi = problem.phi
    f = problem.f

    mesh_base_path = Path(args.mesh_path)

    path_to_elements = mesh_base_path / Path('elements.dat')
    path_to_coordinates = mesh_base_path / Path('coordinates.dat')
    path_to_dirichlet = mesh_base_path / Path('dirichlet.dat')

    elements = read_elements(
        path_to_elements=path_to_elements, shift_indices=True)
    coordinates = read_coordinates(
        path_to_coordinates=path_to_coordinates)
    dirichlet = read_boundary_condition(
        path_to_boundary=path_to_dirichlet, shift_indices=True)
    
    n_vertices = coordinates.shape[0]
    indices_of_free_nodes = np.setdiff1d(
        ar1=np.arange(n_vertices),
        ar2=np.unique(dirichlet.flatten()))
    free_nodes = np.zeros(n_vertices, dtype=bool)
    free_nodes[indices_of_free_nodes] = 1
    n_dofs = np.sum(free_nodes)

    stifness_matrix = csr_matrix(get_general_stiffness_matrix(
        coordinates=coordinates,
        elements=elements,
        a_11=a_11, a_12=a_12, a_21=a_21, a_22=a_22,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR
    ))

    right_hand_side = get_right_hand_side(
        coordinates=coordinates,
        elements=elements,
        f=f,
        cubature_rule=CubatureRuleEnum.DAYTAYLOR
    )

    # initial guess set to zero
    current_iterate = np.zeros(n_vertices)
    print(current_iterate)
    print(current_iterate[free_nodes])

    n = ceil(gamma * np.log(n_dofs))

    for _ in tqdm(range(n)):
        # this changes in each iteration
        non_linear_load_vector = \
            get_load_vector_of_composition_nonlinear_with_fem(
                f=phi,
                u=current_iterate,
                coordinates=coordinates,
                elements=elements,
                cubature_rule=CubatureRuleEnum.DAYTAYLOR
            )

        current_iterate[free_nodes] = \
            spsolve(
                A = stifness_matrix[free_nodes, :][:, free_nodes],
                b = (
                    (1. - alpha)
                    *
                    stifness_matrix[free_nodes,:][:, free_nodes].dot(
                        current_iterate[free_nodes])
                    +
                    alpha*(
                        right_hand_side[free_nodes]
                        -
                        non_linear_load_vector[free_nodes]
                    )
                )
            )

    dump_object(
        obj=current_iterate,
        path_to_file=output_path)

if __name__ == '__main__':
    main()
