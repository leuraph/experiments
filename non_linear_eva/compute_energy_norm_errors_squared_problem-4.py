from load_save_dumps import load_dump, dump_object
from pathlib import Path
from tqdm import tqdm
import argparse
from triangle_cubature.cubature_rule import CubatureRuleEnum
from p1afempy.solvers import \
    get_general_stiffness_matrix, get_right_hand_side
from scipy.sparse import csr_matrix
from problems import get_problem
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    problem_4 = get_problem(number=4)

    base_result_path = Path(args.path)

    def u(r):
        x, y = r[:, 0], r[:, 1]
        radii = np.sqrt(x**2 + y**2)
        return 2. * radii**(-4. / 3.) * x * y *(1 - x**2) * (1 - y**2)

    # reference value of |u|^2_a
    # corresponding to graded mesh
    # hmax = 0.00025, n_vertices = 14'678'654
    # and quadrature rule with degree of exactness equal to 6
    energy_norm_squared_exact = 4.687722548683771

    for path_to_n_dofs in tqdm(list(base_result_path.iterdir())):
        if not path_to_n_dofs.is_dir():
            continue
        
        # specifying paths
        # ----------------
        path_to_elements = path_to_n_dofs / Path('elements.pkl')
        path_to_coordinates = path_to_n_dofs / Path('coordinates.pkl')
        path_to_last_iterate = path_to_n_dofs \
            / Path('last_iterate.pkl')

        # loading data
        # ------------
        elements = load_dump(path_to_dump=path_to_elements)
        coordinates = load_dump(path_to_dump=path_to_coordinates)
        last_iterate = load_dump(
            path_to_dump=path_to_last_iterate)
        
        stiffness_matrix = csr_matrix(get_general_stiffness_matrix(
            coordinates=coordinates,
            elements=elements,
            a_11=problem_4.a_11,
            a_12=problem_4.a_12,
            a_21=problem_4.a_21,
            a_22=problem_4.a_22,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR))
        
        right_hand_side = get_right_hand_side(
            coordinates=coordinates,
            elements=elements,
            f=problem_4.f,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)
        
        phiu_load_vector = get_right_hand_side(
            coordinates=coordinates,
            elements=elements,
            f=lambda r: problem_4.phi(u(r)),
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)
        
        # computing
        # |u - u*|^2_a =
        # |u|^2_a + |u*|^2_a - 2 (int f u* - int phi(u) u*)
        energy_norm_error_squared = (
            energy_norm_squared_exact
            +
            last_iterate.dot(stiffness_matrix.dot(last_iterate))
            - 2.*(
                right_hand_side.dot(last_iterate)
                -
                phiu_load_vector.dot(last_iterate)
            )
        )

        if energy_norm_error_squared < 0:
            raise RuntimeError('this should be positive!!!')

        # saving the energy norm error to disk
        # ------------------------------------
        dump_object(
            obj=energy_norm_error_squared,
            path_to_file=(
                path_to_n_dofs /
                Path('energy_norm_error_squared.pkl')))


if __name__ == '__main__':
    main()
