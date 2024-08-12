from experiment_setup import get_exact_galerkin_solution
from load_save_dumps import load_dump, dump_object
from pathlib import Path
from tqdm import tqdm
import argparse
from p1afempy.solvers import get_stiffness_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--fudge", type=float, required=True)
    args = parser.parse_args()

    theta = args.theta
    fudge = args.fudge

    base_result_path = Path(
        f'results/experiment_1/theta-{theta}_fudge-{fudge}')

    for path_to_result in tqdm(list(base_result_path.iterdir())):
        path_to_solution = path_to_result / Path('solution.pkl')
        path_to_elements = path_to_result / Path('elements.pkl')
        path_to_coordinates = path_to_result / Path('coordinates.pkl')
        path_to_boundaries = path_to_result / Path('boundaries.pkl')

        solution = load_dump(path_to_dump=path_to_solution)
        elements = load_dump(path_to_dump=path_to_elements)
        coordinates = load_dump(path_to_dump=path_to_coordinates)
        boundaries = load_dump(path_to_dump=path_to_boundaries)
        dirichlet = boundaries[0]

        stiffness_matrix = get_stiffness_matrix(
            coordinates=coordinates, elements=elements)

        exact_galerkin_solution = get_exact_galerkin_solution(
            coordinates=coordinates, elements=elements, boundaries=[dirichlet])

        du = exact_galerkin_solution - solution
        energy_norm_error_squared = du.dot(stiffness_matrix.dot(du))

        dump_object(
            obj=energy_norm_error_squared,
            path_to_file=path_to_result / Path('energy_norm_error.pkl'))


if __name__ == '__main__':
    main()
