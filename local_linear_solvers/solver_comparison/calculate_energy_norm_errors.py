from experiment_setup import get_exact_galerkin_solution
from load_save_dumps import load_dump, dump_object
from pathlib import Path
from tqdm import tqdm
import argparse
from p1afempy.solvers import get_stiffness_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, required=True,
                        help="which experiment do you want to process?"
                        " Available: 1, 2, 3")
    parser.add_argument("--theta", type=float, required=False)
    args = parser.parse_args()
    experiment_number: int = args.experiment

    base_mesh_path = Path(f'results/{experiment_number}/mesh')
    path_to_coordinates = base_mesh_path / Path('coordinates.pkl')
    path_to_elements = base_mesh_path / Path('elements.pkl')
    path_to_dirichlet = base_mesh_path / Path('dirichlet.pkl')

    if experiment_number == 1:
        base_result_paths = [
            Path(f'results/{experiment_number}/local_jacobi'),
            Path(f'results/{experiment_number}/local_block_jacobi'),
            Path(f'results/{experiment_number}/local_gauss_seidel'),
            Path(f'results/{experiment_number}/'
                'local_context_solver_non_simultaneous'),
            Path(f'results/{experiment_number}/local_context_solver_simultaneous')
        ]
    else:
        base_result_paths = [
            Path(f'results/{experiment_number}/{args.theta}/local_jacobi'),
            Path(f'results/{experiment_number}/{args.theta}/local_block_jacobi'),
            Path(f'results/{experiment_number}/{args.theta}/local_gauss_seidel'),
            Path(f'results/{experiment_number}/{args.theta}/local_context_solver_non_simultaneous'),
            Path(f'results/{experiment_number}/{args.theta}/local_context_solver_simultaneous')
        ]

    coordinates = load_dump(path_to_dump=path_to_coordinates)
    elements = load_dump(path_to_dump=path_to_elements)
    dirichlet = load_dump(path_to_dump=path_to_dirichlet)

    stiffness_matrix = get_stiffness_matrix(
        coordinates=coordinates, elements=elements)

    exact_galerkin_solution = get_exact_galerkin_solution(
        coordinates=coordinates, elements=elements, boundaries=[dirichlet])

    for base_result_path in base_result_paths:
        print(f'processing directory {base_result_path}')
        for path_to_solution in tqdm(list((
                base_result_path / Path('solutions')).iterdir())):
            n_local_refienements = int(path_to_solution.stem)
            solution = load_dump(path_to_dump=path_to_solution)
            du = exact_galerkin_solution - solution
            energy_norm_error_squared = du.dot(stiffness_matrix.dot(du))

            dump_object(
                obj=energy_norm_error_squared,
                path_to_file=base_result_path / Path(
                    'energy_norm_errors') / Path(
                        f'{n_local_refienements}.pkl'))


if __name__ == '__main__':
    main()
