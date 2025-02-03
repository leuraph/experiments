from experiment_setup import grad_u
from load_save_dumps import load_dump, dump_object
from pathlib import Path
from tqdm import tqdm
import argparse
from iterative_methods.energy_norm import calculate_energy_norm_error
from triangle_cubature.cubature_rule import CubatureRuleEnum
import re


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    base_result_path = Path(args.path)

    # extracting the experiment number as integer
    pattern = r"experiment_(\d+)"
    match = re.search(pattern, str(base_result_path))
    experiment_number = int(match.group(1))

    print(f'processing some results for experiment number {experiment_number}')
    print('------------------------------------------------')

    if experiment_number in [1, 2, 11]:

        perform_calculation_for_experiment_without_sweeps(
            base_result_path)

    if experiment_number in [3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17]:

        perform_calculations_for_experiment_with_sweeps(
            base_result_path)


def perform_calculation_for_experiment_without_sweeps(
        base_result_path):
    # Cubature rule used for approximating
    # energy norm distance to exact solution
    cubature_rule = CubatureRuleEnum.SMPLX1

    for path_to_n_sweep_dir in tqdm(list(base_result_path.iterdir())):
        if path_to_n_sweep_dir.is_dir():
            path_to_elements = path_to_n_sweep_dir / Path('elements.pkl')
            path_to_coordinates = path_to_n_sweep_dir / Path('coordinates.pkl')
            path_to_approximate_solution = path_to_n_sweep_dir \
                / Path('solution.pkl')
            path_to_exact_solution = path_to_n_sweep_dir \
                / Path('exact_solution.pkl')

            elements = load_dump(path_to_dump=path_to_elements)
            coordinates = load_dump(path_to_dump=path_to_coordinates)
            approximate_solution = load_dump(
                path_to_dump=path_to_approximate_solution)
            exact_solution = load_dump(
                path_to_dump=path_to_exact_solution)

            energy_norm_error_squared_approximate = \
                calculate_energy_norm_error(
                    current_iterate=approximate_solution,
                    gradient_u=grad_u,
                    elements=elements,
                    coordinates=coordinates,
                    cubature_rule=cubature_rule)

            energy_norm_error_squared_exact = calculate_energy_norm_error(
                current_iterate=exact_solution,
                gradient_u=grad_u,
                elements=elements,
                coordinates=coordinates,
                cubature_rule=cubature_rule)

            dump_object(
                obj=energy_norm_error_squared_approximate,
                path_to_file=(
                    path_to_n_sweep_dir /
                    Path('energy_norm_error_squared.pkl')))
            dump_object(
                obj=energy_norm_error_squared_exact,
                path_to_file=(
                    path_to_n_sweep_dir /
                    Path('energy_norm_error_squared_exact.pkl')))


def perform_calculations_for_experiment_with_sweeps(
        base_result_path):
    # Cubature rule used for approximating
    # energy norm distance to exact solution
    cubature_rule = CubatureRuleEnum.SMPLX1

    for path_to_ndof_dir in list(base_result_path.iterdir()):
        if path_to_ndof_dir.is_dir():
            print(f'considering subdirectory: {path_to_ndof_dir.name}')
            path_to_elements = path_to_ndof_dir / Path('elements.pkl')
            path_to_coordinates = path_to_ndof_dir / Path('coordinates.pkl')
            path_to_exact_solution = (
                path_to_ndof_dir / Path('exact_solution.pkl'))
            elements = load_dump(path_to_dump=path_to_elements)
            coordinates = load_dump(path_to_dump=path_to_coordinates)
            exact_solution = load_dump(path_to_dump=path_to_exact_solution)

            energy_norm_error_squared_exact = calculate_energy_norm_error(
                current_iterate=exact_solution,
                gradient_u=grad_u,
                elements=elements,
                coordinates=coordinates,
                cubature_rule=cubature_rule)

            dump_object(
                obj=energy_norm_error_squared_exact,
                path_to_file=(
                    path_to_ndof_dir /
                    Path('energy_norm_error_squared_exact.pkl')))

            for n_sweep_dir in tqdm(list(path_to_ndof_dir.iterdir())):
                if n_sweep_dir.is_dir():
                    path_to_solution = n_sweep_dir / Path('solution.pkl')
                    solution = load_dump(path_to_dump=path_to_solution)

                    energy_norm_error_squared = calculate_energy_norm_error(
                        current_iterate=solution,
                        gradient_u=grad_u,
                        elements=elements,
                        coordinates=coordinates,
                        cubature_rule=cubature_rule)

                    dump_object(
                        obj=energy_norm_error_squared,
                        path_to_file=(
                            n_sweep_dir /
                            Path('energy_norm_error_squared.pkl')))

if __name__ == '__main__':
    main()
