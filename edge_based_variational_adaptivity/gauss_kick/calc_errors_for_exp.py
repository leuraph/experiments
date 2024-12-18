from configuration import grad_u
from load_save_dumps import load_dump, dump_object
from pathlib import Path
from tqdm import tqdm
import argparse
from iterative_methods.energy_norm import calculate_energy_norm_error
from triangle_cubature.cubature_rule import CubatureRuleEnum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    base_result_path = Path(args.path)

    # Cubature rule used for approximating
    # energy norm distance to exact solution
    cubature_rule = CubatureRuleEnum.SMPLX1

    for path_to_n_dofs in tqdm(list(base_result_path.iterdir())):
        if path_to_n_dofs.is_dir():
            path_to_elements = path_to_n_dofs / Path('elements.pkl')
            path_to_coordinates = path_to_n_dofs / Path('coordinates.pkl')
            path_to_approximate_solution = path_to_n_dofs \
                / Path('solution.pkl')
            path_to_exact_solution = path_to_n_dofs \
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
                    path_to_n_dofs /
                    Path('energy_norm_error_squared.pkl')))
            dump_object(
                obj=energy_norm_error_squared_exact,
                path_to_file=(
                    path_to_n_dofs /
                    Path('energy_norm_error_squared_exact.pkl')))

if __name__ == '__main__':
    main()
