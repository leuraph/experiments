from experiment_setup import grad_u
from load_save_dumps import load_dump, dump_object
from pathlib import Path
from tqdm import tqdm
import argparse
from iterative_methods.energy_norm import calculate_energy_norm_error
from triangle_cubature.cubature_rule import CubatureRuleEnum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--sweeps", type=int, required=True)
    args = parser.parse_args()

    theta = args.theta
    sweeps = args.sweeps

    base_result_path = Path(
        f'results/experiment_3/theta-{theta}_sweeps-{sweeps}')

    for path_to_ndof_dir in tqdm(list(base_result_path.iterdir())):
        if path_to_ndof_dir.is_dir():
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
                cubature_rule=CubatureRuleEnum.SMPLX1)

            dump_object(
                obj=energy_norm_error_squared_exact,
                path_to_file=(
                    path_to_ndof_dir /
                    Path('energy_norm_error_squared_exact.pkl')))

            for n_sweep_dir in list(path_to_ndof_dir.iterdir()):
                if n_sweep_dir.is_dir():
                    path_to_solution = n_sweep_dir / Path('solution.pkl')
                    solution = load_dump(path_to_dump=path_to_solution)

                    energy_norm_error_squared = calculate_energy_norm_error(
                        current_iterate=solution,
                        gradient_u=grad_u,
                        elements=elements,
                        coordinates=coordinates,
                        cubature_rule=CubatureRuleEnum.SMPLX1)

                    dump_object(
                        obj=energy_norm_error_squared,
                        path_to_file=(
                            n_sweep_dir /
                            Path('energy_norm_error_squared.pkl')))


if __name__ == '__main__':
    main()
