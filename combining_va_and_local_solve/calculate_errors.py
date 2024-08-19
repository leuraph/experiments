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
    parser.add_argument("--fudge", type=float, required=True)
    parser.add_argument("--experiment", type=int, required=True)
    args = parser.parse_args()

    theta = args.theta
    fudge = args.fudge
    experiment = args.experiment

    base_result_path = Path(
        f'results/experiment_{experiment}/theta-{theta}_fudge-{fudge}')

    for path_to_result in tqdm(list(base_result_path.iterdir())):
        path_to_solution = path_to_result / Path('solution.pkl')
        path_to_elements = path_to_result / Path('elements.pkl')
        path_to_coordinates = path_to_result / Path('coordinates.pkl')

        solution = load_dump(path_to_dump=path_to_solution)
        elements = load_dump(path_to_dump=path_to_elements)
        coordinates = load_dump(path_to_dump=path_to_coordinates)

        energy_norm_error_squared = calculate_energy_norm_error(
            current_iterate=solution,
            gradient_u=grad_u,
            elements=elements,
            coordinates=coordinates,
            cubature_rule=CubatureRuleEnum.SMPLX1)

        dump_object(
            obj=energy_norm_error_squared,
            path_to_file=path_to_result / Path('energy_norm_error.pkl'))


if __name__ == '__main__':
    main()