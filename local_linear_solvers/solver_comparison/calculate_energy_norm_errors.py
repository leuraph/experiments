from experiment_setup import grad_u
from triangle_cubature.cubature_rule import CubatureRuleEnum
from iterative_methods.energy_norm import calculate_energy_norm_error
from load_save_dumps import load_dump, dump_object
from pathlib import Path
from tqdm import tqdm


def main() -> None:
    base_mesh_path = Path('results/mesh')
    path_to_coordinates = base_mesh_path / Path('coordinates.pkl')
    path_to_elements = base_mesh_path / Path('elements.pkl')

    base_result_paths = [
        Path('results/local_jacobi'),
        Path('results/local_block_jacobi'),
        Path('results/local_gauss_seidel'),
        Path('results/local_context_solver_non_simultaneous'),
        Path('results/local_context_solver_simultaneous')
    ]

    coordinates = load_dump(path_to_dump=path_to_coordinates)
    elements = load_dump(path_to_dump=path_to_elements)

    for base_result_path in base_result_paths:
        print(f'processing directory {base_result_path}')
        for path_to_solution in tqdm(list((
                base_result_path / Path('solutions')).iterdir())):
            n_local_refienements = int(path_to_solution.stem)
            solution = load_dump(path_to_dump=path_to_solution)
            energy_norm_error_squared = calculate_energy_norm_error(
                current_iterate=solution,
                gradient_u=grad_u,
                elements=elements,
                coordinates=coordinates,
                cubature_rule=CubatureRuleEnum.MIDPOINT)

            dump_object(
                obj=energy_norm_error_squared,
                path_to_file=base_result_path / Path(
                    'energy_norm_errors') / Path(
                        f'{n_local_refienements}.pkl'))


if __name__ == '__main__':
    main()
