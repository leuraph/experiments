from load_save_dumps import load_dump, dump_object
from pathlib import Path
from tqdm import tqdm
from p1afempy.solvers import get_stiffness_matrix
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    base_result_path = Path(args.path)

    for path_to_n_dofs_dir in tqdm(list(base_result_path.iterdir())):
        if path_to_n_dofs_dir.is_dir():
            path_to_elements = path_to_n_dofs_dir / Path('elements.pkl')
            path_to_coordinates = path_to_n_dofs_dir / Path('coordinates.pkl')
            path_to_galerkin_solution = path_to_n_dofs_dir \
                / Path('solution.pkl')

            elements = load_dump(path_to_dump=path_to_elements)
            coordinates = load_dump(path_to_dump=path_to_coordinates)
            galerkin_solution = load_dump(
                path_to_dump=path_to_galerkin_solution)

            stiffness_matrix = get_stiffness_matrix(
                coordinates=coordinates, elements=elements)
            energy_norm_squared_galerkin = galerkin_solution.dot(
                stiffness_matrix.dot(galerkin_solution))

            dump_object(
                obj=energy_norm_squared_galerkin,
                path_to_file=(
                    path_to_n_dofs_dir /
                    Path('energy_norm_squared.pkl')))


if __name__ == '__main__':
    main()
