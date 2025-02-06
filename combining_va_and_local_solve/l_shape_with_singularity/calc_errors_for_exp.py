from load_save_dumps import load_dump, dump_object
from pathlib import Path
from tqdm import tqdm
import argparse
from iterative_methods.energy_norm import calculate_energy_norm_error
from triangle_cubature.cubature_rule import CubatureRuleEnum
from p1afempy.solvers import get_stiffness_matrix
from scipy.sparse import csr_matrix
import re


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--energy-path", type=str, required=True,
                        help="path to the file holding the numerical value of "
                        "the solution's energy norm squared")
    args = parser.parse_args()

    base_result_path = Path(args.path)

    with open(args.energy_path) as f:
        energy_norm_squared_exact = float(f.readline())

    # extracting the experiment number as integer
    pattern = r"experiment_(\d+)"
    match = re.search(pattern, str(base_result_path))
    experiment_number = int(match.group(1))

    if experiment_number in [2, 3]:
        print(f'post-processing results from experiment {experiment_number}')
        calculate_energy_norm_error_squared_last_iterate_to_galerkin(
            base_result_path=base_result_path, verbose=True)
        calculate_energy_norm_error_squared_galerkin_with_orthogonality(
            base_result_path=base_result_path,
            energy_norm_squared_exact=energy_norm_squared_exact,
            verbose=True)

    else:
        print(
            f'experiment {experiment_number} not covered by this script, '
            'doing nothinng...')


def calculate_energy_norm_error_squared_last_iterate_to_galerkin(
        base_result_path: Path,
        verbose: bool = True) -> None:
    if verbose:
        print(
            'Calculating |u_n^\star - u_h|_a^2')
    for path_to_n_dofs in tqdm(list(base_result_path.iterdir())):
        if not path_to_n_dofs.is_dir():
            continue

        # specifying paths
        # ----------------
        path_to_elements = path_to_n_dofs / Path('elements.pkl')
        path_to_coordinates = path_to_n_dofs / Path('coordinates.pkl')
        path_to_galerkin_solution = path_to_n_dofs \
            / Path('galerkin_solution.pkl')
        path_to_last_iterate = path_to_n_dofs \
            / Path('last_iterate.pkl')

        # loading data
        # ------------
        elements = load_dump(path_to_dump=path_to_elements)
        coordinates = load_dump(path_to_dump=path_to_coordinates)
        galerkin_solution = load_dump(
            path_to_dump=path_to_galerkin_solution)
        last_iterate = load_dump(
            path_to_dump=path_to_last_iterate)

        # calculating the energy norm errors
        # ----------------------------------

        stiffness_matrix = csr_matrix(get_stiffness_matrix(
            coordinates=coordinates, elements=elements))

        du = galerkin_solution - last_iterate
        energy_norm_error_squared_last_iterate_to_galerkin =\
            du.dot(stiffness_matrix.dot(du))

        # saving the energy norm error to disk
        # ------------------------------------
        dump_object(
            obj=energy_norm_error_squared_last_iterate_to_galerkin,
            path_to_file=(
                path_to_n_dofs /
                Path(
                    'energy_norm_error_squared_last_iterate_to_galerkin.pkl')))


def calculate_energy_norm_error_squared_galerkin_with_orthogonality(
        base_result_path: Path,
        energy_norm_squared_exact: float,
        verbose: bool = True) -> None:
    if verbose:
        print(
            'Calculating |u - u_h|_a^2 '
            'using Galerkin Orthogonality...')
    for path_to_n_dofs in tqdm(list(base_result_path.iterdir())):
        if not path_to_n_dofs.is_dir():
            continue

        # specifying paths
        # ----------------
        path_to_elements = path_to_n_dofs / Path('elements.pkl')
        path_to_coordinates = path_to_n_dofs / Path('coordinates.pkl')
        path_to_galerkin_solution = path_to_n_dofs \
            / Path('galerkin_solution.pkl')

        # loading data
        # ------------
        elements = load_dump(path_to_dump=path_to_elements)
        coordinates = load_dump(path_to_dump=path_to_coordinates)
        galerkin_solution = load_dump(
            path_to_dump=path_to_galerkin_solution)

        # calculating the energy norm errors
        # ----------------------------------

        # |u - u_h|_a^2 with orthogonality, i.e.
        # |u - u_h|_a^2 = |u|_a^2 - |u_h|_a^2
        stiffness_matrix = csr_matrix(get_stiffness_matrix(
            coordinates=coordinates, elements=elements))

        energy_norm_squared_galerkin = galerkin_solution.dot(
            stiffness_matrix.dot(galerkin_solution))

        energy_norm_error_squared_exact_with_orthogonality =\
            energy_norm_squared_exact - energy_norm_squared_galerkin

        # saving the energy norm error to disk
        # ------------------------------------
        dump_object(
            obj=energy_norm_error_squared_exact_with_orthogonality,
            path_to_file=(
                path_to_n_dofs /
                Path(
                    'energy_norm_error_squared_galerkin_with_orthogonality.pkl')))


if __name__ == '__main__':
    main()
