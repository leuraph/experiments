from configuration import grad_u
from load_save_dumps import load_dump, dump_object
from pathlib import Path
from tqdm import tqdm
import argparse
from iterative_methods.energy_norm import calculate_energy_norm_error
from triangle_cubature.cubature_rule import CubatureRuleEnum
from p1afempy.solvers import get_stiffness_matrix
from scipy.sparse import csr_matrix


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

    # Cubature rule used for approximating
    # energy norm distance to exact solution
    cubature_rule = CubatureRuleEnum.DAYTAYLOR

    for path_to_n_dofs in tqdm(list(base_result_path.iterdir())):
        if not path_to_n_dofs.is_dir():
            continue

        # specifying paths
        # ----------------
        path_to_elements = path_to_n_dofs / Path('elements.pkl')
        path_to_coordinates = path_to_n_dofs / Path('coordinates.pkl')
        path_to_approximate_solution = path_to_n_dofs \
            / Path('last_iterate.pkl')
        path_to_exact_solution = path_to_n_dofs \
            / Path('galerkin_solution.pkl')

        # loading data
        # ------------
        elements = load_dump(path_to_dump=path_to_elements)
        coordinates = load_dump(path_to_dump=path_to_coordinates)
        approximate_solution = load_dump(
            path_to_dump=path_to_approximate_solution)
        exact_solution = load_dump(
            path_to_dump=path_to_exact_solution)

        # calculating the energy norm errors
        # ----------------------------------

        # |u_n - u_h|^2
        if approximate_solution is not None:
            energy_norm_error_squared_approximate = \
                calculate_energy_norm_error(
                    current_iterate=approximate_solution,
                    gradient_u=grad_u,
                    elements=elements,
                    coordinates=coordinates,
                    cubature_rule=cubature_rule)

        # |u - u_h|^2 without orthogonality
        energy_norm_error_squared_exact = calculate_energy_norm_error(
            current_iterate=exact_solution,
            gradient_u=grad_u,
            elements=elements,
            coordinates=coordinates,
            cubature_rule=cubature_rule)

        # |u - u_h|^2 with orthogonality
        stiffness_matrix = csr_matrix(get_stiffness_matrix(
            coordinates=coordinates, elements=elements))
        energy_norm_squared_galerkin = exact_solution.dot(
            stiffness_matrix.dot(exact_solution))
        energy_norm_error_squared_exact_with_orthogonality =\
            energy_norm_squared_exact - energy_norm_squared_galerkin

        # saving the energy norm errors to disk
        # -------------------------------------
        if approximate_solution is not None:
            dump_object(
                obj=energy_norm_error_squared_approximate,
                path_to_file=(
                    path_to_n_dofs /
                    Path('energy_norm_error_squared_last_iterate.pkl')))
        dump_object(
            obj=energy_norm_error_squared_exact,
            path_to_file=(
                path_to_n_dofs /
                Path('energy_norm_error_squared_galerkin_without_orthogonality.pkl')))
        dump_object(
            obj=energy_norm_error_squared_exact_with_orthogonality,
            path_to_file=(
                path_to_n_dofs /
                Path(
                    'energy_norm_error_squared_galerkin_with_orthogonality.pkl')))


def calculate_energy_norm_error_squared_galerkin_with_orthogonality(
        base_result_path: Path,
        energy_norm_squared_exact: float,
        verbose: bool = True):
    print('Calculating |u - u_h|_a^2 using Galerkin Orthogonality...')
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
