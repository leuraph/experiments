from load_save_dumps import load_dump, dump_object
from pathlib import Path
from tqdm import tqdm
import argparse
from triangle_cubature.cubature_rule import CubatureRuleEnum
from p1afempy.solvers import \
    get_general_stiffness_matrix, get_right_hand_side, get_mass_matrix
from scipy.sparse import csr_matrix
import re
from p1afempy.data_structures import CoordinatesType
import numpy as np


class Square:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def has_coordinates(self, coordinates: CoordinatesType) -> list[bool]:
        x, y = coordinates[:, 0], coordinates[:, 1]
        return (
            (self.x_min < x) &
            (x < self.x_max) &
            (self.y_min < y) &
            (y < self.y_max))


def kappa(coordinates: CoordinatesType):
    omega_1 = Square(0.1, 0.3, 0.1, 0.2)
    omega_2 = Square(0.4, 0.7, 0.1, 0.3)
    omega_3 = Square(0.4, 0.6, 0.5, 0.8)

    in_omega_1 = omega_1.has_coordinates(coordinates)
    in_omega_2 = omega_2.has_coordinates(coordinates)
    in_omega_3 = omega_3.has_coordinates(coordinates)

    # Values for each region
    values = [1e2, 1e4, 1e6]

    # Default value (like `else`)
    default_value = 1.0

    return np.select(
        [in_omega_1, in_omega_2, in_omega_3],
        values, default=default_value)


def f(r: CoordinatesType) -> float:
    """returns ones only"""
    return np.ones(r.shape[0], dtype=float)


def a_11(r: CoordinatesType) -> np.ndarray:
    return - kappa(coordinates=r)


def a_22(r: CoordinatesType) -> np.ndarray:
    return - kappa(coordinates=r)


def a_12(r: CoordinatesType) -> np.ndarray:
    n_vertices = r.shape[0]
    return np.zeros(n_vertices, dtype=float)


def a_21(r: CoordinatesType) -> np.ndarray:
    n_vertices = r.shape[0]
    return np.zeros(n_vertices, dtype=float)


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

    print(f'post-processing results from experiment {experiment_number}')
    calculate_energy_norm_error_squared_last_iterate_to_galerkin(
        base_result_path=base_result_path, verbose=True)
    calculate_energy_norm_error_squared_galerkin_with_orthogonality(
        base_result_path=base_result_path,
        energy_norm_squared_exact=energy_norm_squared_exact,
        verbose=True)
    calculate_energy_norm_error_squared_last_iterate_to_exact(
        base_result_path=base_result_path,
        energy_norm_squared_exact=energy_norm_squared_exact,
        verbose=True)


def calculate_energy_norm_error_squared_last_iterate_to_exact(
        base_result_path: Path,
        energy_norm_squared_exact: float,
        verbose: bool = True) -> None:
    if verbose:
        print(
            'Calculating |u_n^\star - u|_a^2')
    for path_to_n_dofs in tqdm(list(base_result_path.iterdir())):
        if not path_to_n_dofs.is_dir():
            continue

        # specifying paths
        # ----------------
        path_to_elements = path_to_n_dofs / Path('elements.pkl')
        path_to_coordinates = path_to_n_dofs / Path('coordinates.pkl')
        path_to_last_iterate = path_to_n_dofs \
            / Path('last_iterate.pkl')

        # loading data
        # ------------
        elements = load_dump(path_to_dump=path_to_elements)
        coordinates = load_dump(path_to_dump=path_to_coordinates)
        last_iterate = load_dump(
            path_to_dump=path_to_last_iterate)

        # calculating the energy norm errors
        # ----------------------------------
        general_stiffness_matrix = csr_matrix(get_general_stiffness_matrix(
            coordinates=coordinates,
            elements=elements,
            a_11=a_11,
            a_12=a_12,
            a_21=a_21,
            a_22=a_22,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR))
        mass_matrix = get_mass_matrix(
            coordinates=coordinates,
            elements=elements)
        lhs_matrix = general_stiffness_matrix + mass_matrix
        rhs_vector = get_right_hand_side(
            coordinates=coordinates,
            elements=elements,
            f=f,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        energy_last_iterate = (
            0.5 * last_iterate.dot(lhs_matrix.dot(last_iterate))
            - rhs_vector.dot(last_iterate))

        energy_norm_error_squared_last_iterate_to_exact = (
            energy_norm_squared_exact
            + 2. * energy_last_iterate)

        if energy_norm_error_squared_last_iterate_to_exact < 0:
            raise RuntimeError('this should be positive!!!')

        # saving the energy norm error to disk
        # ------------------------------------
        dump_object(
            obj=energy_norm_error_squared_last_iterate_to_exact,
            path_to_file=(
                path_to_n_dofs /
                Path(
                    'energy_norm_error_squared_last_iterate_to_exact.pkl')))


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
        general_stiffness_matrix = csr_matrix(get_general_stiffness_matrix(
            coordinates=coordinates,
            elements=elements,
            a_11=a_11,
            a_12=a_12,
            a_21=a_21,
            a_22=a_22,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR))
        mass_matrix = get_mass_matrix(
            coordinates=coordinates,
            elements=elements)
        lhs_matrix = general_stiffness_matrix + mass_matrix

        du = galerkin_solution - last_iterate
        energy_norm_error_squared_last_iterate_to_galerkin =\
            du.dot(lhs_matrix.dot(du))

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
        general_stiffness_matrix = csr_matrix(get_general_stiffness_matrix(
            coordinates=coordinates,
            elements=elements,
            a_11=a_11,
            a_12=a_12,
            a_21=a_21,
            a_22=a_22,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR))
        mass_matrix = get_mass_matrix(
            coordinates=coordinates,
            elements=elements)
        lhs_matrix = general_stiffness_matrix + mass_matrix

        energy_norm_squared_galerkin = galerkin_solution.dot(
            lhs_matrix.dot(galerkin_solution))

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
