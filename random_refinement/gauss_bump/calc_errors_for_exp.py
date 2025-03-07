from load_save_dumps import load_dump, dump_object
from pathlib import Path
from tqdm import tqdm
import argparse
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

    for path_to_n_dofs in tqdm(list(base_result_path.iterdir())):
        if not path_to_n_dofs.is_dir():
            continue

        # specifying paths
        # ----------------
        path_to_elements = path_to_n_dofs / Path('elements.pkl')
        path_to_coordinates = path_to_n_dofs / Path('coordinates.pkl')
        path_to_exact_solution = path_to_n_dofs \
            / Path('galerkin_solution.pkl')

        # loading data
        # ------------
        elements = load_dump(path_to_dump=path_to_elements)
        coordinates = load_dump(path_to_dump=path_to_coordinates)
        exact_solution = load_dump(
            path_to_dump=path_to_exact_solution)

        # calculating the energy norm error
        # ---------------------------------

        # |u - u_h|^2 with orthogonality
        stiffness_matrix = csr_matrix(get_stiffness_matrix(
            coordinates=coordinates, elements=elements))
        energy_norm_squared_galerkin = exact_solution.dot(
            stiffness_matrix.dot(exact_solution))
        energy_norm_error_squared_exact_with_orthogonality =\
            energy_norm_squared_exact - energy_norm_squared_galerkin

        # saving the energy norm errors to disk
        # -------------------------------------

        dump_object(
            obj=energy_norm_error_squared_exact_with_orthogonality,
            path_to_file=(
                path_to_n_dofs /
                Path(
                    "energy_norm_error_squared_"
                    "galerkin_with_orthogonality.pkl")))


if __name__ == '__main__':
    main()
