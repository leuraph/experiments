import pickle
import sys
from pathlib import Path
from p1afempy.mesh import show_mesh
from variational_adaptivity.utils import plot_result


def main() -> None:
    base_path = Path(sys.argv[1])

    path_to_coordinates = base_path / Path('coordinates.pkl')
    path_to_solution = base_path / Path('solution.pkl')

    with open(path_to_coordinates, mode='rb') as file:
        coordinates = pickle.load(file=file)
    with open(path_to_solution, mode='rb') as file:
        solution = pickle.load(file=file)

    plot_result(scalar_values=solution, vertices=coordinates)


if __name__ == '__main__':
    main()
