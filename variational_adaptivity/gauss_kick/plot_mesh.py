import pickle
import sys
from pathlib import Path
from p1afempy.mesh import show_mesh
from variational_adaptivity.utils import plot_result


def main() -> None:
    base_path = Path(sys.argv[1])

    path_to_coordinates = base_path / Path('coordinates.pkl')
    path_to_elements = base_path / Path('elements.pkl')

    with open(path_to_coordinates, mode='rb') as file:
        coordinates = pickle.load(file=file)
    with open(path_to_elements, mode='rb') as file:
        elements = pickle.load(file=file)

    show_mesh(coordinates=coordinates, elements=elements)


if __name__ == '__main__':
    main()
