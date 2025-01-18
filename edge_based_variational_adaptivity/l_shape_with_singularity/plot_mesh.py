from p1afempy.mesh import show_mesh
from load_save_dumps import load_dump
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
                        help="path to the mesh files")
    args = parser.parse_args()

    base_path = Path(args.path)
    path_to_coordinates = base_path / 'coordinates.pkl'
    path_to_elements = base_path / 'elements.pkl'

    coordinates = load_dump(path_to_dump=path_to_coordinates)
    elements = load_dump(path_to_dump=path_to_elements)

    show_mesh(coordinates=coordinates, elements=elements)


if __name__ == '__main__':
    main()
