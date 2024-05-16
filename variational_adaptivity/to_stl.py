import numpy as np
from stl import mesh
from pathlib import Path
import pickle
import sys
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
                        help="path to directory holding the mesh")
    parser.add_argument("-o", type=str, required=False,
                        default='mesh.stl',
                        help="path to the outputted STL file")
    args = parser.parse_args()

    base_path = Path(args.path)
    path_to_elements = base_path / Path('elements.pkl')
    path_to_coordinates = base_path / Path('coordinates.pkl')
    path_to_solution = base_path / Path('solution.pkl')

    with open(path_to_elements, mode='rb') as file:
        elements = pickle.load(file=file)
    with open(path_to_coordinates, mode='rb') as file:
        coordinates = pickle.load(file=file)
    with open(path_to_solution, mode='rb') as file:
        solution = pickle.load(file=file)

    vertices = np.column_stack([coordinates, solution])
    faces = elements

    # Create the mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j],:]

    # Write the mesh to STL file
    cube.save(args.o)


if __name__ == '__main__':
    main()
