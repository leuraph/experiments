from p1afempy.mesh import show_mesh
from load_save_dumps import load_dump
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm


def show_solution(coordinates, solution):
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 12

    x_coords, y_coords = zip(*coordinates)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points with scalar values as colors (adjust colormap as needed)
    _ = ax.plot_trisurf(x_coords, y_coords, solution, linewidth=0.2,
                        antialiased=True, cmap=cm.viridis)
    # Add labels to the axes
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    #ax.set_zticks([0, 0.02, 0.04, 0.06])

    # Show and save the plot
    # fig.savefig(out_path, dpi=300)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
                        help="path to the mesh files")
    args = parser.parse_args()

    base_path = Path(args.path)
    path_to_coordinates = base_path / 'coordinates.pkl'
    path_to_elements = base_path / 'elements.pkl'
    path_to_solution = base_path / 'galerkin_solution.pkl'

    coordinates = load_dump(path_to_dump=path_to_coordinates)
    elements = load_dump(path_to_dump=path_to_elements)
    galerkin_solution = load_dump(path_to_dump=path_to_solution)

    # show_mesh(coordinates=coordinates, elements=elements)
    show_solution(coordinates=coordinates, solution=galerkin_solution)


if __name__ == '__main__':
    main()
