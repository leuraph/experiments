import pickle
import argparse
from pathlib import Path
from matplotlib import cm
import matplotlib.pyplot as plt
from load_save_dumps import load_dump


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", type=str, required=True,
                        help="path to the solution file")
    parser.add_argument("--coordinates", type=str, required=True,
                        help="path to the solution's coordinates file")
    parser.add_argument("-o", type=str, required=False,
                        default='solution.pdf',
                        help="path to the outputted plot")
    args = parser.parse_args()

    path_to_solution = Path(args.solution)
    path_to_coordinates = Path(args.coordinates)
    out_path = Path(args.o)

    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 12

    coordinates = load_dump(path_to_dump=path_to_coordinates)
    solution = load_dump(path_to_dump=path_to_solution)

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
    ax.set_zticks([0, 0.02, 0.04, 0.06])

    # Show and save the plot
    # fig.savefig(out_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
