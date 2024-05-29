import pickle
import argparse
from pathlib import Path
from matplotlib import cm
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
                        help="path to the experiment's results directory")
    parser.add_argument("-o", type=str, required=False,
                        default='solution.pdf',
                        help="path to the outputted plot")
    args = parser.parse_args()

    base_path = Path(args.path)
    out_path = Path(args.o)

    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 12

    path_to_coordinates = base_path / Path('coordinates.pkl')
    path_to_solution = base_path / Path('solution.pkl')

    with open(path_to_coordinates, mode='rb') as file:
        coordinates = pickle.load(file=file)
    with open(path_to_solution, mode='rb') as file:
        solution = pickle.load(file=file)

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
    fig.savefig(out_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
