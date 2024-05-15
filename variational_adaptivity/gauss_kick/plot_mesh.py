import pickle
import sys
from pathlib import Path
import matplotlib.pyplot as plt


def main() -> None:
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 12

    base_path = Path(sys.argv[1])

    path_to_coordinates = base_path / Path('coordinates.pkl')
    path_to_elements = base_path / Path('elements.pkl')

    with open(path_to_coordinates, mode='rb') as file:
        coordinates = pickle.load(file=file)
    with open(path_to_elements, mode='rb') as file:
        elements = pickle.load(file=file)

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$x$-coordinate')
    ax.set_ylabel(r'$y$-coordinate')

    for element in elements:
        r0, r1, r2 = coordinates[element, :]
        ax.plot(
            [r0[0], r1[0], r2[0], r0[0]],
            [r0[1], r1[1], r2[1], r0[1]],
            'black', linewidth=0.3)
    plt.show()


if __name__ == '__main__':
    main()
