import pickle
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
                        help="path to the experiment's results directory")
    parser.add_argument("-o", type=str, required=False,
                        default='mesh.pdf',
                        help="path to the outputted plot")
    args = parser.parse_args()

    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 12

    base_path = Path(args.path)
    output_path = Path(args.o)

    path_to_coordinates = base_path / Path('coordinates.pkl')
    path_to_elements = base_path / Path('elements.pkl')

    with open(path_to_coordinates, mode='rb') as file:
        coordinates = pickle.load(file=file)
    with open(path_to_elements, mode='rb') as file:
        elements = pickle.load(file=file)

    fig, ax = plt.subplots()
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$', rotation=True)
    ax.set_aspect(1)

    # remove the plot's frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # remove the axes' ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])

    for element in elements:
        r0, r1, r2 = coordinates[element, :]
        ax.plot(
            [r0[0], r1[0], r2[0], r0[0]],
            [r0[1], r1[1], r2[1], r0[1]],
            'black', linewidth=0.1)
    fig.savefig(output_path, dpi=300)

    plt.show()


if __name__ == '__main__':
    main()
