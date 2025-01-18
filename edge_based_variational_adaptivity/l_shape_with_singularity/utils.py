import numpy as np
from p1afempy.data_structures import ElementsType, CoordinatesType
import matplotlib.pyplot as plt


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
    # ax.set_zticks([0, 0.02, 0.04, 0.06])

    # Show and save the plot
    # fig.savefig(out_path, dpi=300)
    plt.show()


def shuffle_elements(elements: ElementsType) -> ElementsType:
    """
    returns shuffled elements

    note
    ----
    only shuffles row-wise

    example
    -------
    >>> elements = np.array([[1,2,3], [4,5,6], [7,8,9]])
    >>> shuffle_elements(elements)
    >>> [[4,5,6], [1,2,3], [7,8,9]]
    """
    indices = np.arange(elements.shape[0])
    np.random.shuffle(indices)  # shuffles sorted indices in place
    return elements[indices, :]


def distort_coordinates(coordinates: CoordinatesType,
                        delta: float,
                        marked: np.ndarray = None) -> CoordinatesType:
    """
    moves the coordinates marked in a random direction
    with distance at most `delta`

    parameters
    ----------
    coordinates: CoordinatesType
        the coordinates to be moved
    delta: float
        the maximum distance of the coordinates to be moved
    marked: np.ndarray
        booloean array indicating the subset of coordinates to be moved
    """
    n_coordinates = coordinates.shape[0]

    if marked is None:
        marked = np.ones(n_coordinates, dtype=bool)

    n_coordinates_to_be_moved = np.sum(marked, dtype=int)

    components = np.random.randn(n_coordinates_to_be_moved, 2)
    lengths = np.linalg.norm(components, axis=1)
    normalized_vectors = components / lengths[:, None]
    distortions = normalized_vectors * np.random.uniform(0, delta)

    coordinates[marked] = coordinates[marked] + distortions
    return coordinates
