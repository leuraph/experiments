import numpy as np
from p1afempy.data_structures import ElementsType, CoordinatesType


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
