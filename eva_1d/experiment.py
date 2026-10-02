import numpy as np

def get_energy_reduction_indicators(
        nodes, f
) -> np.ndarray:
    heights = nodes[1:] - nodes[:-1]
    midpoints = 0.5 * (nodes[:-1] + nodes[1:])
    n_nodes = len(nodes)
    n_elements = n_nodes - 1
    
    indiators = np.zeros(n_elements)
    for k in range(n_elements):
        indiators[k] = (
            0.5 * 
            (
                heights[k] * f(midpoints[k]) + heights[k+1] * f(midpoints[k+1])
            )**2/(
                1/heights[k] + 1/heights[k+1]
            )
        )
    return indiators


def get_load_vector(nodes, f) -> np.ndarray:
    n_nodes = len(nodes)
    midpoints = 0.5 * (nodes[:-1] + nodes[1:])
    n_dofs = n_nodes - 2
    lengths = nodes[:-1] - nodes[1:]
    load_vector = np.zeros(n_dofs)
    for k in range(n_dofs):
        load_vector[k] = (
            0.5 * 
            (
                lengths[k]*f(midpoints[k])
                +
                lengths[k+1]*f(midpoints[k+1])
            )
        )
    
    return load_vector


def get_stiffness_matrix(
    nodes: np.ndarray
) -> np.ndarray:
    n_nodes = len(nodes)
    lengths = nodes[1:] - nodes[:-1]

    diagonal = 1./lengths[:-1] + 1./lengths[1:]
    off_diagonal = - 1/lengths[1:-1]
    
    stiffness_matrix = np.zeros((n_nodes-2, n_nodes-2))
    
    np.fill_diagonal(stiffness_matrix, diagonal)
    np.fill_diagonal(stiffness_matrix[:-1, 1:], off_diagonal)
    np.fill_diagonal(stiffness_matrix[1:, :-1], off_diagonal)

    return stiffness_matrix

def main():
    a, b = 0., 1.
    n_initial_nodes = 5
    nodes = np.linspace(a, b, n_initial_nodes)

    stiffness_matrix = get_stiffness_matrix(nodes)
    print(stiffness_matrix)

    f = lambda x: np.sin(np.pi * x)

    print(get_load_vector(nodes, f))

    print(get_energy_reduction_indicators(nodes, f))

if __name__ == '__main__':
    main()
