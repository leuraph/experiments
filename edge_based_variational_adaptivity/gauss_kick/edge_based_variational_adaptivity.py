import numpy as np
from p1afempy.data_structures import CoordinatesType, ElementsType, BoundaryConditionType
from scipy.sparse import csr_matrix
from tqdm import tqdm
from p1afempy.mesh import get_local_patch_edge_based
from p1afempy.refinement import refine_single_edge
from p1afempy.solvers import get_stiffness_matrix, get_right_hand_side


def get_energy_gains_and_values_on_new_nodes(
        coordinates: CoordinatesType,
        elements: ElementsType,
        non_boundary_edges: np.ndarray,
        current_iterate: np.ndarray,
        f: BoundaryConditionType,
        verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    returns energy gains based on edge based
    variational adaptivity and corresponding
    values on new nodes

    parameters
    ----------
    coordinates: CoordinatesType
        coordinates of the mesh at hand
    elements: ElementsType
        elements of the mesh at hand
    non_boundary_edges: np.ndarray
        non-boundary edges of the mesh, i.e.
        `non_boundary_edges[k]` represents the
        k-th edge (i, j), order does not matter
        but edges must be unique i.e. if
        `non_boundary_edges[k] == (i, j)` then
        there must be no `k'` such that
        `non_boundary_edges[k'] == (j, i)`
    current_iterate: np.ndarray
        approximation to the solution
    f: BoundaryConditionType
        right hand side function of
        the problem at hand
    verbose: bool = False
        if true, displays a progression bar
        when looping over edges

    returns
    -------
    energy_gains: np.ndarray
        energy gain corresponding to
        bisection of non-boundary edges
    values_on_new_nodes: np.ndarray
        values on new nodes corresponding to
        bisection of non-boundary edges

    notes
    -----
    - this implementation asssumes you are looking at
      the problem given by $-\Delta u = f$
      with homogeneous dirichlet boundary conditions
    - both arrays returned have the shape `(n_non_boundary_edges,)`,
      where `n_non_boundary_edges=non_boundary_edges.shape[0]`
    """

    # we get a new value for each new edge
    n_non_boundary_edges = non_boundary_edges.shape[0]
    values_on_new_nodes = np.zeros(n_non_boundary_edges)
    energy_gains = np.zeros(n_non_boundary_edges)

    # building the global stiffness matrix / rhs vector
    lhs_matrix = csr_matrix(get_stiffness_matrix(
        coordinates=coordinates, elements=elements))
    rhs_vector = get_right_hand_side(
        coordinates=coordinates, elements=elements, f=f)

    # computing global terms before looping over all edges
    L_1 = rhs_vector.dot(
        current_iterate)
    A_11 = current_iterate.dot(
        lhs_matrix.dot(
            current_iterate))

    for k, non_boundary_edge in enumerate(
            tqdm(non_boundary_edges, disable=not verbose)):

        local_elements, local_coordinates, \
            local_iterate, local_edge_indices = \
            get_local_patch_edge_based(
                elements=elements,
                coordinates=coordinates,
                current_iterate=current_iterate,
                edge=non_boundary_edge)
        tmp_local_coordinates, tmp_local_elements, \
            tmp_local_solution =\
            refine_single_edge(
                coordinates=local_coordinates,
                elements=local_elements,
                edge=local_edge_indices,
                to_embed=local_iterate)
        tmp_stiffness_matrix = csr_matrix(get_stiffness_matrix(
            coordinates=tmp_local_coordinates,
            elements=tmp_local_elements))
        tmp_rhs_vector = get_right_hand_side(
            coordinates=tmp_local_coordinates,
            elements=tmp_local_elements, f=f)

        # building the local 2x2 system
        A_12 = tmp_stiffness_matrix.dot(tmp_local_solution)[-1]
        A_22 = tmp_stiffness_matrix[-1, -1]

        L_2 = tmp_rhs_vector[-1]

        detA = (A_11 * A_22 - A_12 * A_12)

        alpha = (A_22 * L_1 - A_12 * L_2)/detA
        beta = (-A_12 * L_1 + A_11 * L_2)/detA

        dE = 0.5*(
            (alpha-1)**2 * A_11
            + 2.*(alpha-1)*beta*A_12
            + beta**2 * A_22)
        energy_gains[k] = dE

        i, j = local_edge_indices
        values_on_new_nodes[k] = beta + 0.5 * (
            local_iterate[i] + local_iterate[j])

    return energy_gains, values_on_new_nodes


def get_energy_gains(
        coordinates: CoordinatesType,
        elements: ElementsType,
        non_boundary_edges: np.ndarray,
        current_iterate: np.ndarray,
        f: BoundaryConditionType,
        verbose: bool = False) -> np.ndarray:
    energy_gains, _ = get_energy_gains_and_values_on_new_nodes(
        coordinates=coordinates,
        elements=elements,
        non_boundary_edges=non_boundary_edges,
        current_iterate=current_iterate,
        f=f,
        verbose=verbose)
    return energy_gains
