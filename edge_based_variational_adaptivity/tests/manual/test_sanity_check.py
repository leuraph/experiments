import numpy as np
import unittest
import random
from pathlib import Path
from p1afempy import io_helpers, solvers, mesh
from p1afempy.data_structures import \
    CoordinatesType, ElementsType, BoundaryType
from p1afempy import refinement
from p1afempy.refinement import refine_single_edge
from p1afempy.mesh import provide_geometric_data, get_local_patch_edge_based
from p1afempy.solvers import get_right_hand_side, get_stiffness_matrix
from ismember import is_row_in
from scipy.sparse import csr_matrix
from tqdm import tqdm


def test_function(x: float, y: float) -> float:
    x_inside_domain = -1. < x < 1.
    y_inside_domain = -1. < y < 1.
    if (not x_inside_domain) or (not y_inside_domain):
        return 0.
    return min(1. - abs(x), 1. - abs(y))


def evaluate_energy_on_mesh(
        coordinates: CoordinatesType,
        elements: ElementsType,
        test_function_vector: np.ndarray = np.array([])) -> float:

    stiffness_matrix = solvers.get_stiffness_matrix(
        coordinates=coordinates,
        elements=elements)

    # if test function vector was not passed, calculate the exact value
    if not test_function_vector.size:
        test_function_vector = np.array([
            test_function(x, y) for (x, y) in coordinates])

    return test_function_vector.dot(stiffness_matrix.dot(test_function_vector))


class SanityChecks(unittest.TestCase):
    """
    These tests constitute some sort of sanity checks to check
    - stiffness matrix assembly
    - mesh refinement strategies

    idea
    ----
    The idea is to compute the discrete version of the energy
    E(u) := a(u, u)
    of a function u(x, y) that is exactly represented already
    on the initial mesh. In this way, we can check the interplay
    of stiffness matrix assembly and mesh refinement by checking
    the computed energy E:= x.T * A * x with the exact value for
    the initial mesh and all subsequent refined meshes thereof.

    implementation
    --------------
    we choose
    - Omega := {(x, y) | -1 < x, y < 1}
    - u(x, y) := min{1-|x|, 1-|y|}
    - an initial mesh that already allows for exact approximation
      of u on Omega
      (for details, see the input data in `tests/data/sanity_check`).
    - the expected energy of u is then given by E(u) = 4.
    """

    @staticmethod
    def get_initial_mesh() -> tuple[CoordinatesType,
                                    ElementsType,
                                    BoundaryType]:
        path_to_coordinates = Path('tests/data/sanity_check/coordinates.dat')
        path_to_elements = Path('tests/data/sanity_check/elements.dat')
        path_to_dirichlet = Path('tests/data/sanity_check/dirichlet.dat')
        coordinates, elements = io_helpers.read_mesh(
            path_to_coordinates=path_to_coordinates,
            path_to_elements=path_to_elements)
        dirichlet = io_helpers.read_boundary_condition(
            path_to_boundary=path_to_dirichlet)
        return coordinates, elements, dirichlet

    def test_refine_nvb_bookkeeping(self) -> None:
        coordinates, elements, dirichlet = SanityChecks.get_initial_mesh()

        # initial vector of values to be interpolated on refined meshes
        solution = np.array([test_function(x, y) for (x, y) in coordinates])

        boundaries = [dirichlet]
        expected_energy = 4.
        n_refinements = 5

        for _ in range(n_refinements):
            element_to_edges, edge_to_nodes, boundaries_to_edges =\
                provide_geometric_data(
                    elements=elements,
                    boundaries=boundaries)

            n_boundaries = edge_to_nodes.shape[0]

            edge_to_nodes_flipped = np.column_stack(
                [edge_to_nodes[:, 1], edge_to_nodes[:, 0]])
            boundary = np.logical_or(
                is_row_in(edge_to_nodes, boundaries[0]),
                is_row_in(edge_to_nodes_flipped, boundaries[0])
            )
            non_boundary = np.logical_not(boundary)
            non_boundary_edges = edge_to_nodes[non_boundary]
            n_non_boundary_edges = non_boundary_edges.shape[0]

            # we get a new value for each new edge
            values_on_new_edges = np.zeros(n_boundaries)
            values_on_new_edges_non_boundary = np.zeros(n_non_boundary_edges)

            for k, non_boundary_edge in enumerate(tqdm(non_boundary_edges)):

                r1 = coordinates[non_boundary_edge[0], :]
                r2 = coordinates[non_boundary_edge[1], :]

                midpoint = 0.5*(r1 + r2)

                x = midpoint[0]
                y = midpoint[1]

                beta = test_function(x, y)

                values_on_new_edges_non_boundary[k] = beta

            values_on_new_edges[non_boundary] = \
                values_on_new_edges_non_boundary

            solution = np.hstack([solution, values_on_new_edges])

            # mark all elements for refinement
            marked_elements = np.arange(elements.shape[0])

            # perform refinement
            coordinates, elements, boundaries, _ = refinement.refineNVB(
                coordinates=coordinates,
                elements=elements,
                marked_elements=marked_elements,
                boundary_conditions=boundaries)

            stiffness_matrix = csr_matrix(get_stiffness_matrix(
                coordinates=coordinates, elements=elements))

            computed_energy = solution.dot(stiffness_matrix.dot(solution))

            self.assertEqual(expected_energy, computed_energy)


if __name__ == '__main__':
    unittest.main()
