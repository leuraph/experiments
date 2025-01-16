import numpy as np
from p1afempy.refinement import refineNVB
from p1afempy.mesh import provide_geometric_data
from p1afempy.solvers import get_stiffness_matrix, get_right_hand_side
from configuration import f
from scipy.sparse import csr_matrix
from ismember import is_row_in
from edge_based_variational_adaptivity import \
    get_energy_gains_and_values_on_new_nodes, get_energy_gains
from p1afempy.data_structures import ElementsType, CoordinatesType
from triangle_cubature.cubature_rule import CubatureRuleEnum


class ConvergedException(Exception):
    energy_gains: np.ndarray
    last_iterate: np.ndarray

    def __init__(self, energy_gains: np.ndarray, last_iterate: np.ndarray):
        self.energy_gains = energy_gains
        self.last_iterate = last_iterate


class CustomCallBack():
    """
    A callback class for iterative solvers that
    monitors energy changes and refines the mesh
    dynamically based on convergence criteria.

    Attributes:
        n_iterations_done (int):
            The number of single iterations completed.
        batch_size (int):
            The frequency of iterations to perform the callback.
        elements (ElementsType):
            The elements of the mesh.
        coordinates (CoordinatesType):
            The coordinates of the mesh nodes.
        boundaries (list[np.ndarray]):
            List of boundary edges/nodes.
        edges (np.ndarray):
            Array of all edges in the mesh.
        non_boundary_edges (np.ndarray):
            Array of edges not on the boundary.
        free_edges (np.ndarray):
            Boolean mask for free edges.
        free_nodes (np.ndarray):
            Boolean mask for nodes not on the boundary.
        energy_of_last_iterate (float):
            Energy of the last iteration.
        lhs_matrix (csr_matrix):
            The left-hand side matrix for the solver.
        rhs_vector (np.ndarray):
            The right-hand side vector for the solver.
        last_energy_gain_eva (float):
            EVA energy gain of the last iterate.
        last_energy_gains (np.ndarray):
            Array of EVA energy gains (per edge) from the last iteration.
        fudge (float):
            Fudge factor to scale thresholds.
        min_n_iterations_per_mesh (int):
            Minimum iterations required before refining the mesh.
    """
    n_iterations_done: int
    batch_size: int
    elements: ElementsType
    coordinates: CoordinatesType
    boundaries: list[np.ndarray]
    edges: np.ndarray
    non_boundary_edges: np.ndarray
    free_edges: np.ndarray
    free_nodes: np.ndarray
    energy_of_last_iterate: float
    lhs_matrix: csr_matrix
    rhs_vector: np.ndarray
    last_energy_gain_eva: float
    last_energy_gains: np.ndarray
    fudge: float
    min_n_iterations_per_mesh: int

    def __init__(
            self,
            batch_size: int,
            elements: ElementsType,
            coordinates: CoordinatesType,
            boundaries: list[np.ndarray],
            initial_guess: np.ndarray,
            fudge: float,
            min_n_iterations_per_mesh: int) -> None:
        self.n_iterations_done = 0
        self.batch_size = batch_size
        self.elements = elements
        self.coordinates = coordinates
        self.boundaries = boundaries
        self.fudge = fudge
        self.min_n_iterations_per_mesh = min_n_iterations_per_mesh

        # mesh specific setup
        # -------------------

        # (non-boundary) edges
        _, edge_to_nodes, _ = \
            provide_geometric_data(
                elements=elements,
                boundaries=boundaries)

        edge_to_nodes_flipped = np.column_stack(
            [edge_to_nodes[:, 1], edge_to_nodes[:, 0]])
        boundary = np.logical_or(
            is_row_in(edge_to_nodes, self.boundaries[0]),
            is_row_in(edge_to_nodes_flipped, self.boundaries[0])
        )
        non_boundary = np.logical_not(boundary)
        self.edges = edge_to_nodes
        self.non_boundary_edges = edge_to_nodes[non_boundary]

        # free nodes / edges
        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        self.free_edges = non_boundary
        self.free_nodes = free_nodes

        # lhs-matrix / rhs-vector
        self.lhs_matrix = csr_matrix(get_stiffness_matrix(
            coordinates=coordinates, elements=elements))
        self.rhs_vector = get_right_hand_side(
            coordinates=coordinates, elements=elements, f=f,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        # initial energy considerations
        # -----------------------------
        initial_energy = self.calculate_energy(
            current_iterate=initial_guess)

        energy_after_eva, energy_gains_eva = \
            get_energy_after_eva_and_local_energy_gains_eva(
                current_iterate=initial_guess,
                coordinates=coordinates,
                elements=elements,
                boundaries=boundaries,
                edges=self.edges,
                free_edges=self.free_edges,
                f=f)

        self.energy_of_last_iterate = initial_energy
        self.last_energy_gain_eva = initial_energy - energy_after_eva
        self.last_energy_gains = energy_gains_eva

    def perform_callback(
            self,
            current_iterate) -> None:
        pass

    def calculate_energy(self, current_iterate) -> float:
        return (
            0.5*current_iterate.dot(self.lhs_matrix.dot(current_iterate))
            - self.rhs_vector.dot(current_iterate))

    def __call__(self, current_iterate_on_free_nodes) -> None:
        # we know that scipy.sparse.linalg.cg calls this after each iteration
        self.n_iterations_done += 1

        batch_size_reached = self.n_iterations_done % self.batch_size == 0
        min_iterations_reached = (
            self.n_iterations_done > self.min_n_iterations_per_mesh)
        if batch_size_reached and min_iterations_reached:
            # restoring the full iterate
            current_iterate = np.zeros(self.coordinates.shape[0])
            current_iterate[self.free_nodes] = current_iterate_on_free_nodes

            # check if we must continue with iterations
            self.perform_callback(
                current_iterate=current_iterate)


class EnergyComparisonCustomCallback(CustomCallBack):
    """
    After each batch of iteration,
    compares EVA energy gain with energy gain
    associated with another batch of iterations.
    """
    def __init__(
            self, batch_size, elements, coordinates, boundaries,
            initial_guess, fudge, min_n_iterations_per_mesh):
        super().__init__(
            batch_size, elements, coordinates, boundaries,
            initial_guess, fudge, min_n_iterations_per_mesh)

    def perform_callback(
            self,
            current_iterate) -> None:

        current_energy = self.calculate_energy(
            current_iterate=current_iterate)
        energy_gain_iteration = self.energy_of_last_iterate - current_energy

        if self.last_energy_gain_eva > self.fudge * energy_gain_iteration:
            converged_exception = ConvergedException(
                energy_gains=self.last_energy_gains,
                last_iterate=current_iterate)
            raise converged_exception

        energy_after_eva, local_energy_gains_eva = \
            self.get_energy_after_eva_and_local_energy_gains_eva(
                current_iterate=current_iterate)

        energy_gain_eva = current_energy - energy_after_eva

        # keep energy considerations in memory
        self.energy_of_last_iterate = current_energy
        self.last_energy_gains = local_energy_gains_eva
        self.last_energy_gain_eva = energy_gain_eva

    def get_energy_after_eva_and_local_energy_gains_eva(
            self,
            current_iterate: np.ndarray) -> tuple[float, np.ndarray]:
        """
        returns both the new global energy and the local energy gains
        associated to EVA.

        returns
        -------
        energy_after_eva: float
            energy of the iterate constructed out of the `current_iterate`
            and the values on bisected free edges found by EVA
        local_energy_gains_eva: np.ndarray
            the local energy gains on all free edges found by EVA
        """
        local_energy_gains_eva, values_on_new_nodes_non_boundary = \
            get_energy_gains_and_values_on_new_nodes(
                coordinates=self.coordinates,
                elements=self.elements,
                non_boundary_edges=self.non_boundary_edges,
                current_iterate=current_iterate,
                f=f,
                verbose=True)

        # we get a new value on each edge
        values_on_new_nodes = np.zeros(self.edges.shape[0])
        values_on_new_nodes[self.free_edges] = \
            values_on_new_nodes_non_boundary

        current_iterate_after_eva = np.hstack(
            [current_iterate, values_on_new_nodes])

        # mark all elements for refinement
        marked_elements = np.arange(self.elements.shape[0])
        new_coordinates, new_elements, _, _ =\
            refineNVB(
                coordinates=self.coordinates,
                elements=self.elements,
                marked_elements=marked_elements,
                boundary_conditions=self.boundaries)

        new_stiffness_matrix = csr_matrix(
            get_stiffness_matrix(
                coordinates=new_coordinates,
                elements=new_elements))
        new_right_hand_side = get_right_hand_side(
            coordinates=new_coordinates, elements=new_elements, f=f,
            cubature_rule=CubatureRuleEnum.DAYTAYLOR)

        energy_after_eva = 0.5 * current_iterate_after_eva.dot(
            new_stiffness_matrix.dot(current_iterate_after_eva)) \
            - new_right_hand_side.dot(current_iterate_after_eva)

        return energy_after_eva, local_energy_gains_eva


class EnergyTailOffCustomCallback(CustomCallBack):
    """
    After each batch of iterations,
    compares the accumulated energy gain with
    the energy gain associated to another batch of iterations.
    """
    accumulated_energy_gain: float

    def __init__(
            self, batch_size, elements, coordinates,
            boundaries, energy_of_initial_guess,
            eva_energy_gain_of_initial_guess,
            energy_gains_of_initial_guess, fudge,
            min_n_iterations_per_mesh,):
        super().__init__(
            batch_size, elements, coordinates, boundaries,
            energy_of_initial_guess, eva_energy_gain_of_initial_guess,
            energy_gains_of_initial_guess, fudge,
            min_n_iterations_per_mesh)
        self.accumulated_energy_gain = 0.

    def perform_callback(
            self,
            current_iterate) -> None:

        current_energy = self.calculate_energy(
            current_iterate=current_iterate)
        energy_gain_iteration = self.energy_of_last_iterate - current_energy
        self.accumulated_energy_gain += energy_gain_iteration

        if energy_gain_iteration < self.fudge * self.accumulated_energy_gain:
            energy_gains = get_energy_gains(
                coordinates=self.coordinates,
                elements=self.elements,
                non_boundary_edges=self.non_boundary_edges,
                current_iterate=current_iterate,
                f=f,
                verbose=True)

            self.last_energy_gains = energy_gains
            converged_exception = ConvergedException(
                energy_gains=self.last_energy_gains,
                last_iterate=current_iterate)
            raise converged_exception

        # keep energy considerations in memory
        self.energy_of_last_iterate = current_energy
