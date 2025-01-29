import numpy as np
from p1afempy.mesh import provide_geometric_data
from p1afempy.solvers import get_stiffness_matrix, get_right_hand_side
from configuration import f
from scipy.sparse import csr_matrix
from ismember import is_row_in
from variational_adaptivity.edge_based_variational_adaptivity import \
    get_energy_gains, get_energy_after_eva_and_local_energy_gains_eva
from p1afempy.data_structures import ElementsType, CoordinatesType, \
    BoundaryType
from triangle_cubature.cubature_rule import CubatureRuleEnum


class ConvergedException(Exception):
    energy_gains: np.ndarray
    last_iterate: np.ndarray
    n_iterations_done: int
    energy_history: list[float]

    def __init__(
            self,
            energy_gains: np.ndarray,
            last_iterate: np.ndarray,
            n_iterations_done: int = None,
            energy_history: list[float] = None):
        self.energy_gains = energy_gains
        self.last_iterate = last_iterate
        self.n_iterations_done = n_iterations_done
        self.energy_history = energy_history


class CustomCallBack():
    """
    a callback class for iterative solvers

    Attributes
    ----------
    n_iterations_done (int):
        The number of single iterations completed.
    batch_size (int):
        The frequency of iterations to perform the callback.
    min_n_iterations_per_mesh (int):
        Minimum iterations required before refining the mesh.
    """
    n_iterations_done: int
    batch_size: int
    min_n_iterations_per_mesh: int

    # Mesh
    elements: ElementsType
    coordinates: CoordinatesType
    boundaries: list[np.ndarray]
    # ---
    free_nodes: np.ndarray
    edges: np.ndarray
    non_boundary_edges: np.ndarray
    free_edges: np.ndarray
    # ---
    cubature_rule: CubatureRuleEnum
    lhs_matrix: csr_matrix
    rhs_vector: np.ndarray

    def __init__(
            self,
            batch_size: int,
            min_n_iterations_per_mesh: int,
            elements: ElementsType,
            coordinates: CoordinatesType,
            boundaries: list[BoundaryType],
            cubature_rule: CubatureRuleEnum) -> None:
        self.n_iterations_done = 0
        self.batch_size = batch_size
        self.min_n_iterations_per_mesh = min_n_iterations_per_mesh
        self.cubature_rule = cubature_rule

        # mesh specific setup
        # -------------------
        self.elements = elements
        self.coordinates = coordinates
        self.boundaries = boundaries

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
            cubature_rule=self.cubature_rule)

    def perform_callback(self, current_iterate) -> None:
        pass

    def __call__(self, current_iterate_on_free_nodes) -> None:
        # we know that scipy.sparse.linalg.cg calls this after each iteration
        self.n_iterations_done += 1

        batch_size_reached = self.n_iterations_done % self.batch_size == 0
        min_iterations_reached = (
            self.n_iterations_done >= self.min_n_iterations_per_mesh)
        if batch_size_reached and min_iterations_reached:
            current_iterate = np.zeros(self.coordinates.shape[0], dtype=float)
            current_iterate[self.free_nodes] = current_iterate_on_free_nodes
            # check if we must continue with iterations
            self.perform_callback(current_iterate=current_iterate)

    @staticmethod
    def get_global_iterate_from_iterate_on_free_nodes(
            current_iterate_on_free_nodes: np.ndarray,
            free_nodes: np.ndarray) -> np.ndarray:
        """
        given the iterate on free nodes, restores the global iterate
        on all nodes

        parameters
        ----------
        current_iterate_on_free_nodes: np.ndarray
            iterate on free nodes only
        free_nodes: np.ndarray
            boolean masak indicating free nods
        """
        global_iterate = np.zeros(free_nodes.shape[0])
        global_iterate[free_nodes] = current_iterate_on_free_nodes
        return global_iterate


class EnergyComparisonCustomCallback(CustomCallBack):
    """
    After each batch of iteration,
    compares EVA energy gain with energy gain
    associated with another batch of iterations.

    Attributes
    ----------
    energy_of_last_iterate (float):
        Energy of the last iteration.
    last_energy_gain_eva (float):
        EVA energy gain of the last iterate.
    last_energy_gains (np.ndarray):
        Array of EVA energy gains (per edge) from the last iteration.
    fudge (float):
        Fudge factor to scale thresholds.
    verbose: bool
    """
    last_energy_gain_eva: float
    last_energy_gains: np.ndarray
    energy_of_last_iterate: float
    fudge: float
    verbose: bool

    def __init__(
            self,
            batch_size: int,
            min_n_iterations_per_mesh: int,
            elements: ElementsType,
            coordinates: CoordinatesType,
            boundaries: list[BoundaryType],
            initial_guess: np.ndarray,
            fudge: float,
            cubature_rule: CubatureRuleEnum = CubatureRuleEnum.MIDPOINT,
            verbose: bool = False):
        super().__init__(
            batch_size=batch_size,
            min_n_iterations_per_mesh=min_n_iterations_per_mesh,
            elements=elements,
            coordinates=coordinates,
            boundaries=boundaries,
            cubature_rule=cubature_rule)

        self.fudge = fudge
        self.verbose = verbose

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
                cubature_rule=self.cubature_rule,
                f=f,
                verbose=self.verbose)

        self.energy_of_last_iterate = initial_energy
        self.last_energy_gain_eva = initial_energy - energy_after_eva
        self.last_energy_gains = energy_gains_eva

    def calculate_energy(self, current_iterate) -> float:
        return (
            0.5*current_iterate.dot(self.lhs_matrix.dot(current_iterate))
            - self.rhs_vector.dot(current_iterate))

    def perform_callback(
            self,
            current_iterate) -> None:

        current_energy = self.calculate_energy(
            current_iterate=current_iterate)
        energy_gain_iteration = self.energy_of_last_iterate - current_energy

        if self.last_energy_gain_eva > self.fudge * energy_gain_iteration:
            converged_exception = ConvergedException(
                energy_gains=self.last_energy_gains,
                last_iterate=current_iterate,
                n_iterations_done=self.n_iterations_done)
            raise converged_exception

        energy_after_eva, local_energy_gains_eva = \
            get_energy_after_eva_and_local_energy_gains_eva(
                current_iterate=current_iterate,
                coordinates=self.coordinates,
                elements=self.elements,
                boundaries=self.boundaries,
                edges=self.edges,
                free_edges=self.free_edges,
                cubature_rule=self.cubature_rule,
                f=f,
                verbose=self.verbose)

        energy_gain_eva = current_energy - energy_after_eva

        # keep energy considerations in memory
        self.energy_of_last_iterate = current_energy
        self.last_energy_gains = local_energy_gains_eva
        self.last_energy_gain_eva = energy_gain_eva


class EnergyTailOffCustomCallback(CustomCallBack):
    """
    After each batch of iterations,
    compares the accumulated energy gain with
    the energy gain associated to another batch of iterations.

    Attributes
    ----------
    energy_of_last_iterate: float
        energy of the last iterate considered
    fudge: float
        fudge parameter used when comparing accumulated
        and current energy gain
    accumulated_energy_gain: float
        energy gain accumulated since initiation of
        global CG iterations
    verbose: bool
    """
    energy_of_last_iterate: float
    fudge: float
    accumulated_energy_gain: float
    verbose: bool
    energy_history: list[float]
    n_callback_called: int

    def __init__(
            self,
            batch_size: int,
            min_n_iterations_per_mesh: int,
            elements: ElementsType,
            coordinates: CoordinatesType,
            boundaries: list[BoundaryType],
            energy_of_initial_guess: float,
            fudge: float,
            cubature_rule: CubatureRuleEnum = CubatureRuleEnum.MIDPOINT):
        super().__init__(
            batch_size=batch_size,
            min_n_iterations_per_mesh=min_n_iterations_per_mesh,
            elements=elements,
            coordinates=coordinates,
            boundaries=boundaries,
            cubature_rule=cubature_rule)
        self.fudge = fudge
        self.accumulated_energy_gain = 0.
        self.energy_of_last_iterate = energy_of_initial_guess
        self.energy_history = []
        self.n_callback_called = 0

    def calculate_energy(self, current_iterate) -> float:
        return (
            0.5*current_iterate.dot(self.lhs_matrix.dot(current_iterate))
            - self.rhs_vector.dot(current_iterate))

    def perform_callback(
            self,
            current_iterate) -> None:
        self.n_callback_called += 1

        current_energy = self.calculate_energy(
            current_iterate=current_iterate)
        self.energy_history.append(current_energy)

        # reset the accumulated energy drop to zero
        # after min_n_iterations is reached
        if self.n_callback_called == 1:
            self.energy_of_last_iterate = current_energy

        energy_gain_iteration = self.energy_of_last_iterate - current_energy
        self.accumulated_energy_gain += energy_gain_iteration

        if energy_gain_iteration < self.fudge * self.accumulated_energy_gain:
            energy_gains = get_energy_gains(
                coordinates=self.coordinates,
                elements=self.elements,
                non_boundary_edges=self.non_boundary_edges,
                current_iterate=current_iterate,
                f=f,
                cubature_rule=self.cubature_rule,
                verbose=True)

            converged_exception = ConvergedException(
                energy_gains=energy_gains,
                last_iterate=current_iterate,
                n_iterations_done=self.n_iterations_done,
                energy_history=self.energy_history)
            raise converged_exception

        # keep energy considerations in memory
        self.energy_of_last_iterate = current_energy


class EnergyTailOffAveragedCustomCallback(CustomCallBack):
    """
    After each batch of iterations,
    compares the accumulated energy gain with
    the energy gain associated to another batch of iterations.

    Attributes
    ----------
    energy_of_last_iterate: float
        energy of the last iterate considered
    fudge: float
        fudge parameter used when comparing accumulated
        and current energy gain
    accumulated_energy_gain: float
        energy gain accumulated since initiation of
        global CG iterations
    verbose: bool
    """
    energy_of_last_iterate: float
    fudge: float
    accumulated_energy_gain: float
    verbose: bool
    energy_history: list[float]
    n_callback_called: int

    def __init__(
            self,
            batch_size: int,
            min_n_iterations_per_mesh: int,
            elements: ElementsType,
            coordinates: CoordinatesType,
            boundaries: list[BoundaryType],
            energy_of_initial_guess: float,
            fudge: float,
            cubature_rule: CubatureRuleEnum = CubatureRuleEnum.MIDPOINT):
        super().__init__(
            batch_size=batch_size,
            min_n_iterations_per_mesh=min_n_iterations_per_mesh,
            elements=elements,
            coordinates=coordinates,
            boundaries=boundaries,
            cubature_rule=cubature_rule)
        self.fudge = fudge
        self.accumulated_energy_gain = 0.
        self.energy_of_last_iterate = energy_of_initial_guess
        self.energy_history = []
        self.n_callback_called = 0

    def calculate_energy(self, current_iterate) -> float:
        return (
            0.5*current_iterate.dot(self.lhs_matrix.dot(current_iterate))
            - self.rhs_vector.dot(current_iterate))

    def perform_callback(
            self,
            current_iterate) -> None:
        self.n_callback_called += 1

        current_energy = self.calculate_energy(
            current_iterate=current_iterate)
        self.energy_history.append(current_energy)

        # reset the accumulated energy drop to zero
        # after min_n_iterationns is reached
        if self.n_callback_called == 1:
            self.energy_of_last_iterate = current_energy

        energy_gain_iteration = self.energy_of_last_iterate - current_energy
        self.accumulated_energy_gain += energy_gain_iteration

        if energy_gain_iteration < self.fudge * self.accumulated_energy_gain/self.n_callback_called:
            energy_gains = get_energy_gains(
                coordinates=self.coordinates,
                elements=self.elements,
                non_boundary_edges=self.non_boundary_edges,
                current_iterate=current_iterate,
                f=f,
                cubature_rule=self.cubature_rule,
                verbose=True)

            converged_exception = ConvergedException(
                energy_gains=energy_gains,
                last_iterate=current_iterate,
                n_iterations_done=self.n_iterations_done,
                energy_history=self.energy_history)
            raise converged_exception

        # keep energy considerations in memory
        self.energy_of_last_iterate = current_energy


class EnergyDifferenceProptoDOFCustomCallback(CustomCallBack):
    """
    After each batch of iterations checks if
    $E(u_n) - E(u_{n+1}) < alpha/nDOF$,
    where alpha is a fudge parameter,
    nDOF is the number of degrees of freedom
    and E denotes the energy.

    Attributes
    ----------
    energy_of_last_iterate: float
        energy of the last iterate considered
    fudge: float
        fudge parameter used when comparing current energy gain
        and alpha/n_DOF
    verbose: bool
    energy_history: list[float]
        energy values after each batch iteration
    n_callback_called: int
        number of times the calback was called
        (gets called after miniter is reached and
        after eatch batch cycle of CG has completed)
    n_dofs: int
        number of degrees of freedom on the current mesh
    """
    energy_of_last_iterate: float
    fudge: float
    verbose: bool
    energy_history: list[float]
    n_callback_called: int
    n_dofs: int

    def __init__(
            self,
            batch_size: int,
            min_n_iterations_per_mesh: int,
            elements: ElementsType,
            coordinates: CoordinatesType,
            boundaries: list[BoundaryType],
            fudge: float,
            cubature_rule: CubatureRuleEnum = CubatureRuleEnum.MIDPOINT):
        super().__init__(
            batch_size=batch_size,
            min_n_iterations_per_mesh=min_n_iterations_per_mesh,
            elements=elements,
            coordinates=coordinates,
            boundaries=boundaries,
            cubature_rule=cubature_rule)
        self.fudge = fudge
        self.energy_of_last_iterate = None
        self.energy_history = []
        self.n_callback_called = 0

        # calculate number of degrees of freedom
        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        self.n_dofs = np.sum(free_nodes)

    def calculate_energy(self, current_iterate) -> float:
        return (
            0.5*current_iterate.dot(self.lhs_matrix.dot(current_iterate))
            - self.rhs_vector.dot(current_iterate))

    def perform_callback(
            self,
            current_iterate) -> None:
        self.n_callback_called += 1

        current_energy = self.calculate_energy(
            current_iterate=current_iterate)
        self.energy_history.append(current_energy)

        # in the first call, set last energy and continue
        if self.n_callback_called == 1:
            self.energy_of_last_iterate = current_energy
            return

        energy_gain_iteration = self.energy_of_last_iterate - current_energy

        if energy_gain_iteration < self.fudge / self.n_dofs:
            energy_gains = get_energy_gains(
                coordinates=self.coordinates,
                elements=self.elements,
                non_boundary_edges=self.non_boundary_edges,
                current_iterate=current_iterate,
                f=f,
                cubature_rule=self.cubature_rule,
                verbose=True)

            converged_exception = ConvergedException(
                energy_gains=energy_gains,
                last_iterate=current_iterate,
                n_iterations_done=self.n_iterations_done,
                energy_history=self.energy_history)
            raise converged_exception

        # keep energy considerations in memory
        self.energy_of_last_iterate = current_energy


class ForcingIterationErrorToDiscretizationErrorCustomCallback(CustomCallBack):

    def __init__(
            self,
            batch_size: int,
            min_n_iterations_per_mesh: int,
            elements: ElementsType,
            coordinates: CoordinatesType,
            boundaries: list[BoundaryType],
            cubature_rule: CubatureRuleEnum):
        super().__init__(
            batch_size=batch_size,
            min_n_iterations_per_mesh=min_n_iterations_per_mesh,
            elements=elements,
            coordinates=coordinates,
            boundaries=boundaries,
            cubature_rule=cubature_rule)

    def perform_callback(self, current_iterate: np.ndarray):
        # TODO implement
        pass
