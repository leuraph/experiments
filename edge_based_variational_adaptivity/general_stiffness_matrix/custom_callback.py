import numpy as np
from p1afempy.mesh import provide_geometric_data
from p1afempy.solvers import get_general_stiffness_matrix, get_right_hand_side
from scipy.sparse import csr_matrix
from ismember import is_row_in
from p1afempy.data_structures import ElementsType, CoordinatesType, \
    BoundaryType
from triangle_cubature.cubature_rule import CubatureRuleEnum


class ConvergedException(Exception):
    energy_gains: np.ndarray
    last_iterate: np.ndarray
    n_iterations_done: int
    energy_history: list[float]
    delay: int

    def __init__(
            self,
            energy_gains: np.ndarray = None,
            last_iterate: np.ndarray = None,
            n_iterations_done: int = None,
            energy_history: list[float] = None,
            delay: int = None):
        self.energy_gains = energy_gains
        self.last_iterate = last_iterate
        self.n_iterations_done = n_iterations_done
        self.energy_history = energy_history
        self.delay = delay


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
            lhs_matrix: csr_matrix,
            rhs_vector: np.ndarray,
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
        self.lhs_matrix = lhs_matrix
        self.rhs_vector = rhs_vector

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

    def get_energy(self, current_iterate: np.ndarray) -> float:
        return (
            0.5 * current_iterate.dot(self.lhs_matrix.dot(current_iterate))
            - self.rhs_vector.dot(current_iterate))

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
            boolean mask indicating free nods
        """
        global_iterate = np.zeros(free_nodes.shape[0])
        global_iterate[free_nodes] = current_iterate_on_free_nodes
        return global_iterate


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

    def perform_callback(
            self,
            current_iterate) -> None:
        self.n_callback_called += 1

        current_energy = self.get_energy(
            current_iterate=current_iterate)
        self.energy_history.append(current_energy)

        # reset the accumulated energy drop to zero
        # after min_n_iterations is reached
        if self.n_callback_called == 1:
            self.energy_of_last_iterate = current_energy

        energy_gain_iteration = self.energy_of_last_iterate - current_energy
        self.accumulated_energy_gain += energy_gain_iteration

        if energy_gain_iteration < self.fudge * self.accumulated_energy_gain:

            converged_exception = ConvergedException(
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

    def perform_callback(
            self,
            current_iterate) -> None:
        self.n_callback_called += 1

        current_energy = self.get_energy(
            current_iterate=current_iterate)
        self.energy_history.append(current_energy)

        # reset the accumulated energy drop to zero
        # after min_n_iterationns is reached
        if self.n_callback_called == 1:
            self.energy_of_last_iterate = current_energy

        energy_gain_iteration = self.energy_of_last_iterate - current_energy
        self.accumulated_energy_gain += energy_gain_iteration

        if energy_gain_iteration < self.fudge * self.accumulated_energy_gain/self.n_callback_called:

            converged_exception = ConvergedException(
                last_iterate=current_iterate,
                n_iterations_done=self.n_iterations_done,
                energy_history=self.energy_history)
            raise converged_exception

        # keep energy considerations in memory
        self.energy_of_last_iterate = current_energy


class AriolisCustomCallback(CustomCallBack):
    """
    Implements the stopping criterion from [1]
    in an an energy fashion.

    Attributes
    ----------
    delay: int
        delay parameter in the Hestenes-Stiefel Estimator,
        see [1] for details

    References
    ----------
    [1] Arioli, M.
    A Stopping Criterion for the Conjugate Gradient Algorithm
    in a Finite Element Method Framework.
    Numerische Mathematik 97, no. 1 (1 March 2004): 1-24.
    https://doi.org/10.1007/s00211-003-0500-y.

    """
    n_dofs: int
    delay: int
    n_callback_called: int
    energy_history: list[float]
    iterate_history: list[np.ndarray]
    fudge: float

    def __init__(
            self,
            batch_size: int,
            min_n_iterations_per_mesh: int,
            elements: ElementsType,
            coordinates: CoordinatesType,
            boundaries: list[BoundaryType],
            delay: int,
            fudge: float,
            cubature_rule: CubatureRuleEnum = CubatureRuleEnum.MIDPOINT):
        super().__init__(
            batch_size=batch_size,
            min_n_iterations_per_mesh=min_n_iterations_per_mesh,
            elements=elements,
            coordinates=coordinates,
            boundaries=boundaries,
            cubature_rule=cubature_rule)
        self.delay = delay
        self.fudge = fudge
        self.n_callback_called = 0
        self.energy_history = []
        self.iterate_history = []

        # calculate number of degrees of freedom
        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        self.n_dofs = np.sum(free_nodes)

    def perform_callback(
            self,
            current_iterate) -> None:
        self.n_callback_called += 1

        current_energy = self.get_energy(
            current_iterate=current_iterate)
        self.energy_history.append(current_energy)
        self.iterate_history.append(current_iterate)

        delay_reached = self.n_callback_called >= self.delay + 1
        if not delay_reached:
            return

        energy_before_delay = self.energy_history[-(self.delay+1)]
        iterate_before_delay = self.iterate_history[-(self.delay+1)]

        lhs = ((self.fudge+self.n_dofs)/self.n_dofs) * energy_before_delay
        rhs = current_energy
        converged = lhs <= rhs

        if converged:
            converged_exception = ConvergedException(
                last_iterate=iterate_before_delay,
                n_iterations_done=self.n_iterations_done,
                energy_history=self.energy_history)
            raise converged_exception


class AriolisAdaptiveDelayCustomCallback(CustomCallBack):
    """
    Implements the stopping criterion from [1]
    in an an energy fashion and with an adaptive
    choice of the delay (also mentioned in [1]).

    Attributes
    ----------
    delay: int
        initial delay parameter in the Hestenes-Stiefel Estimator,
        see [1] for details

    References
    ----------
    [1] Arioli, M.
    A Stopping Criterion for the Conjugate Gradient Algorithm
    in a Finite Element Method Framework.
    Numerische Mathematik 97, no. 1 (1 March 2004): 1-24.
    https://doi.org/10.1007/s00211-003-0500-y.

    """
    n_dofs: int
    delay: int
    delay_increase: int
    tau: float
    energy_history: list[float]
    candidates: list[np.ndarray]
    fudge: float

    def __init__(
            self,
            batch_size: int,
            min_n_iterations_per_mesh: int,
            elements: ElementsType,
            coordinates: CoordinatesType,
            boundaries: list[BoundaryType],
            initial_delay: int,
            delay_increase: int,
            tau: float,
            fudge: float,
            cubature_rule: CubatureRuleEnum = CubatureRuleEnum.MIDPOINT):
        super().__init__(
            batch_size=batch_size,
            min_n_iterations_per_mesh=min_n_iterations_per_mesh,
            elements=elements,
            coordinates=coordinates,
            boundaries=boundaries,
            cubature_rule=cubature_rule)
        self.delay = initial_delay
        self.delay_increase = delay_increase
        self.tau = tau
        self.fudge = fudge
        self.energy_history = []
        self.candidates = []

        # calculate number of degrees of freedom
        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        self.n_dofs = np.sum(free_nodes)

    def perform_callback(
            self,
            current_iterate) -> None:

        # calculate and save energy of current iterate
        current_energy = self.get_energy(
            current_iterate=current_iterate)
        self.energy_history.append(current_energy)
        self.candidates.append(current_iterate)

        while True:

            # keep on iterating until we can calculate both
            # relevant HS-estimates
            if not self.can_calculate_hs_estimates():
                break

            # check if the current HS-estimate is a "good" approximation
            # with the rule given in [1]
            # if not, increase delay and continue
            if self.need_to_increase_delay():
                self.delay += self.delay_increase
                continue

            if self.has_converged():
                converged_exception = ConvergedException(
                    last_iterate=self.candidates[0],
                    n_iterations_done=self.n_iterations_done,
                    delay=self.delay,
                    energy_history=self.energy_history)
                raise converged_exception

            # throw away the oldest iterate and continue
            self.candidates = self.candidates[1:]

    def need_to_increase_delay(self) -> bool:
        """implements the criterion for increasing the delay from [1]"""
        hs_1, hs_2 = self.get_hs_estimates()
        return hs_2 > hs_1 * self.tau

    def can_calculate_hs_estimates(self) -> bool:
        """
        returns True if we have enough candidates in memory
        in order to calculate both HS estimates
        """
        return self.delay + 2 <= len(self.candidates)

    def get_hs_estimates(self) -> tuple[float, float]:
        """
        returns both HS estimates needed in the adaptive delay scheme
        """
        e_1 = self.get_energy(self.candidates[0])
        e_2 = self.get_energy(self.candidates[1])
        e_1_d = self.get_energy(self.candidates[self.delay])
        e_2_d = self.get_energy(self.candidates[self.delay + 1])

        hs_1 = 2. * (e_1 - e_1_d)
        hs_2 = 2. * (e_2 - e_2_d)

        return hs_1, hs_2

    def has_converged(self) -> float:
        """
        checks for the CG stopping criterion
        as given in [1] but formulated in an
        energy fashion
        """
        e_1 = self.get_energy(self.candidates[0])
        e_1_d = self.get_energy(self.candidates[self.delay])

        lhs = ((self.fudge+self.n_dofs)/self.n_dofs) * e_1
        rhs = e_1_d
        return lhs <= rhs
