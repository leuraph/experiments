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


class MonitorException(Exception):
    energy_history: np.ndarray
    energy_norm_squared_history: np.ndarray
    iterate_history: np.ndarray

    def __init__(
            self,
            energy_history: np.ndarray,
            energy_norm_squared_history: np.ndarray,
            iterate_history: np.ndarray):
        self.energy_history = energy_history
        self.energy_norm_squared_history = energy_norm_squared_history
        self.iterate_history = iterate_history


class ConvergedException(Exception):
    energy_gains: np.ndarray
    last_iterate: np.ndarray
    n_iterations_done: int
    energy_history: list[float]

    def __init__(
            self,
            energy_gains: np.ndarray = None,
            last_iterate: np.ndarray = None,
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
    n_callback_called: int

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
        self.n_callback_called = 0

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
        self.n_callback_called += 1

        if self.n_callback_called == 1:
            # calculate current energy
            current_energy = self.calculate_energy(
                current_iterate=current_iterate)
            # set last energy to current
            self.energy_of_last_iterate = current_energy
            # calculate energy gain eva and its gains
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
            self.last_energy_gain_eva = current_energy - energy_after_eva
            self.last_energy_gains = local_energy_gains_eva
            return

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

    energy_norm_error_squared_galerkin_to_exact: float
    fudge: float
    galerkin_solution: np.ndarray
    energy_history: list[float]

    def __init__(
            self,
            batch_size: int,
            min_n_iterations_per_mesh: int,
            elements: ElementsType,
            coordinates: CoordinatesType,
            boundaries: list[BoundaryType],
            cubature_rule: CubatureRuleEnum,
            energy_norm_error_squared_galerkin_to_exact: float,
            galerkin_solution: np.ndarray,
            fudge: float,
            parallel_eva: bool = False):
        super().__init__(
            batch_size=batch_size,
            min_n_iterations_per_mesh=min_n_iterations_per_mesh,
            elements=elements,
            coordinates=coordinates,
            boundaries=boundaries,
            cubature_rule=cubature_rule)
        self.energy_norm_error_squared_galerkin_to_exact = \
            energy_norm_error_squared_galerkin_to_exact
        self.galerkin_solution = galerkin_solution
        self.fudge = fudge
        self.parallel_eva = parallel_eva
        self.energy_history = []

    def calculate_energy(self, current_iterate) -> float:
        return (
            0.5*current_iterate.dot(self.lhs_matrix.dot(current_iterate))
            - self.rhs_vector.dot(current_iterate))

    def perform_callback(self, current_iterate: np.ndarray):

        self.energy_history.append(
            self.calculate_energy(current_iterate=current_iterate))

        energy_norm_error_squared_iterate_to_galerkin = \
            self.get_energy_norm_error_squared_iterate_to_galerkin(
                current_iterate=current_iterate)
        if energy_norm_error_squared_iterate_to_galerkin < \
                self.fudge * self.energy_norm_error_squared_galerkin_to_exact:

            energy_gains = get_energy_gains(
                coordinates=self.coordinates,
                elements=self.elements,
                non_boundary_edges=self.non_boundary_edges,
                current_iterate=current_iterate,
                f=f,
                cubature_rule=self.cubature_rule,
                verbose=True,
                parallel=self.parallel_eva)

            converged_exception = ConvergedException(
                energy_gains=energy_gains,
                last_iterate=current_iterate,
                n_iterations_done=self.n_iterations_done,
                energy_history=self.energy_history)
            raise converged_exception

    def get_energy_norm_error_squared_iterate_to_galerkin(
            self,
            current_iterate: np.ndarray) -> float:
        du = current_iterate - self.galerkin_solution
        energy_err_squared = du.dot(self.lhs_matrix.dot(du))
        return energy_err_squared


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


class ArioliSanityCheckCustomCallback(CustomCallBack):
    """
    Implements the stopping criterion from [1]
    in an an exact fashion, without approximations.
    This custom callback is used to perform sanity
    checks on the stopping criterion found in [1].

    Attributes
    ----------
    galerkin_solution: np.ndarray
        galerkin solution on the current mesh

    References
    ----------
    [1] Arioli, M.
    A Stopping Criterion for the Conjugate Gradient Algorithm
    in a Finite Element Method Framework.
    Numerische Mathematik 97, no. 1 (1 March 2004): 1-24.
    https://doi.org/10.1007/s00211-003-0500-y.

    """
    n_dofs: int
    galerkin_solution: np.ndarray
    n_callback_called: int
    fudge: float

    def __init__(
            self,
            batch_size: int,
            min_n_iterations_per_mesh: int,
            elements: ElementsType,
            coordinates: CoordinatesType,
            boundaries: list[BoundaryType],
            galerkin_solution: np.ndarray,
            fudge: float,
            cubature_rule: CubatureRuleEnum = CubatureRuleEnum.MIDPOINT):
        super().__init__(
            batch_size=batch_size,
            min_n_iterations_per_mesh=min_n_iterations_per_mesh,
            elements=elements,
            coordinates=coordinates,
            boundaries=boundaries,
            cubature_rule=cubature_rule)
        self.galerkin_solution = galerkin_solution
        self.fudge = fudge
        self.n_callback_called = 0

        # calculate number of degrees of freedom
        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        self.n_dofs = np.sum(free_nodes)

    def calculate_energy_norm_squared(self, u) -> float:
        return u.dot(self.lhs_matrix.dot(u))

    def perform_callback(
            self,
            current_iterate: np.ndarray) -> None:
        self.n_callback_called += 1

        du = current_iterate - self.galerkin_solution
        lhs = self.calculate_energy_norm_squared(du)
        rhs = (self.fudge/self.n_dofs) * self.calculate_energy_norm_squared(
            self.galerkin_solution)
        converged = lhs <= rhs

        if converged:
            converged_exception = ConvergedException(
                last_iterate=current_iterate,
                n_iterations_done=self.n_iterations_done)
            raise converged_exception


class ArioliHeuristicCustomCallback(CustomCallBack):
    """
    Implements a heuristic approximation of
    the stopping criterion from [1].

    Attributes
    ----------
    galerkin_solution: np.ndarray
        galerkin solution on the current mesh

    References
    ----------
    [1] Arioli, M.
    A Stopping Criterion for the Conjugate Gradient Algorithm
    in a Finite Element Method Framework.
    Numerische Mathematik 97, no. 1 (1 March 2004): 1-24.
    https://doi.org/10.1007/s00211-003-0500-y.

    """
    n_dofs: int
    galerkin_solution: np.ndarray
    n_callback_called: int
    fudge: float

    def __init__(
            self,
            batch_size: int,
            min_n_iterations_per_mesh: int,
            elements: ElementsType,
            coordinates: CoordinatesType,
            boundaries: list[BoundaryType],
            galerkin_solution: np.ndarray,
            fudge: float,
            cubature_rule: CubatureRuleEnum = CubatureRuleEnum.MIDPOINT):
        super().__init__(
            batch_size=batch_size,
            min_n_iterations_per_mesh=min_n_iterations_per_mesh,
            elements=elements,
            coordinates=coordinates,
            boundaries=boundaries,
            cubature_rule=cubature_rule)
        self.galerkin_solution = galerkin_solution
        self.fudge = fudge
        self.n_callback_called = 0

        # calculate number of degrees of freedom
        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        self.n_dofs = np.sum(free_nodes)

    def calculate_energy_norm_squared(self, u) -> float:
        return u.dot(self.lhs_matrix.dot(u))

    def calculate_energy(self, current_iterate) -> float:
        return (
            0.5*current_iterate.dot(self.lhs_matrix.dot(current_iterate))
            - self.rhs_vector.dot(current_iterate))

    def perform_callback(
            self,
            current_iterate: np.ndarray) -> None:
        self.n_callback_called += 1

        current_energy = self.calculate_energy(
            current_iterate=current_iterate)
        current_energy_norm_squared = self.calculate_energy_norm_squared(
            u=current_iterate)
        gamma_squared = 1./self.n_dofs
        lhs = (1.-gamma_squared) * current_energy_norm_squared
        rhs = -2.*current_energy
        converged = lhs <= self.fudge * rhs

        if converged:
            converged_exception = ConvergedException(
                last_iterate=current_iterate,
                n_iterations_done=self.n_iterations_done)
            raise converged_exception


class ArioliEllipsoidMaxCustomCallback(CustomCallBack):
    delay: int  # how many times we bounce inside the ellipsoid
    n_dofs: int  # number of degrees of freedom on mesh at hand

    def __init__(
            self,
            batch_size,
            min_n_iterations_per_mesh,
            elements,
            coordinates,
            boundaries,
            cubature_rule,
            delay: int = None):
        super().__init__(
            batch_size,
            min_n_iterations_per_mesh,
            elements,
            coordinates,
            boundaries,
            cubature_rule)

        # calculate number of degrees of freedom
        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        self.n_dofs = np.sum(free_nodes)

        if delay is None:
            self.delay = self.n_dofs
            return
        self.delay = delay

    def calculate_energy(self, current_iterate) -> float:
        return (
            0.5*current_iterate.dot(self.lhs_matrix.dot(current_iterate))
            - self.rhs_vector.dot(current_iterate))

    @staticmethod
    def apply_householder(v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """returns H(v)w"""
        return w - 2 * (v.dot(w))/(v.dot(v)) * v

    def get_potential_upper_bounds(
            self, current_iterate: np.ndarray) -> np.ndarray:
        """
        bounces inside the current level set
        and produes potential upper bounds at the same time

        parameters
        ----------
        current_iterate: np.ndarray
            bouncing inside ellipsoid is
            initiated with current iterate

        returns
        -------
        potential_upper_bounds: np.ndarray
            array containing the values
            |w_n|^2_a, n=0, ..., delay
        """

        stiffness_matrix_on_free_nodes = self.lhs_matrix[
            self.free_nodes, :][:, self.free_nodes]
        load_vector_on_free_nodes = self.rhs_vector[self.free_nodes]

        def get_residual(current_iterate_ellipsoid: np.ndarray) -> np.ndarray:
            return (
                load_vector_on_free_nodes
                - stiffness_matrix_on_free_nodes.dot(
                    current_iterate_ellipsoid))

        def get_energy_norm_squared(
                current_iterate_ellipsoid: np.ndarray) -> float:
            return current_iterate_ellipsoid.dot(
                stiffness_matrix_on_free_nodes.dot(current_iterate_ellipsoid))

        def get_step_size(current_residual, current_direction) -> np.ndarray:
            """returns the step size in order to stay on the level set"""
            numerator = 2. * current_residual.dot(current_direction)
            denominator = get_energy_norm_squared(current_direction)
            return numerator / denominator

        def calculate_energy_free_nodes(
                current_iterate_ellipsoid: np.ndarray) -> float:
            return (
                0.5 * current_iterate_ellipsoid.dot(
                    stiffness_matrix_on_free_nodes.dot(
                        current_iterate_ellipsoid))
                - load_vector_on_free_nodes.dot(current_iterate_ellipsoid))

        current_iterate_on_free_nodes = np.copy(
            current_iterate[self.free_nodes])

        current_iterate_ellipsoid = current_iterate_on_free_nodes
        current_residual = get_residual(
            current_iterate_ellipsoid=current_iterate_ellipsoid)
        current_direction = current_residual

        potential_upper_bounds = []
        energy_history = [calculate_energy_free_nodes(
            current_iterate_ellipsoid)]
        for _ in range(self.delay):
            current_iterate_ellipsoid = (
                current_iterate_ellipsoid
                + get_step_size(
                    current_residual=current_residual,
                    current_direction=current_direction)
                * current_direction)
            current_residual = get_residual(
                current_iterate_ellipsoid=current_iterate_ellipsoid)
            current_direction = self.apply_householder(
                current_residual, current_direction)

            energy_history.append(
                calculate_energy_free_nodes(current_iterate_ellipsoid))
            potential_upper_bounds.append(
                get_energy_norm_squared(
                    current_iterate_ellipsoid=current_iterate_ellipsoid))

        # sanity check: are all points generated on the level set??
        if not np.allclose(np.array(energy_history), energy_history[0]):
            raise RuntimeError('not all points lie on the same level set')

        return np.array(potential_upper_bounds)

    def perform_callback(
            self,
            current_iterate: np.ndarray) -> None:

        gamma_squared = 1./self.n_dofs
        current_energy = self.calculate_energy(current_iterate=current_iterate)

        potential_upper_bounds = self.get_potential_upper_bounds(
            current_iterate=current_iterate)

        potential_upper_bound = np.max(potential_upper_bounds)

        lhs = (1-gamma_squared)*potential_upper_bound
        rhs = -2*current_energy

        converged = lhs <= rhs

        if converged:
            converged_exception = ConvergedException(
                last_iterate=current_iterate,
                n_iterations_done=self.n_iterations_done)
            raise converged_exception


class ArioliEllipsoidAvgCustomCallback(CustomCallBack):
    delay: int  # how many times we bounce inside the ellipsoid
    n_dofs: int  # number of degrees of freedom on mesh at hand

    def __init__(
            self,
            batch_size,
            min_n_iterations_per_mesh,
            elements,
            coordinates,
            boundaries,
            cubature_rule,
            delay: int = None):
        super().__init__(
            batch_size,
            min_n_iterations_per_mesh,
            elements,
            coordinates,
            boundaries,
            cubature_rule)

        # calculate number of degrees of freedom
        n_vertices = coordinates.shape[0]
        indices_of_free_nodes = np.setdiff1d(
            ar1=np.arange(n_vertices),
            ar2=np.unique(boundaries[0].flatten()))
        free_nodes = np.zeros(n_vertices, dtype=bool)
        free_nodes[indices_of_free_nodes] = 1
        self.n_dofs = np.sum(free_nodes)

        if delay is None:
            self.delay = self.n_dofs
            return
        self.delay = delay

    def calculate_energy(self, current_iterate) -> float:
        return (
            0.5*current_iterate.dot(self.lhs_matrix.dot(current_iterate))
            - self.rhs_vector.dot(current_iterate))

    @staticmethod
    def apply_householder(v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """returns H(v)w"""
        return w - 2 * (v.dot(w))/(v.dot(v)) * v

    def get_potential_upper_bound(
            self, current_iterate: np.ndarray) -> np.ndarray:
        """
        bounces inside the current level set
        and averages all points on the level set

        parameters
        ----------
        current_iterate: np.ndarray
            bouncing inside ellipsoid is
            initiated with current iterate

        returns
        -------
        potential_upper_bound: float
            energy norm squared of the averaged point cloud
            |1/d * sum w_n|^2_a, n=1, ..., delay
        """

        stiffness_matrix_on_free_nodes = self.lhs_matrix[
            self.free_nodes, :][:, self.free_nodes]
        load_vector_on_free_nodes = self.rhs_vector[self.free_nodes]

        def get_residual(current_iterate_ellipsoid: np.ndarray) -> np.ndarray:
            return (
                load_vector_on_free_nodes
                - stiffness_matrix_on_free_nodes.dot(
                    current_iterate_ellipsoid))

        def get_energy_norm_squared(
                current_iterate_ellipsoid: np.ndarray) -> float:
            return current_iterate_ellipsoid.dot(
                stiffness_matrix_on_free_nodes.dot(current_iterate_ellipsoid))

        def get_step_size(current_residual, current_direction) -> np.ndarray:
            """returns the step size in order to stay on the level set"""
            numerator = 2. * current_residual.dot(current_direction)
            denominator = get_energy_norm_squared(current_direction)
            return numerator / denominator

        def calculate_energy_free_nodes(
                current_iterate_ellipsoid: np.ndarray) -> float:
            return (
                0.5 * current_iterate_ellipsoid.dot(
                    stiffness_matrix_on_free_nodes.dot(
                        current_iterate_ellipsoid))
                - load_vector_on_free_nodes.dot(current_iterate_ellipsoid))

        current_iterate_on_free_nodes = np.copy(
            current_iterate[self.free_nodes])

        current_iterate_ellipsoid = current_iterate_on_free_nodes
        current_residual = get_residual(
            current_iterate_ellipsoid=current_iterate_ellipsoid)
        current_direction = current_residual

        energy_history = [calculate_energy_free_nodes(
            current_iterate_ellipsoid)]
        average = np.zeros(self.n_dofs)
        for _ in range(self.delay):
            current_iterate_ellipsoid = (
                current_iterate_ellipsoid
                + get_step_size(
                    current_residual=current_residual,
                    current_direction=current_direction)
                * current_direction)
            current_residual = get_residual(
                current_iterate_ellipsoid=current_iterate_ellipsoid)
            current_direction = self.apply_householder(
                current_residual, current_direction)

            energy_history.append(
                calculate_energy_free_nodes(current_iterate_ellipsoid))
            average += (1./self.delay) * current_iterate_ellipsoid

        # sanity check: are all points generated on the level set??
        if not np.allclose(np.array(energy_history), energy_history[0]):
            raise RuntimeError('not all points lie on the same level set')

        return get_energy_norm_squared(average)

    def perform_callback(
            self,
            current_iterate: np.ndarray) -> None:

        gamma_squared = 1./self.n_dofs
        current_energy = self.calculate_energy(current_iterate=current_iterate)

        potential_upper_bound = self.get_potential_upper_bound(
            current_iterate=current_iterate)

        lhs = (1-gamma_squared)*potential_upper_bound
        rhs = -2*current_energy

        converged = lhs <= rhs

        if converged:
            converged_exception = ConvergedException(
                last_iterate=current_iterate,
                n_iterations_done=self.n_iterations_done)
            raise converged_exception


class RecorderCustomCallback(CustomCallBack):
    """
    Keeps track of the energy and energy norm squared.
    Raises an Exception when `max_n_iterations` is reached.
    """
    max_n_iterations: int
    energy_history: list[float]
    energy_norm_squared_history: list[float]
    iterate_history: list[np.ndarray]

    def __init__(
            self,
            batch_size: int,
            min_n_iterations_per_mesh: int,
            elements: ElementsType,
            coordinates: CoordinatesType,
            boundaries: list[BoundaryType],
            max_n_iterations: int,
            cubature_rule: CubatureRuleEnum = CubatureRuleEnum.MIDPOINT,):
        super().__init__(
            batch_size=batch_size,
            min_n_iterations_per_mesh=min_n_iterations_per_mesh,
            elements=elements,
            coordinates=coordinates,
            boundaries=boundaries,
            cubature_rule=cubature_rule)
        self.max_n_iterations = max_n_iterations
        self.energy_history = []
        self.energy_norm_squared_history = []
        self.iterate_history = []

    def calculate_energy_norm_squared(self, u) -> float:
        return u.dot(self.lhs_matrix.dot(u))

    def calculate_energy(self, current_iterate) -> float:
        return (
            0.5*current_iterate.dot(self.lhs_matrix.dot(current_iterate))
            - self.rhs_vector.dot(current_iterate))

    def perform_callback(self, current_iterate: np.ndarray) -> None:

        # calculating the current energy and energy norm squared
        current_energy = self.calculate_energy(
            current_iterate=current_iterate)
        current_energy_norm_squared = self.calculate_energy_norm_squared(
            u=current_iterate)

        # appending to history
        self.energy_history.append(current_energy)
        self.energy_norm_squared_history.append(current_energy_norm_squared)
        self.iterate_history.append(current_iterate)

        if self.n_iterations_done >= self.max_n_iterations:

            # changing lists to arrays
            energy_history = np.array(self.energy_history)
            energy_norm_squared_history = np.array(
                self.energy_norm_squared_history)
            iterate_history = np.array(self.iterate_history)

            # prparing the exception and raising it
            monitor_exception = MonitorException(
                energy_history=energy_history,
                energy_norm_squared_history=energy_norm_squared_history,
                iterate_history=iterate_history)
            raise monitor_exception
