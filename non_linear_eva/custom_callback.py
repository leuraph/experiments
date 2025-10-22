from abc import abstractmethod
import numpy as np
from collections.abc import Callable


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
    energy_history: list[float]
    compute_energy: Callable[[np.ndarray], float]

    def __init__(
            self,
            batch_size: int,
            min_n_iterations_per_mesh: int,
            compute_energy: Callable[[np.ndarray], float]) -> None:
        self.n_iterations_done = 0
        self.batch_size = batch_size
        self.min_n_iterations_per_mesh = min_n_iterations_per_mesh
        self.compute_energy = compute_energy
        self.energy_history = []

    def perform_callback(self, current_iterate) -> None:
        """must be implemented by child classes"""

    def __call__(self, current_iterate) -> None:
        # callback is called after each iteration
        self.energy_history.append(self.compute_energy(current_iterate))
        self.n_iterations_done += 1

        batch_size_reached = self.n_iterations_done % self.batch_size == 0
        min_iterations_reached = (
            self.n_iterations_done >= self.min_n_iterations_per_mesh)
        if batch_size_reached and min_iterations_reached:
            self.perform_callback(current_iterate=current_iterate)


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
    n_callback_called: int

    def __init__(
            self,
            batch_size: int,
            min_n_iterations_per_mesh: int,
            fudge: float,
            compute_energy: Callable[[np.ndarray], float]):
        super().__init__(
            batch_size=batch_size,
            min_n_iterations_per_mesh=min_n_iterations_per_mesh,
            compute_energy=compute_energy)
        self.fudge = fudge
        self.accumulated_energy_gain = 0.
        self.energy_of_last_iterate = None
        self.n_callback_called = 0

    def perform_callback(
            self,
            current_iterate) -> None:
        self.n_callback_called += 1

        current_energy = self.compute_energy(
            current_iterate=current_iterate)

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
    candidates: list[np.ndarray]
    fudge: float

    def __init__(
            self,
            batch_size: int,
            min_n_iterations_per_mesh: int,
            initial_delay: int,
            delay_increase: int,
            tau: float,
            fudge: float,
            n_dofs: int,
            compute_energy: Callable[[np.ndarray], float]):
        super().__init__(
            batch_size=batch_size,
            min_n_iterations_per_mesh=min_n_iterations_per_mesh,
            compute_energy=compute_energy)
        self.delay = initial_delay
        self.delay_increase = delay_increase
        self.tau = tau
        self.fudge = fudge
        self.n_dofs = n_dofs
        self.candidates = []

    def perform_callback(
            self,
            current_iterate) -> None:

        # calculate and save energy of current iterate
        current_energy = self.compute_energy(
            current_iterate=current_iterate)
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
        e_1 = self.compute_energy(self.candidates[0])
        e_2 = self.compute_energy(self.candidates[1])
        e_1_d = self.compute_energy(self.candidates[self.delay])
        e_2_d = self.compute_energy(self.candidates[self.delay + 1])

        hs_1 = 2. * (e_1 - e_1_d)
        hs_2 = 2. * (e_2 - e_2_d)

        return hs_1, hs_2

    def has_converged(self) -> float:
        """
        checks for the CG stopping criterion
        as given in [1] but formulated in an
        energy fashion
        """
        e_1 = self.compute_energy(self.candidates[0])
        e_1_d = self.compute_energy(self.candidates[self.delay])

        lhs = ((self.fudge+self.n_dofs)/self.n_dofs) * e_1
        rhs = e_1_d
        return lhs <= rhs