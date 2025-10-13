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

    def __init__(
            self,
            batch_size: int,
            min_n_iterations_per_mesh: int) -> None:
        self.n_iterations_done = 0
        self.batch_size = batch_size
        self.min_n_iterations_per_mesh = min_n_iterations_per_mesh

    @abstractmethod
    def perform_callback(self, current_iterate) -> None:
        pass

    def __call__(self, current_iterate) -> None:
        # we know that scipy.sparse.linalg.cg calls this after each iteration
        self.n_iterations_done += 1

        batch_size_reached = self.n_iterations_done % self.batch_size == 0
        min_iterations_reached = (
            self.n_iterations_done >= self.min_n_iterations_per_mesh)
        if batch_size_reached and min_iterations_reached:
            # NOTE this part becomes obsolete because we plan to iterate on the full space
            # 
            # current_iterate = np.zeros(self.coordinates.shape[0], dtype=float)
            # current_iterate[self.free_nodes] = current_iterate_on_free_nodes
            # # check if we must continue with iterations
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
    energy_history: list[float]
    n_callback_called: int

    compute_energy: Callable[[np.ndarray], float]

    def __init__(
            self,
            batch_size: int,
            min_n_iterations_per_mesh: int,
            fudge: float,
            compute_energy: Callable[[np.ndarray], float]):
        super().__init__(
            batch_size=batch_size,
            min_n_iterations_per_mesh=min_n_iterations_per_mesh)
        self.fudge = fudge
        self.accumulated_energy_gain = 0.
        self.energy_of_last_iterate = None
        self.energy_history = []
        self.n_callback_called = 0
        self.compute_energy = compute_energy

    def perform_callback(
            self,
            current_iterate) -> None:
        self.n_callback_called += 1

        current_energy = self.compute_energy(
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
