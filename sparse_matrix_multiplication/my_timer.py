import time


class Timer:
    # Note
    # ----
    # The reference point of the returned value is undefined,
    # so that only the difference between the results of two calls is valid.
    start_time: float
    stop_time: float
    running: bool

    def __init__(self) -> None:
        self.start_time = 0.
        self.stop_time = 0.
        self.running = False

    def start(self) -> None:
        self.start_time = time.process_time_ns()
        self.running = True

    def stop(self) -> None:
        self.stop_time = time.process_time_ns()
        self.running = False

    def reset(self) -> None:
        self.start_time = 0.
        self.stop_time = 0.
        self.running = False

    def read(self) -> float:
        """Returns the process time in units of nanoseconds."""
        if self.running:
            return time.process_time_ns() - self.start_time
        return self.stop_time - self.start_time
