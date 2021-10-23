import numpy as np
from typing import Callable
from functionals import stateFunctions



class System:
    state: np.ndarray
    particle_amt: int
    dimension: int
    L: float
    jump_scale: float

    pressure: Callable[[np.ndarray], float]
    energy: Callable[[np.ndarray], float]

    def __init__(self, particle_amt: int, L: float, state_functions: stateFunctions, **kwargs): ...

    def reset(self, L: float=None, recompile: bool=False):
        ...

    def explore(self, iterations: int, log_interval: int = 1):
        """
        Jitted method of performing multiple goto_next-steps, moving a single, randomly selected
        particle at a time
        :param iterations: int; iterations to perform
        :return:
        """
        ...