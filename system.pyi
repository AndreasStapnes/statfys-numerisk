import numpy as np
from typing import Callable



class System:
    state: np.ndarray
    particle_amt: int
    dimension: int
    L: float
    jump_scale: float

    pressure: Callable[[np.ndarray], float]
    energy: Callable[[np.ndarray], float]

    def __init__(self, particle_amt: float, L: float, **kwargs): ...

    def jump(self):
        """
        Generate a new random state by tanking a randomized step from the current state
        :return: float-np.ndarray of shape state
        """


    def goto_next(self):
        """
        Manually aquire to next state
        :return: float-np.ndarray of shape state
        """
        ...

    def explore(self, iterations: int):
        """
        Jitted method of performing multiple goto_next-steps
        :param iterations: Iterations to perform
        :return: [list-of-pressure-values,]
        """