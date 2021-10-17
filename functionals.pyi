from typing import Callable
import numpy as np
from enum import Enum


class ENERGY(Enum):
    HARDCORE_ENERGY         :int = 1
    BOX_ENERGY              :int = 2
    HARDCORE_AND_BOX_ENERGY :int = 3


class stateFunctions:
    pressure: Callable[[np.ndarray], float]
    hardcore_energy: Callable[[np.ndarray], float]
    box_energy: Callable[[np.ndarray], float]
    box_and_hardcore_energy: Callable[[np.ndarray], float]

    def __init__(self, L: float, box_k: float, **kwargs):
        """
        Creates state-functions object with parameter L and box_k
        :param L: float
        :param box_k: float
        """
        ...

    def compile(self):
        """
        Recompile the energy- and pressure- methods of stateFunctions
        :return: None
        """
        ...

    def set_energy_type(self, energy_type: ENERGY):
        ...

    def get_energy(self) -> Callable[[np.ndarray], float]:
        ...

    def get_pressure(self) -> Callable[[np.ndarray], float]:
        ...
