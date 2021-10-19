from __future__ import annotations
from typing import List, Tuple, Callable, Dict, NamedTuple
from constants import hardcore_pot, hardcore_diameter, a, epsilon
import numpy as np
from numba import njit
from enum import Enum


class ENERGY(Enum):
    HARDCORE_ENERGY         :int = 1
    BOX_ENERGY              :int = 2
    HARDCORE_AND_BOX_ENERGY :int = 3
    LJ_AND_BOX_ENERGY       :int = 4


class stateFunctions:
    pressure: Callable[[np.ndarray], float]
    hardcore_energy: Callable[[np.ndarray], float]
    box_energy: Callable[[np.ndarray], float]
    box_and_hardcore_energy: Callable[[np.ndarray], float]
    box_and_lj_energy: Callable[[np.ndarray], float]
    single_particle_pressure: Callable[[np.ndarray, int], float]
    single_particle_box_energy: Callable[[np.ndarray, int], float]
    single_particle_hardcore_energy: Callable[[np.ndarray, int], float]
    single_particle_box_and_hardcore_energy: Callable[[np.ndarray, int], float]
    single_particle_box_and_lj_energy: Callable[[np.ndarray, int], float]

    def __init__(self, L: float, box_k: float, **kwargs):
        self.L = L
        self.box_k = box_k
        self.compile()
        self.energy_type = kwargs.get("energy_type", ENERGY.HARDCORE_AND_BOX_ENERGY)

    def set_energy_type(self, energy_type: ENERGY):
        self.energy_type = energy_type

    def get_energy(self) -> Callable[[np.ndarray], float]:
        return {ENERGY.HARDCORE_ENERGY: self.hardcore_energy,
                ENERGY.BOX_ENERGY: self.box_energy,
                ENERGY.HARDCORE_AND_BOX_ENERGY: self.box_and_hardcore_energy,
                ENERGY.LJ_AND_BOX_ENERGY: self.box_and_lj_energy}[self.energy_type]

    def get_pressure(self) -> Callable[[np.ndarray], float]:
        return self.pressure

    def get_single_particle_energy(self) -> Callable[[np.ndarray, int], float]:
        return {ENERGY.HARDCORE_ENERGY: self.single_particle_hardcore_energy,
                ENERGY.BOX_ENERGY: self.single_particle_box_energy,
                ENERGY.HARDCORE_AND_BOX_ENERGY: self.single_particle_box_and_hardcore_energy,
                ENERGY.LJ_AND_BOX_ENERGY: self.single_particle_box_and_lj_energy}[self.energy_type]

    def get_single_particle_pressure(self) -> Callable[[np.ndarray, int], float]:
        return self.single_particle_pressure

    def compile(self):
        L = self.L
        box_k = self.box_k

        @njit()
        def pressure(state: np.ndarray) -> float:
            pressure_contrib = np.sum(np.where(state < 0, -state, 0)) + \
                               np.sum(np.where(state > L, state - L, 0))
            pressure_contrib *= box_k / L / 4
            return pressure_contrib

        @njit()
        def single_particle_pressure(state:np.ndarray, i: int):
            return pressure(state[i])

        @njit()
        def hardcore_energy(state: np.ndarray) -> float:
            particle_amt = len(state)
            energy_contrib = 0.0
            for i in range(particle_amt):
                for j in range(i+1, particle_amt):
                    energy_contrib += hardcore_pot if np.linalg.norm(state[i] - state[j]) < hardcore_diameter else 0
            return energy_contrib

        @njit()
        def LJ_energy(state: np.ndarray) -> float:
            particle_amt = len(state)
            energy_contrib = 0.0
            for i in range(particle_amt):
                for j in range(i+1, particle_amt):
                    r = np.linalg.norm(state[i]-state[j])
                    energy_contrib += epsilon*((a/r)**12 - 2*(a/r)**6)
            return energy_contrib

        @njit()
        def single_particle_LJ_energy(state: np.ndarray, i: int) -> float:
            energy_contrib = 0.0
            particle_i = state[i]
            for j in range(len(state)):
                if j != i:
                    r = np.linalg.norm(particle_i - state[j])
                    energy_contrib += epsilon*((a/r)**12 - 2*(a/r)**6)
            return energy_contrib

        @njit()
        def single_particle_hardcore_energy(state: np.ndarray, i: int) -> float:
            energy_contrib = 0.0
            particle_i = state[i]
            for j in range(len(state)):
                if j != i:
                    energy_contrib += hardcore_pot if np.linalg.norm(particle_i - state[j]) < hardcore_diameter else 0
            return energy_contrib


        @njit()
        def box_energy(state: np.ndarray) -> float:
            energy_contrib: float = 1 / 2 * box_k * np.sum(np.where(state < 0, state ** 2, 0)) \
                                    + 1 / 2 * box_k * np.sum(np.where(state > L, (state - L) ** 2, 0))
            return energy_contrib

        @njit()
        def single_particle_box_energy(state: np.ndarray, i: int) -> float:
            return box_energy(state[i])

        @njit()
        def box_and_hardcore_energy(state: np.ndarray) -> float:
            return box_energy(state) + hardcore_energy(state)

        @njit()
        def single_particle_box_and_hardcore_energy(state: np.ndarray, i: int) -> float:
            return single_particle_box_energy(state, i) + single_particle_hardcore_energy(state, i)

        @njit()
        def box_and_LJ_energy(state: np.ndarray):
            return box_energy(state) + LJ_energy(state)

        @njit()
        def single_particle_box_and_LJ_energy(state: np.ndarray, i: int):
            return single_particle_box_energy(state, i) + single_particle_LJ_energy(state, i)

        self.pressure = pressure
        self.hardcore_energy = hardcore_energy
        self.box_energy = box_energy
        self.box_and_hardcore_energy = box_and_hardcore_energy
        self.box_and_lj_energy = box_and_LJ_energy
        self.single_particle_pressure = single_particle_pressure
        self.single_particle_box_energy = single_particle_box_energy
        self.single_particle_hardcore_energy = single_particle_hardcore_energy
        self.single_particle_box_and_hardcore_energy = single_particle_box_and_hardcore_energy
        self.single_particle_box_and_lj_energy = single_particle_box_and_LJ_energy




