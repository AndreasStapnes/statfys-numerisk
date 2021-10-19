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
<<<<<<< HEAD
=======
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

<<<<<<< HEAD
=======
box_k = 3*k_b*T
box_length = 1
>>>>>>> master


<<<<<<< HEAD

=======
jump_scale = 0.04

N = 400




class System:
    state: np.ndarray
    particle_amt: int
    dimension: int
    L: float

    def __init__(self, particle_amt: int, L: float):
        self.particle_amt = particle_amt
        self.positional_dimension = 2
        self.dimension = self.particle_amt*self.positional_dimension
        self.state = np.random.random((self.particle_amt, self.positional_dimension)) * L
>>>>>>> a3ad53ad505d7c4f866ee207acd93f23a68db2a0
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

<<<<<<< HEAD

=======
    def __len__(self):
        return self.particle_amt

    def __getitem__(self, index: int):
        return self.state[index]

    def jump(self):
        next_state = self.state + np.random.normal(0, jump_scale, np.shape(self.state))
        return next_state

    def goto_next(self):
        proposed_next = self.jump()
        delta_energy = System.energy_calculation(proposed_next, self.L) - self.energy()
        if delta_energy < 0 or np.random.random() < np.exp(-beta * delta_energy):
            self.state = proposed_next

    def pressure(self) -> float:
        pressure_contrib = np.sum(box_k*self.state*self.L*np.where(self.state < 0, self.state, 0)) \
            + np.sum(box_k*(self.state-self.L)*np.where(self.state > self.L, self.state, 0))
        return pressure_contrib

    def energy(self):
        return System.energy_calculation(self.state, self.L)

    @classmethod
    def energy_calculation(self, state:np.ndarray, L:float):
        energy_contrib = np.sum(1/2*box_k*np.where(state < 0, state, 0)**2) \
            + np.sum(1/2*box_k*(np.where(state > L, state, 0) - L)**2)
        return energy_contrib
>>>>>>> a3ad53ad505d7c4f866ee207acd93f23a68db2a0


if __name__ == "__main__":
    sys = System(N, box_length)
    pressures = []
    for i in range(100000):
        sys.goto_next()
        pressures.append(sys.pressure())
    plt.plot(pressures)
    print(f" the average pressure was {np.average(pressures)}")
    plt.show()
>>>>>>> master
