from __future__ import annotations
from typing import List, Tuple, Callable, Dict, NamedTuple
from constants import *
import numpy as np
from numba import njit



def method_setup(box_k: float, L: float) -> Tuple[Callable[[np.ndarray], float],
                                                  Callable[[np.ndarray], float],
                                                  Callable[[np.ndarray], float]]:
    def pressure(state: np.ndarray) -> float:
        pressure_contrib = np.sum(np.where(state < 0, -state, 0)) + \
                           np.sum(np.where(state > L, state - L, 0))
        pressure_contrib *= box_k / L / 4
        return pressure_contrib

    def hardcore_energy(state: np.ndarray) -> float:
        particle_amt = len(state)
        energy_contrib = 0.0
        for i in range(particle_amt):
            for j in range(i,particle_amt):
                energy_contrib += hardcore_pot if np.sum(state[i]*state[j]) < hardcore_diameter else 0
        return energy_contrib

    def box_energy(state: np.ndarray) -> float:
        energy_contrib: float = 1 / 2 * box_k * np.sum(np.where(state < 0, state ** 2, 0)) \
                                + 1 / 2 * box_k * np.sum(np.where(state > L, (state - L) ** 2, 0))
        return energy_contrib

    return pressure, hardcore_energy, box_energy

pressure, hardcore_energy, box_energy = method_setup(box_k, box_length)


jitted_pressure = njit()(pressure)
jitted_hardcore_energy = njit()(hardcore_energy)
jitted_box_energy = njit()(box_energy)



@njit()
def jitted_energy(state: np.ndarray):
    return jitted_hardcore_energy(state) + jitted_box_energy(state)


@njit()
def jitted_explore(initial_state: np.ndarray, L:float, iterations: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    def jump(state):
        next_state = state + np.random.normal(0, jump_scale, np.shape(state))
        return next_state

    def goto_next(state):
        energy = jitted_energy(state)
        proposed_next = jump(state)
        delta_energy = jitted_energy(proposed_next) - energy
        if delta_energy < 0 or np.random.random() < np.exp(-beta * delta_energy):
            state = proposed_next
        return state

    pressures = np.zeros((iterations,))*1.0
    state = initial_state
    for i in range(iterations):
        state = goto_next(state)
        pressures[i] = jitted_pressure(state)

    return state, [pressures]

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
        self.L = L

    def __len__(self):
        return self.particle_amt

    def __getitem__(self, index: int):
        return self.state[index]

    def jump(self):
        next_state = self.state + np.random.normal(0, jump_scale, np.shape(self.state))
        return next_state

    def goto_next(self):
        proposed_next = self.jump()
        delta_energy = jitted_energy(proposed_next) - self.energy()
        if delta_energy < 0 or np.random.random() < np.exp(-beta * delta_energy):
            self.state = proposed_next

    def explore(self, iterations: int):
        state, values = jitted_explore(self.state, self.L, iterations)
        self.state = state
        return values

    def pressure(self) -> float:
        return jitted_pressure(self.state)

    def energy(self):
        return jitted_energy(self.state)

    @classmethod
    def energy_calculation(self, state:np.ndarray, L:float):
        energy_contrib = np.sum(np.where(state < 0, state**2, 0)) \
                       + np.sum(np.where(state > L, state**2, 0) - L)
        energy_contrib *= 1/2*box_k
        return energy_contrib


