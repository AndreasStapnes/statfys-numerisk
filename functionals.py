from __future__ import annotations
from typing import List, NamedTuple, Callable, Tuple
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
import numpy as np
from numba import njit
import copy

k_b = 1.38e-23
T = 100
beta = 1 / (k_b * T)

box_k = 50*k_b*T
box_length = 1

hardcore_diameter = 5e-2
hardcore_pot = 1000 * k_b * T


jump_scale = 0.04

N = 100
jump_scale = 0.01


@njit()
def jitted_pressure(state: np.ndarray, L: float):
    pressure_contrib = np.sum(np.where(state < 0, -state, 0)) + \
                       np.sum(np.where(state > L, state - L, 0))
    pressure_contrib *= box_k / L / 4
    return pressure_contrib

@njit()
def hardcore_energy(state: np.ndarray):
    particle_amt = len(state)
    energy_contrib = 0.0
    for i in range(particle_amt):
        for j in range(i,particle_amt):
            energy_contrib += hardcore_pot if np.sum(state[i]*state[j]) < hardcore_diameter else 0
    return energy_contrib

@njit()
def box_energy(state: np.ndarray, L: float):
    energy_contrib: float = 1 / 2 * box_k * np.sum(np.where(state < 0, state ** 2, 0)) \
                            + 1 / 2 * box_k * np.sum(np.where(state > L, (state - L) ** 2, 0))
    return energy_contrib

@njit()
def jitted_energy(state: np.ndarray, L: float):
    return hardcore_energy(state) + box_energy(state, L)

@njit()
def jitted_explore(initial_state: np.ndarray, L:float, iterations: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    def jump(state):
        next_state = state + np.random.normal(0, jump_scale, np.shape(state))
        return next_state

    def goto_next(state):
        energy = jitted_energy(state, L)
        proposed_next = jump(state)
        delta_energy = jitted_energy(proposed_next, L) - energy
        if delta_energy < 0 or np.random.random() < np.exp(-beta * delta_energy):
            state = proposed_next
        return state

    pressures = np.zeros((iterations,))*1.0
    state = initial_state
    for i in range(iterations):
        state = goto_next(state)
        pressures[i] = jitted_pressure(state, L)

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
        delta_energy = jitted_energy(proposed_next, self.L) - self.energy()
        if delta_energy < 0 or np.random.random() < np.exp(-beta * delta_energy):
            self.state = proposed_next

    def explore(self, iterations: int):
        state, values = jitted_explore(self.state, self.L, iterations)
        self.state = state
        return values

    def pressure(self) -> float:
        return jitted_pressure(self.state, self.L)

    def energy(self):
        return jitted_energy(self.state, self.L)

    @classmethod
    def energy_calculation(self, state:np.ndarray, L:float):
        energy_contrib = np.sum(np.where(state < 0, state**2, 0)) \
                       + np.sum(np.where(state > L, state**2, 0) - L)
        energy_contrib *= 1/2*box_k
        return energy_contrib


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1)
    b_ls = np.linspace(0.2, 2, 1)
    avg_pressures = []
    for b_l in b_ls:
        sys = System(N, b_l)
        pressures = sys.explore(iterations=1000)[0]
        curve: Line2D
        curve, = ax.plot(pressures, label=f"L={b_l}")
        avg_pressure = np.average(pressures)
        avg_pressures.append(avg_pressure)
        print("*", end="")
    ax.set_xlabel("iterations")
    ax.set_ylabel("p")
    fig.show()

    ax: Axes
    fig,ax = plt.subplots(1,1)
    ax.loglog(b_ls, avg_pressures)
    ax.set_xlabel("L")
    ax.set_ylabel("p")
    plt.show()
