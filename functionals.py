from __future__ import annotations
from typing import List, NamedTuple, Callable, Tuple
import matplotlib.pyplot as plt
import numpy as np
import copy

k_b = 1.38e-23
T = 100
beta = 1 / (k_b * T)

box_k = 3*k_b*T
box_length = 1

hardcore_diameter = 1e-3
hardcore_pot = 1000 * k_b * T

jump_scale = 0.1

N = 300




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
        delta_energy = System.energy_calculation(proposed_next, self.L) - self.energy()
        if delta_energy < 0 or np.random.random() < np.exp(-beta * delta_energy):
            self.state = proposed_next

    def pressure(self) -> float:
        pressure_contrib = np.sum(np.where(self.state < 0, -self.state, 0)) + \
                           np.sum(np.where(self.state > self.L, self.state-self.L, 0))
        pressure_contrib *= box_k/self.L
        return pressure_contrib

    def energy(self):
        return System.energy_calculation(self.state, self.L)

    @classmethod
    def energy_calculation(self, state: np.ndarray, L: float):
        energy_contrib = 1/2*box_k*np.sum(np.where(state < 0, state**2, 0)) \
            + 1/2*box_k*np.sum(np.where(state > L, (state-L)**2, 0))
        return energy_contrib


if __name__ == "__main__":
    sys = System(N, box_length)
    pressures = []
    for i in range(1000): sys.goto_next()
    for i in range(100000):
        sys.goto_next()
        pressures.append(sys.pressure())
    plt.plot(pressures)
    avg_pressure = np.average(pressures)
    print(f" the average pressure was {avg_pressure}")
    print(f"Assuming V=L^2, (N kb T)/(p V) = {N*k_b*T/(avg_pressure*box_length**2)}")

    plt.show()
