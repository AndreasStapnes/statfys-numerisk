from __future__ import annotations
from typing import List, NamedTuple, Callable, Tuple
import matplotlib.pyplot as plt
import numpy as np
import copy

k_b = 1.38e-23
T = 298
beta = 1 / (k_b * T)

box_k = k_b*T
box_length = 1

hardcore_diameter = 1e-3
hardcore_pot = 1000 * k_b * T

jump_scale = 0.1

N = 400


class Particle(NamedTuple):
    x: float
    y: float

    @classmethod
    def distance(cls, particle_i: Particle, particle_j: Particle):
        return np.sqrt((particle_i.x - particle_j.x) ** 2 + (particle_i.y - particle_j.y) ** 2)

    def jump(self):
        return Particle(self.x + np.random.normal(0, jump_scale),
                        self.y + np.random.normal(0, jump_scale))


class System:
    state: List[Particle]
    particle_amt: int
    dimension: int
    global_potential: Callable[[Particle], float] = lambda self, particle: 0
    inter_particle_potential: Callable[[Particle, Particle], float] = lambda self, particle_i, particle_j: 0
    pressure_method: Callable[[Particle], float] = lambda self, particle: 0

    def __init__(self, state: List[Particle] = None):
        self.particle_amt = len(state)
        self.dimension = self.particle_amt * len(state[0])
        if state is not None:
            self.state = [Particle(*elem) for elem in state]

    def __len__(self):
        return self.particle_amt

    def __getitem__(self, index: int):
        return self.state[index]

    def __abs__(self):
        amt = len(self)
        energy = 0.0
        for i in range(amt):
            particle_i = self[i]
            energy += self.global_potential(particle_i)
            for j in range(i, amt):
                energy += self.inter_particle_potential(particle_i, self[j])
        return energy

    def jump(self):
        System2 = System([particle.jump() for particle in sys.state])
        System2.global_potential = self.global_potential
        System2.inter_particle_potential = self.inter_particle_potential
        System2.pressure_method = self.pressure_method
        return System2

    def next_state(self):
        proposed_next = self.jump()
        delta_energy = abs(proposed_next) - abs(self)
        if delta_energy < 0:
            return proposed_next
        if np.random.random() < np.exp(-beta * delta_energy):
            return proposed_next
        else:
            return self

    def pressure(self) -> float:
        total_pressure = 0.0
        for particle in self.state:
            total_pressure += self.pressure_method(particle)
        return total_pressure




def single_potential(x: float, L: float):
    if x > L:
        return 1 / 2 * box_k * (x - L) ** 2
    elif x < 0:
        return 1 / 2 * box_k * x ** 2
    else:
        return 0


def single_force(x: float, L: float):
    if x > L:
        return -box_k * (x - L)
    elif x < 0:
        return box_k * x
    else:
        return 0


def box_potential(particle: Particle):
    x, y = particle.x, particle.y
    return single_potential(x, box_length) + single_potential(y, box_length)


def box_force(particle: Particle, L: float) -> Tuple[float, float]:
    x, y = particle.x, particle.y
    return single_force(x, L), single_force(y, L)


def single_pressure(particle: Particle):
    return np.sum(np.abs(np.array(box_force(particle, box_length)) / (4 * box_length)))


def hardcore_potential(particle_i: Particle, particle_j: Particle):
    return hardcore_pot if Particle.distance(particle_i, particle_j) < 2 * hardcore_diameter else 0


if __name__ == "__main__":
    sys = System(np.random.random((N, 2)))
    sys.global_potential = box_potential
    sys.pressure_method = single_pressure
    pressures = []
    for i in range(3000):
        sys = sys.next_state()
        pressures.append(sys.pressure())
    plt.plot(pressures)
    print(f" the average pressure was {np.average(pressures)}")
    plt.show()
