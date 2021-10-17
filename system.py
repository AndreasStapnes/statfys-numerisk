import numpy as np
from numba import njit
from typing import List, Tuple, Callable
from constants import beta


class System:
    state: np.ndarray
    particle_amt: int
    dimension: int
    L: float
    jump_scale: float

    pressure: Callable[[np.ndarray], float]
    energy: Callable[[np.ndarray], float]

    def __init__(self, particle_amt: int, L: float, **kwargs):
        self.particle_amt = particle_amt
        self.positional_dimension = 2           #2D system
        self.dimension = self.particle_amt*self.positional_dimension
        self.L = L
        self.pressure = kwargs.get('pressure', None)
        self.energy = kwargs.get('energy', None)
        self.jump_scale = kwargs.get('jump_scale', 1)

        self.compile()

        #Generating initial random state
        self.state = np.random.random((self.particle_amt, self.positional_dimension)) * L

    def compile(self):
        jump_scale = self.jump_scale
        energy = self.energy
        pressure = self.pressure

        @njit()
        def jitted_explore(initial_state: np.ndarray, iterations: int) -> Tuple[np.ndarray, List[np.ndarray]]:
            def jump(state):
                next_state = state + np.random.normal(0, jump_scale, np.shape(state))
                return next_state

            def goto_next(state):
                current_energy = energy(state)
                proposed_next = jump(state)
                delta_energy = energy(proposed_next) - current_energy
                if delta_energy < 0 or np.random.random() < np.exp(-beta * delta_energy):
                    state = proposed_next
                return state

            pressures = np.zeros((iterations,)) * 1.0
            state = initial_state
            for i in range(iterations):
                state = goto_next(state)
                pressures[i] = pressure(state)

            return state, [pressures]
        self._explore = jitted_explore

    def __len__(self):
        return self.particle_amt

    def __getitem__(self, index: int):
        return self.state[index]

    def jump(self):
        next_state = self.state + np.random.normal(0, self.jump_scale, np.shape(self.state))
        return next_state

    def goto_next(self):
        proposed_next = self.jump()
        delta_energy = self.energy(proposed_next) - self.energy()
        if delta_energy < 0 or np.random.random() < np.exp(-beta * delta_energy):
            self.state = proposed_next

    def explore(self, iterations: int):
        state, values = self._explore(self.state, iterations)
        self.state = state
        return values
