import numpy as np
from numba import njit
from typing import List, Tuple, Callable
from constants import beta
from functionals import stateFunctions


class System:
    state: np.ndarray
    particle_amt: int
    dimension: int
    L: float
    jump_scale: float

    pressure: Callable[[np.ndarray], float]
    energy: Callable[[np.ndarray], float]

    def __init__(self, particle_amt: int, L: float, state_functions: stateFunctions, **kwargs):
        self.particle_amt = particle_amt
        self.positional_dimension = 2           #2D system
        self.dimension = self.particle_amt*self.positional_dimension
        self.L = L
        self.state_functions = state_functions
        self.jump_scale = kwargs.get("jump_scale", 1)

        self.compile()

        #Generating initial random state
        self.state = np.random.random((self.particle_amt, self.positional_dimension)) * L

    def compile(self):
        jump_scale = self.jump_scale
        energy = self.state_functions.get_energy()
        pressure = self.state_functions.get_pressure()
        single_energy = self.state_functions.get_single_particle_energy()
        single_pressure = self.state_functions.get_single_particle_pressure()

        pos_dim = self.positional_dimension

        @njit()
        def jitted_traverse(initial_state: np.ndarray, iterations: int) -> Tuple[np.ndarray, List[np.ndarray]]:
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


        @njit()
        def jitted_explore(initial_state: np.ndarray, iterations: int, log_interval=100) -> Tuple[np.ndarray, List[np.ndarray]]:
            def particle_choice(state):
                i = np.random.randint(0, len(state))
                return i

            def jump():
                stepped_length = np.random.uniform(-jump_scale, jump_scale, (pos_dim,))
                return stepped_length

            def goto_next(state, i):
                single_step = jump()
                current_energy = single_energy(state, i)
                position_i_current = state[i] + 0
                state[i] += single_step
                next_energy = single_energy(state, i)
                delta_energy = next_energy - current_energy
                if not (delta_energy < 0 or np.random.random() < np.exp(-beta * delta_energy)) :
                    state[i] = position_i_current
                return state

            pressures = []

            state = initial_state
            for iter in range(iterations):
                i = particle_choice(state)
                state = goto_next(state, i)
                if iter % log_interval == 0:
                    pressures.append(pressure(state))

            return state, [pressures]

        self._traverse = jitted_traverse
        self._explore = jitted_explore


    def __len__(self):
        return self.particle_amt

    def __getitem__(self, index: int):
        return self.state[index]

    def traverse(self, iterations: int):
        state, values = self._traverse(self.state, iterations)
        self.state = state
        return values

    def explore(self, iterations: int, log_interval: int = 1):
        state, values = self._explore(self.state, iterations, log_interval)
        self.state = state
        return values
