import numpy as np
from numba import njit
from typing import List, Tuple, Callable, Dict
from constants import k_b, T
from functionals import stateFunctions


class System:
    state: np.ndarray
    particle_amt: int
    dimension: int
    L: float
    jump_scale: float

    pressure: Callable[[np.ndarray, float], float]
    energy: Callable[[np.ndarray, float], float]

    def __init__(self, particle_amt: int, L: float, state_functions: stateFunctions, temperature: float = T, **kwargs):
        self.particle_amt = particle_amt
        self.positional_dimension = 2           # 2D system
        self.dimension = self.particle_amt*self.positional_dimension
        self.L = L
        self.state_functions = state_functions
        self.jump_scale = kwargs.get("jump_scale", 1)
        self.T = temperature

        self.logEnergy = kwargs.get("logEnergy", True)
        self.logPressure = kwargs.get("logPressure", True)

        self.compile()

        # Generating initial random state
        self.reset()

    def reset(self, L: float=None, recompile: bool=False):
        if L is not None:
            self.L = L
        self.state = np.random.random((self.particle_amt, self.positional_dimension)) * self.L
        if recompile:
            self.compile()

    def compile(self):
        jump_scale = self.jump_scale
        energy = self.state_functions.get_energy()
        pressure = self.state_functions.get_pressure()
        single_energy = self.state_functions.get_single_particle_energy()
        single_pressure = self.state_functions.get_single_particle_pressure()
        T = self.T
        pos_dim = self.positional_dimension
        logEnergy = self.logEnergy
        logPressure = self.logPressure
        @njit()
        def jitted_explore(initial_state: np.ndarray, L: float, iterations: int, log_interval=100) -> Tuple[np.ndarray, List[List[float]]]:
            beta = 1.0 / (k_b * T)

            def particle_choice(state):
                i = np.random.randint(0, len(state))
                return i

            def jump():
                stepped_length = np.random.uniform(-jump_scale, jump_scale, (pos_dim,))
                return stepped_length

            def goto_next(state, i):
                single_step = jump()
                current_energy = single_energy(state, L, i)
                position_i_current = state[i] + 0
                state[i] += single_step
                next_energy = single_energy(state, L, i)
                delta_energy = next_energy - current_energy
                if not (delta_energy < 0 or np.random.random() < np.exp(-beta * delta_energy)) :
                    state[i] = position_i_current
                return state

            pressures = [0.0][:0]
            energies = [0.0][:0]

            state = initial_state
            for iter in range(iterations):
                i = particle_choice(state)
                state = goto_next(state, i)
                if iter % log_interval == 0:
                    if logPressure: pressures.append(pressure(state, L))
                    if logEnergy: energies.append(energy(state, L))
            return state, [pressures, energies]
        self._explore = jitted_explore

    def __len__(self):
        return self.particle_amt

    def __getitem__(self, index: int):
        return self.state[index]

    def explore(self, iterations: int, log_interval: int = 1):
        state, values = self._explore(self.state, self.L, iterations, log_interval)
        self.state = state
        return dict(zip(["pressure", "energy"], values))
