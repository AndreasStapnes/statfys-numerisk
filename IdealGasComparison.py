import numpy as np
from system import System
from functionals import ENERGY, stateFunctions
import matplotlib.pyplot as plt
from constants import box_k, k_b, T


lower_l_lim = 0.1
upper_l_lim = 2
stabilization_iterations = 100
average_iterations = 5000

particle_amt = 500

if __name__ == "__main__":
    fig, ax = plt.subplots(1,1)
    lengths = np.logspace(lower_l_lim, upper_l_lim, 20)
    pressures = []
    for length in lengths:
        state_fcns = stateFunctions(length, box_k, energy_type=ENERGY.BOX_ENERGY)
        sys = System(particle_amt, length, state_fcns, jump_scale=0.01)
        sys.explore(stabilization_iterations, log_interval=1000)
        computed_pressure = sys.explore(average_iterations)
        avg_pressure = np.average(computed_pressure)
        pressures.append(avg_pressure)
        print("*",end="")
    ax.plot(lengths, pressures, label="simulation p")


    ideal_pressures = particle_amt * k_b * T / (lengths**2)
    ax.loglog(lengths, ideal_pressures, label="ideal p")
    ax.set_xlabel("length [m]")
    ax.set_ylabel("pressure [pa]")
    ax.set_title("Ideal gas comparison")
    ax.legend()
    fig.show()
