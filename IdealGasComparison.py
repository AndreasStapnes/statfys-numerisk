import numpy as np
from system import System
from functionals import ENERGY, stateFunctions
import matplotlib.pyplot as plt
from constants import box_k, k_b, T


lower_l_lim = -1
upper_l_lim = 1
stabilization_iterations = 5000
average_iterations = 50000

particle_amt = 100

if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1)
    lengths = np.logspace(lower_l_lim, upper_l_lim, 10)
    pressures = []
    energies = []
    state_fcns = stateFunctions(box_k, energy_type=ENERGY.HARDCORE_AND_BOX_ENERGY)
    sys = System(particle_amt, 1, state_fcns, jump_scale=0.08, logEnergy=False)
    print("entering for loop")
    for length in lengths:
        sys.reset(length)
        sys.explore(stabilization_iterations, log_interval=1000)
        vals = sys.explore(average_iterations, log_interval=10)
        computed_pressure = vals["pressure"]
        avg_pressure = np.average(computed_pressure)
        #avg_energy = np.average(computed_energy)
        pressures.append(avg_pressure)
        #energies.append(avg_energy)
        print("*", end="")

    ax.plot(lengths, pressures, label="simulation p")
    #bx.plot(lengths, energies, label="simulation E")
    #bx.set_xscale("log")
    #bx.set_yscale("log")

    ideal_pressures = particle_amt * k_b * T / (lengths ** 2)
    ax.loglog(lengths, ideal_pressures, label="ideal p")
    ax.set_xlabel("length [m]")
    ax.set_ylabel("pressure [pa]")
    ax.set_title("Ideal gas comparison")
    ax.legend()
    fig.show()




