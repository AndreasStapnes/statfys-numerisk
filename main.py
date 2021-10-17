import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from system import System
from constants import N, box_k

from functionals import stateFunctions, ENERGY

if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1)
    b_ls = np.linspace(0.2, 2, 1)
    avg_pressures = []
    for b_l in b_ls:
        state_funcs = stateFunctions(b_l, box_k, energy_type=ENERGY.HARDCORE_AND_BOX_ENERGY)
        sys = System(N, b_l,
                     energy=state_funcs.get_energy(),
                     pressure=state_funcs.get_pressure(),
                     jump_scale=0.01)

        pressures = sys.explore(iterations=100000)[0]
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
