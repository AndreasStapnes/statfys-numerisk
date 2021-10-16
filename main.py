import matplotlib.pyplot as plt
from functionals import System
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from constants import *


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
