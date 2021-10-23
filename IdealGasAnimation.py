import numpy as np
from system import System
from functionals import ENERGY, stateFunctions
import matplotlib.pyplot as plt
from constants import box_k, k_b, T
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from typing import List
from matplotlib.axes import Axes

plot_amt = 100      #antall partikler

anim_iters = 1000   #Iterasjoner mellom hvert plot
anim_frames = 300   #Antall frames

if __name__ == "__main__":

    #Plotting a single system

    state_fcns = stateFunctions(box_k, energy_type=ENERGY.LJ_AND_BOX_ENERGY)
    sys = System(plot_amt, 1, state_fcns, jump_scale=0.01, logEnergy=False)
    ax: Axes
    fig, ax = plt.subplots(1,1)
    data_storage = [[[x_data],[y_data]] for x_data, y_data in sys.state]
    curves:List[Line2D] = ax.plot(*data_storage)
    ax.set_aspect("equal")
    ax.set_xlim([-0.2, 1.2])
    ax.set_ylim([-0.2, 1.2])
    ax.axvline(1, color="k"); ax.axvline(0, color="k")
    ax.axhline(0, color="k"), ax.axhline(1, color="k")


    def animate(*args):
        pressure= sys.explore(anim_iters, log_interval=anim_iters)["pressure"]
        print("*", end="")
        ax.set_title(f"p={pressure[0]:.3g} pa")
        for enum, (x_data, y_data) in enumerate(sys.state):
            data_storage[enum][0].append(x_data)
            data_storage[enum][1].append(y_data)
            data_storage[enum][0] = data_storage[enum][0][-10:]
            data_storage[enum][1] = data_storage[enum][1][-10:]
        for enum, line in enumerate(curves):
            line.set_data(data_storage[enum])
        return curves
    anim = FuncAnimation(fig, animate, anim_frames)
    anim.save("anim.gif", fps=30)
    print("")
    print("anim.gif may now be found in CWD")



