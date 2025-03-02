import numpy as np
import matplotlib.pyplot as plt

def velocity(E: float) -> float:
    """
    Parameters
    ----------
    E: float
        Electron kinetic energy in eV

    Returns
    -------
    velocity: float
        Electron velocity with given kinetic energy E in mm/s
    """
    return np.sqrt(2*E*1.60218e-19/9.1093837e-31) * 1e3 # mm/s

def time_of_flight(E: float): # ns
     """
    Parameters
    ----------
    E: float
        Electron kinetic energy in eV

    Returns
    -------
    time_of_flight: float
        Electron time of flight in a straight line in ns
    """
     return 1120/velocity(E)*1e9

def decorate(axes: plt.axes, x_range: tuple, y_range: tuple, xlabel='', ylabel='', set=True):
    """
    Parameters
    ----------
    axes: plt.axes
        object from matplotlib.pyplot
    x_range: tuple
        (x_start, x_stop, x_step)
    y_range: tuple
        (y_start, y_stop, y_step)
    xlabel: string
        x axis label
    ylabel: string
        y axis label
    set: bool
        Whether to perform axes.set() method
    """
    x_start, x_end, x_interval = x_range[0], x_range[1], x_range[2]
    y_start, y_end, y_interval = y_range[0], y_range[1], y_range[2]
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.tick_params(which = "both", direction = "in")
    axes.grid(True, which="major", ls="-", c="gray")
    axes.grid(True, which="minor", ls=":")
    axes.set_xticks(np.arange(x_start, x_end+1, x_interval))
    axes.set_xticks(np.arange(x_start, x_end+1, x_interval/2), minor=True)
    axes.set_yticks(np.arange(y_start, y_end+1, y_interval))
    axes.set_yticks(np.arange(y_start, y_end+1, y_interval/2), minor=True)
    if set:
        axes.set(xlim=(x_start, x_end), ylim=(y_start, y_end))