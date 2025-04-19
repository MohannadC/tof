import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, find_peaks, peak_widths
from scipy.optimize import curve_fit
from IPython.display import display_html
from itertools import chain,cycle

ELECTRON_MASS = 9.1093837139e-31
TUBE_LENGTH = 1000
RET_POS = 90
ELECTRON_CHARGE = 1.60217663e-19


def read_file(path: str, delimeter=",") -> list:
    """
    Parameters
    ----------
    path: str
        Path to file
    delimeter: str
        Delimeter symbol, default ','
    """
    with open(path, mode="r") as ifile:
        cols = len(ifile.readline().split(delimeter))
        ifile.seek(0)
        data = [[] for _ in range(cols)]
        for line in ifile:
            tmp = line.split(delimeter)
            for i in range(cols):
                data[i].append(float(tmp[i]))
    return data


def gaussian(x, x0, sigma, A):
    """
    Gaussian distribution function

    Parameters
    ----------
    x:
        x-value
    x0: float
        Center of distribution
    sigma: float
        Standard deviation
    A: float
        Amplitude/2πσ
    """
    return (A / (2 * np.pi * sigma)) * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def exp_tail(x, x0, tau, B):
    """
    Exponential tail function. Equals zero below x0

    Parameters
    ----------
    x:
        x-value
    x0: float
        Start value
    tau: float
        Exponent multiplier
    B: float
        Amplitude
    """
    return B * np.exp(-(x - x0) / tau) * np.heaviside(x - x0, 1)


def gauss_with_tail(x, x0, sigma, A, tau, B):
    """
    Convolution of gaussian and exponential tail functions

    Parameters
    ----------
    x:
        x-value
    x0: float
        Center of distribution and start value for exp_tail
    sigma: float
        Standard deviation
    A: float
        Gaussian amplitude/2πσ
    tau: float
        Exponent multiplier
    B: float
        Exponent amplitude
    """
    step = x[1] - x[0]
    return (
        fftconvolve(gaussian(x, x0, sigma, A),
                    exp_tail(x, x0, tau, B), "full") * step
    )[::2]


def approximate(x, y, ax, p0=None):
    """
    Gaussian with exponential tail approximating function.

    Parameters
    ----------
    x: array_like
        x-data
    y: array_like
        y-data
    p0: array_like
        Initial guess for the parameters
    ax:
        Axes to plot

    Returns
    -------
    (x_peak, x_peak_error, peak_width)
    """
    step = x[1] - x[0]
    params, cov = curve_fit(gauss_with_tail, x, y, p0)
    sigma = np.sqrt(np.diag(cov))
    y_approx = gauss_with_tail(x, *params)
    ax.plot(x, y_approx, color="blue", lw=1)
    peaks, _ = find_peaks(y_approx, height=0.01)
    results_half = peak_widths(y_approx, peaks, rel_height=0.5)[0]
    return (x[peaks][0], sigma[0], results_half[0] * step)


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
    return np.sqrt(2 * E * 1.60218e-19 / 9.1093837e-31) * 1e3  # mm/s


def time_of_flight(E: float):  # ns
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
    return TUBE_LENGTH / velocity(E) * 1e9


def time_ret(E):
    """
    Parameters
    ----------
    E: float
        Electron kinetic energy in eV
    
    Returns
    -------
    time_ret: float
        Electron time of flight in a straight line with retarding field in ns
    """
    return np.sqrt(ELECTRON_MASS/2)*(RET_POS*1e-3/np.sqrt(E*ELECTRON_CHARGE) +
                                     (TUBE_LENGTH-RET_POS)*1e-3/np.sqrt((E-50)*ELECTRON_CHARGE))*1e9


def res(energies):
    t_ret, t_tof, R_ret, R_tof = [], [], [], []
    for E in energies:
        t1 = time_ret(E)
        t2 = time_ret(E - 1)
        t1_tof = time_of_flight(E)
        t2_tof = time_of_flight(E - 1)
        dt = t2-t1
        dt_tof = t2_tof - t1_tof
        R1 = dt_tof/time_of_flight(E)/2 # двойка потому что по критерию рэлея
        R = dt/time_ret(E)/2
        t_ret.append(t1)
        t_tof.append(t1_tof)
        R_ret.append(np.round(R*100, 2))
        R_tof.append(np.round(R1*100, 2))
    return {
        "Energy, eV": energies[:len(t_ret)],
        "TOF, ns": t_tof,
        "Rtof": R_tof,
        "TRET, ns": t_ret,
        "Rret": R_ret
    }


def time_to_energy(t, t0, E0, s):
    """
    Conversion function for time-of-flight in ns

    Parameters
    ----------
    t:
        Time-of-flight in ns
    t0: float
        Temporal offset in ns
    E0: float
        Energy offset in eV
    s: float
        Drift length in mm
    Returns
    -------
    Corresponding energy value in eV
    """
    return 6.24e18 * 0.5 * ELECTRON_MASS * (1e-3 * s / (1e-9 * t - t0)) ** 2 + E0


def get_bins(xpoints, xdata) -> tuple:
    """
    Returns bin indexes to corresponging xpoints

    Parameters
    ----------
    xpoints: tuple
        (xstart, xstop)
    xdata:
        Data divided into bins
    Returns
    -------
    (istart, istop) - start and stop bin numbers
    """
    step = xdata[1] - xdata[0]
    istart = int(np.floor((xpoints[0] - xdata[0]) / step))
    istop = int(np.floor((xpoints[1] - xdata[0]) / step))
    if istart < 0:
        istart = 0
    elif istop > len(xdata):
        istop = len(xdata)
    return istart, istop


def decorate(axes: plt.axes, x_range: tuple, y_range: tuple, xlabel='', ylabel='', title='', legend=False, set=True):
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
    axes.set_title(title)
    axes.tick_params(which="both", direction="in")
    axes.grid(True, which="major", ls="-", c="gray")
    axes.grid(True, which="minor", ls=":")
    axes.set_xticks(np.arange(x_start, x_end+1, x_interval))
    axes.set_xticks(np.arange(x_start, x_end+1, x_interval/2), minor=True)
    axes.set_yticks(np.arange(y_start, y_end+1, y_interval))
    axes.set_yticks(np.arange(y_start, y_end+1, y_interval/2), minor=True)
    if set:
        axes.set(xlim=(x_start, x_end), ylim=(y_start, y_end))
    if legend:
        axes.legend()

def display_side_by_side(*args,titles=cycle([''])):
    """Displays pandas DataFrames side by side"""
    # Add div to put output in a flexbox with the required properties.
    open_div = '<div style="display: flex; justify-content: center; align-items: flex-start";>'
    close_div = "</div>"
    html_str=''
    for df,title in zip(args, chain(titles,cycle(['</br>'])) ):
        # Put everything in a div and add the headline before to appear on top
        html_str+=f'<div>'
        html_str+=f'<h2 style="text-align: center;">{title}</h2>'
        html_str+='<th style="text-align:center;"><td style="vertical-align:top">'
        html_str+=df.to_html().replace('table','table style="display:inline"')
        html_str+='</td></th></div>'
        # Add spacing to tables.
        html_str+='<span style="display:inline-block; width: 10px;"></span>'
    # Put everything together.
    display_html(f"{open_div}{html_str}{close_div}",raw=True)

def display_in_column(*args,titles=cycle([''])):
    html_str=''
    for df,title in zip(args, chain(titles,cycle(['</br>'])) ):
        html_str+='<th style="text-align:center"><td style="vertical-align:top">'
        html_str+=f'<h2 style="text-align: center;">{title}</h2>'
        html_str+=df.to_html().replace('table','table style="display:inline"')
        html_str+='</td></th>'
    display_html(html_str,raw=True)