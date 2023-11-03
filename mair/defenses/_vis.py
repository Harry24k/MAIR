# Modified from https://github.com/Harry24k/pytorch-custom-utils
from collections.abc import Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import torch


def init_settings(
    font_size=None,
    title_size=None,
    label_size=None,
    xlabel_size=None,
    ylabel_size=None,
    legend_size=None,
):
    params = {
        "font.size": font_size,
        "axes.titlesize": title_size,
        "axes.labelsize": label_size,
        "xtick.labelsize": xlabel_size,
        "ytick.labelsize": ylabel_size,
        "legend.fontsize": legend_size,
    }
    _del_none(params)
    pylab.rcParams.update(params)
    print("rcParams updated.")


def init_plot(
    ax=None,
    figsize=(6, 6),
    title=None,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    pad_ratio=0,
    show_axis=True,
    show_grid=False,
    tight=True,
):
    if ax is None:
        ax = plt.subplots(1, 1, figsize=figsize)[1]

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xlim is not None:
        xlim = _add_margin(*xlim, pad_ratio)
        ax.set_xlim(xlim)

    if ylim is not None:
        ylim = _add_margin(*ylim, pad_ratio)
        ax.set_ylim(ylim)

    if not show_axis:
        ax.axis("off")

    if show_grid:
        ax.grid()

    if tight:
        plt.tight_layout()

    return ax


def make_twin(ax, ylabel=None, ylim=None, pad_ratio=0):
    ax2 = ax.twinx()

    if ylabel is not None:
        ax2.set_ylabel(ylabel)

    if ylim is not None:
        ylim = _add_margin(*ylim, pad_ratio)
        ax2.set_ylim(ylim)

    return ax2


def plot_line(
    ax,
    input,
    x=None,
    linewidth=None,
    linestyle=None,
    color=None,
    label=None,
    alpha=None,
    dashes=None,
    marker=None,
    markerfacecolor=None,
    markersize=None,
):
    kwargs = {
        "linewidth": linewidth,
        "linestyle": linestyle,
        "color": color,
        "label": label,
        "alpha": alpha,
        "dashes": dashes,
        "marker": marker,
        "markerfacecolor": markerfacecolor,
        "markersize": markersize,
    }
    input = _to_numpy(input)
    _del_none(kwargs)

    if x is None:
        x = list(range(len(input) + 1))[1:]

    ax.plot(x, input, **kwargs)


def _add_margin(x_min, x_max, ratio=0.3):
    range = x_max - x_min
    mean = (x_max + x_min) / 2
    return mean - range / 2 * (1 + ratio), mean + range / 2 * (1 + ratio)


def _del_none(input):
    keys = input.copy().keys()
    for key in keys:
        if input[key] is None:
            del input[key]


def _to_numpy(input):
    if isinstance(input, np.ndarray):
        return input
    elif isinstance(input, pd.core.series.Series):
        return input.values
    elif isinstance(input, pd.core.frame.DataFrame):
        return input.values
    elif isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    else:
        return input


#         raise RuntimeError("Please input tensor or numpy array.")


def _to_array(inputs, lengths):
    for i, input in enumerate(inputs):
        if not isinstance(input, Iterable):
            inputs[i] = [input] * lengths
    return inputs
