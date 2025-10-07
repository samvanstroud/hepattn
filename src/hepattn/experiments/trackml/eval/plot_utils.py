import numpy as np


def binned(selection, qty, bin_edges, underflow=True, overflow=True, binomial=False):
    """Histogramming with variable bin widths and efficiency calculation.

    Arguments:
    ----------
    selection: array_like[bool]
        binary classification status
    qty: array_like
        the physical quantity to be binned
    bin_edges: array_like
        bin edges for histogramming, in increasing order
    underflow: bool
        specify whether to include underflow (entries lesser than min bin edge added into first bin), truncated otherwise
    overflow: bool
        specify whether to include overflow (entries greater max bin edge added into last bin), truncated otherwise
    binomial: bool
        specify whether to calculate bin error using SEM of a binomial distribution: [sqrt(VAR) / n]

    Returns:
    --------
    bin_count: ndarray
        bin height
    bin_error:
        standard error of the mean for each bin

    """
    # assign bin_id "i" to entries that belong to the i-th bin (starts from 1)
    bin_id = np.digitize(qty, bin_edges)
    if underflow:
        bin_id = np.where(bin_id == 0, 1, bin_id)
    if overflow:
        bin_id = np.where(bin_id == len(bin_edges), len(bin_edges) - 1, bin_id)

    bin_count = np.bincount(bin_id, minlength=len(bin_edges))[1::]
    # bin_id starts from 1 but bin_counts starts from 0
    bin_error = []
    bin_eff = []

    for i in range(len(bin_edges) - 1):
        # calculate efficiency
        # select particles in the i-th bin (starts from 1...)
        in_pt_bin = selection[bin_id == i + 1]
        # count total entries
        total_n = bin_count[i]
        # count remaining entries
        valid_n = np.sum(in_pt_bin)

        if total_n == 0:
            # if bin is empty
            bin_n = 0
            bin_err = 0.0
        elif binomial:
            # calculate sqrt(variance) of the sample mean
            bin_n = valid_n / total_n
            bin_err = np.sqrt(bin_n * (1 - bin_n) / total_n)
        else:
            # calculate SEM
            bin_n = valid_n / total_n
            bin_err = np.std(in_pt_bin) / np.sqrt(total_n)

        bin_eff.append(bin_n)
        bin_error.append(bin_err)

    return bin_eff, bin_error


def profile_plot(xs, y_span, x_bins, axes, colour, label=None, ls="solid"):
    """Create a profile plot at specified subplot axes.

    Arguments:
    ----------
    xs: array_like
        bin height
    y_span: array_like
        error bar for each bin
    x_bins: array_like
        historgram bin edges
    axes: Axes
        the subplot axes on which plot is created
    colour: str
        colour of the line segment and error band
    label: str
        string identifier for the histogram
    ls: str
        line style for line segement

    """
    for i in range(len(x_bins) - 1):
        label = label if i == 0 else None
        lb, ub = x_bins[i], x_bins[i + 1]
        axes.hlines(y=xs[i], xmin=lb, xmax=ub, color=colour, ls=ls, label=label)
        axes.fill_between(
            [lb, ub], [xs[i] - y_span[i], xs[i] - y_span[i]], [xs[i] + y_span[i], xs[i] + y_span[i]], color=colour, alpha=0.15, edgecolor="none"
        )


def hist_plot(xs, bins, xrange, name, axes, colour, density=True, lw=1.5):
    """Create a histogram plot at specified subplot axes (for regression residuals).

    Arguments:
    ----------
    xs: array_like
        Input data
    bins: int
        Amount of bins
    xrange: (float, float)
        The lower and upper range of the bins.
    name: str
        Label of the input data
    axes: Axes
        the subplot axes on which plot is created
    colour: str
        colour of histogram outline
    density: bool
        specify whether to normalize the histogram
    lw: float
        line weight for the histogram outline

    Returns:
    --------
    label: str
        Formatted label for the input data

    """
    xs_mean = np.mean(xs)
    # xs_std = np.std(xs)
    xs_q25 = np.quantile(xs, 0.25)
    xs_q75 = np.quantile(xs, 0.75)
    xs_iqr = xs_q75 - xs_q25
    xs = np.clip(xs, xrange[0], xrange[1])
    label = name + "\n" + rf"$\mu = {xs_mean:.3f}$" + " " + rf"IQR $ = {xs_iqr:.3f}$"
    axes.hist(xs, bins=bins, histtype="step", color=colour, density=density, lw=lw)

    return label
