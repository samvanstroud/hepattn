import numpy as np


def binned(selection, qty, bin_edges, underflow=True, overflow=True, binomial=False):

    """Histogramming with variable bin widths and efficiency calculation

    Arguments:
    ----------
    selection: array_like[bool]
        binary classification status
    qty: array_like
        the physical quantity to be binned
    bin_edges: array_like
        bin edges for histogramming

    Returns:
    --------
    bin_count: ndarray
        bin height
    bin_error:
        standard error of the mean for each bin
    """

    # bin particles based on qty (e.g. pT)
    bin_id = np.full(shape=len(selection), fill_value=-1)
    
    # initialize qty bin id
    bin_count = []
    bin_error = []

    for i in range(len(bin_edges) - 1):
        # find greater than (or equal to) qty bin lower bound
        lb = np.where(qty >= bin_edges[i], True, False)
        if underflow and i == 0:
            # include underflow into 1st bin
            lb = np.where(qty < bin_edges[0], True, lb)
        # find lesser than qty bin upper bound
        ub = np.where(qty < bin_edges[i + 1], True, False)
        if overflow and i == len(bin_edges) - 2:
            # include overflow into last bin
            ub = np.where(qty >= bin_edges[-1], True, ub)
        # qty window boolean array
        bin_select = lb & ub
        # assign bin_id "i" to entries that belongs to the i-th bin
        bin_id = np.where(bin_select, int(i), bin_id)
        
        # calculate efficiency
        # select particles in the i-th bin
        in_pt_bin = selection[bin_select]
        # count total entries
        total_n = len(in_pt_bin)
        # count remaining entries
        valid_n = np.sum(in_pt_bin)
        # calculate SEM
        bin_n = 0 if total_n == 0 else valid_n / total_n
        if binomial:
            # calculate (sqrt)variance of the sample mean
            bin_err = np.sqrt(bin_n * (1 - bin_n) / total_n)
        else:
            # calculate SEM
            bin_err = 0 if total_n == 0 else np.std(in_pt_bin) / np.sqrt(total_n)
        bin_count.append(bin_n)
        bin_error.append(bin_err)
        
    return bin_count, bin_error


def profile_plot(xs, y_span, x_bins, axes, color, label=None, ls="solid"):

    """Create a profile plot at specified subplot axes

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

    Returns:
    --------
    
    """

    for i in range(len(x_bins) - 1):
        label = label if i == 0 else None
        lb, ub = x_bins[i], x_bins[i + 1]
        axes.hlines(y=xs[i], xmin=lb, xmax=ub, color=color, label=label, ls=ls)
        axes.fill_between([lb, ub], [xs[i] - y_span[i], xs[i] - y_span[i]],
                          [xs[i] + y_span[i], xs[i] + y_span[i]],
                          color=color, alpha=0.15, edgecolor="none"
        )


"""
==============
TODO: hist plot for regression evaluation
==============
def histplot():

"""
