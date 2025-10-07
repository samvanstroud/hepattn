import numpy as np
from scipy.stats import binned_statistic

from hepattn.utils.stats import combine_mean_std


class CountingHistogram:
    """Simple counting histogram over fixed bin edges.

    Attributes:
        bins (np.ndarray): Monotonically increasing bin edges of shape (nbins + 1,).
        counts (np.ndarray): Bin counts of shape (nbins,), dtype float32.
    """

    def __init__(self, bins: np.ndarray):
        """Initialize a counting histogram.

        Args:
            bins (np.ndarray): Monotonically increasing bin edges of shape (nbins + 1,).
        """
        self.bins = np.asarray(bins)
        self.counts = np.zeros(len(self.bins) - 1, dtype=np.float32)

    def fill(self, values: np.ndarray) -> None:
        """Accumulate counts from values into the histogram bins.

        Args:
            values (np.ndarray): Values to be binned. Any array-like of numeric dtype.
        """
        values = np.asarray(values)
        if values.size == 0:
            return
        counts, _, _ = binned_statistic(values, values, statistic="count", bins=self.bins)
        self.counts += counts.astype(np.float32)


class PoissonHistogram:
    """Accumulates Poisson-binomial style (k, n) tallies per bin.

    Each fill call applies a selection, then sums the numerator (k) and denominator (n)
    within bins of a given field.

    Attributes:
        field (str): Key in `data` whose values define the bin coordinate.
        bins (np.ndarray): Bin edges for the field.
        selection (str): Key in `data` for a boolean mask to select events.
        numerator (str): Key in `data` for the per-event numerator contributions (k).
        denominator (str): Key in `data` for the per-event denominator contributions (n).
        n (np.ndarray): Accumulated denominators per bin.
        k (np.ndarray): Accumulated numerators per bin.
    """

    def __init__(self, field: str, bins: np.ndarray, selection: str, numerator: str, denominator: str):
        """Initialize a Poisson histogram.

        Args:
            field (str): Field name for the binning coordinate in `data`.
            bins (np.ndarray): Bin edges for `field`.
            selection (str): Field name for a boolean selection mask in `data`.
            numerator (str): Field name for per-event numerator (k) in `data`.
            denominator (str): Field name for per-event denominator (n) in `data`.
        """
        self.field = field
        self.bins = np.asarray(bins)
        self.selection = selection
        self.numerator = numerator
        self.denominator = denominator

        self.n = np.zeros(len(self.bins) - 1, dtype=np.float32)
        self.k = np.zeros(len(self.bins) - 1, dtype=np.float32)

    def fill(self, data: dict[str, np.ndarray]) -> None:
        """Accumulate k and n into bins.

        Args:
            data (dict[str, np.ndarray]): Mapping from field names to NumPy arrays.
                Required keys: `self.selection` (bool), `self.numerator` (float-like),
                `self.denominator` (float-like), `self.field` (float-like).
                All arrays must be 1D and broadcastable to the same length.
        """
        sel = np.asarray(data[self.selection]).astype(bool)
        x = np.asarray(data[self.field])[sel].astype(np.float32)
        k = np.asarray(data[self.numerator])[sel].astype(np.float32)
        n = np.asarray(data[self.denominator])[sel].astype(np.float32)

        if x.size == 0:
            return

        n_binned, _, _ = binned_statistic(x, n, statistic="sum", bins=self.bins)
        k_binned, _, _ = binned_statistic(x, k, statistic="sum", bins=self.bins)

        self.n += n_binned.astype(np.float32)
        self.k += k_binned.astype(np.float32)


class GaussianHistogram:
    """Mantains per-bin Gaussian summary stats (n, mean, std) and merges across fills.

    Attributes:
        field (str): Key in `data` whose values define the bin coordinate.
        bins (np.ndarray): Bin edges for the field.
        selection (str): Key in `data` for a boolean mask to select events.
        values (str): Key in `data` for the per-event values to summarize.
        n (np.ndarray): Number of entries per bin (float32).
        mu (np.ndarray): Mean value per bin (float32).
        sigma (np.ndarray): Standard deviation per bin (float32).
    """

    def __init__(self, field: str, bins: np.ndarray, selection: str, values: str):
        """Initialize a Gaussian histogram.

        Args:
            field (str): Field name for the binning coordinate in `data`.
            bins (np.ndarray): Bin edges for `field`.
            selection (str): Field name for a boolean selection mask in `data`.
            values (str): Field name for the values to summarize in `data`.
        """
        self.field = field
        self.bins = np.asarray(bins)
        self.selection = selection
        self.values = values

        self.n = np.zeros(len(self.bins) - 1, dtype=np.float32)
        self.mu = np.zeros(len(self.bins) - 1, dtype=np.float32)
        self.sigma = np.zeros(len(self.bins) - 1, dtype=np.float32)

    def fill(self, data: dict[str, np.ndarray]) -> None:
        """Accumulate Gaussian summary statistics per bin and merge with existing ones.

        This computes per-bin count, mean, and standard deviation for the selected
        subset of events, then combines these statistics with the running totals
        using `combine_mean_std`.

        Args:
            data (dict[str, np.ndarray]): Mapping from field names to NumPy arrays.
                Required keys: `self.selection` (bool), `self.values` (float-like),
                `self.field` (float-like).
                All arrays must be 1D and broadcastable to the same length.
        """
        sel = np.asarray(data[self.selection]).astype(bool)
        x = np.asarray(data[self.field])[sel].astype(np.float32)
        vals = np.asarray(data[self.values])[sel].astype(np.float32)

        if x.size == 0:
            return

        n_new, _, _ = binned_statistic(x, vals, statistic="count", bins=self.bins)
        mu_new, _, _ = binned_statistic(x, vals, statistic="mean", bins=self.bins)
        sig_new, _, _ = binned_statistic(x, vals, statistic="std", bins=self.bins)

        # Merge with existing stats. Assumes combine_mean_std returns (mu, sigma, n).
        mu_comb, sig_comb, n_comb = combine_mean_std(self.mu, self.sigma, self.n, mu_new, sig_new, n_new)

        self.n = n_comb.astype(np.float32)
        self.mu = mu_comb.astype(np.float32)
        self.sigma = sig_comb.astype(np.float32)
