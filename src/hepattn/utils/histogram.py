import numpy as np
from scipy.stats import binned_statistic
from torch import Tensor

from hepattn.utils.stats import combine_mean_std


class CountingHistogram:
    def __init__(self, bins: np.ndarray):
        self.bins = bins
        self.counts = np.zeros(len(bins) - 1, dtype=np.float32)

    def fill(self, values):
        counts, _, _ = binned_statistic(values, values, statistic="count", bins=self.bins)
        self.counts += counts


class PoissonHistogram:
    def __init__(self, field: str, bins: np.ndarray, selection: str, numerator: str, denominator: str):
        self.field = field
        self.bins = bins
        self.selection = selection
        self.numerator = numerator
        self.denominator = denominator

        self.n = np.zeros(len(bins) - 1, dtype=np.float32)
        self.k = np.zeros(len(bins) - 1, dtype=np.float32)

    def fill(self, data: dict[str:Tensor]) -> None:
        selection = data[self.selection].bool()
        k = data[self.numerator][selection].float()
        n = data[self.denominator][selection].float()
        x = data[self.field][selection].float()

        # If the selection is empty then can just return
        if len(x) == 0:
            return

        n_binned, _, _ = binned_statistic(x, n, statistic="sum", bins=self.bins)
        k_binned, _, _ = binned_statistic(x, k, statistic="sum", bins=self.bins)

        self.n += n_binned
        self.k += k_binned


class GaussianHistogram:
    def __init__(self, field: str, bins: np.ndarray, selection: str, values: str):
        self.field = field
        self.bins = bins
        self.selection = selection
        self.values = values

        self.n = np.zeros(len(bins) - 1, dtype=np.float32)
        self.mu = np.zeros(len(bins) - 1, dtype=np.float32)
        self.sigma = np.zeros(len(bins) - 1, dtype=np.float32)

    def fill(self, data: dict[str:Tensor]) -> None:
        selection = data[self.selection].bool()
        x = data[self.field][selection].float()
        values = data[self.values][selection].float()

        n, _, _ = binned_statistic(x, values, statistic="count", bins=self.bins)
        mu, _, _ = binned_statistic(x, values, statistic="mean", bins=self.bins)
        sig, _, _ = binned_statistic(x, values, statistic="std", bins=self.bins)

        mu, sig, n = combine_mean_std(self.mu, self.sig, self.n, mu, sig, n)

        self.n = n
        self.mu = mu
        self.sig = sig
