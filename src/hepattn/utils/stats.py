import numpy as np


def frequentist_binomial_error(k, n):
    return np.sqrt((k / n) * (1 - (k / n)) / n)


def bayesian_binomial_error(k, n):
    return np.sqrt(((k + 1) * (k + 2)) / ((n + 2) * (n + 3)) - ((k + 1) / (n + 2)) ** 2)


def combine_mean_std(mu1, sigma1, n1, mu2, sigma2, n2):
    """Combine mean, standard deviation, and sample size from two independent samples.
    Works with scalars or NumPy arrays.
    Handles cases where n1 or n2 can be zero.

    Returns:
    -------
    mu_combined : np.ndarray
        Combined mean.
    sigma_combined : np.ndarray
        Combined standard deviation.
    n_combined : np.ndarray
        Combined sample size.
    """
    mu1, sigma1, mu2, sigma2 = map(np.asarray, (mu1, sigma1, mu2, sigma2))
    n1, n2 = np.asarray(n1), np.asarray(n2)

    total_n = n1 + n2

    # Initialize outputs
    shape = np.broadcast(mu1, mu2, sigma1, sigma2, n1, n2).shape
    mu_combined = np.full(shape, np.nan, dtype=float)
    sigma_combined = np.full(shape, np.nan, dtype=float)

    # Case: n1 == 0 & n2 > 0 - take group 2's stats
    mask_1_empty = (n1 == 0) & (n2 > 0)
    mu_combined[mask_1_empty] = mu2[mask_1_empty]
    sigma_combined[mask_1_empty] = sigma2[mask_1_empty]

    # Case: n2 == 0 & n1 > 0 - take group 1's stats
    mask_2_empty = (n2 == 0) & (n1 > 0)
    mu_combined[mask_2_empty] = mu1[mask_2_empty]
    sigma_combined[mask_2_empty] = sigma1[mask_2_empty]

    # Case: both > 0 - combine normally
    mask_both = (n1 > 0) & (n2 > 0)
    if np.any(mask_both):
        mu_c = (n1[mask_both] * mu1[mask_both] + n2[mask_both] * mu2[mask_both]) / total_n[mask_both]
        numerator = (
            (n1[mask_both] - 1) * sigma1[mask_both] ** 2
            + (n2[mask_both] - 1) * sigma2[mask_both] ** 2
            + n1[mask_both] * (mu1[mask_both] - mu_c) ** 2
            + n2[mask_both] * (mu2[mask_both] - mu_c) ** 2
        )
        denominator = total_n[mask_both] - 1
        mu_combined[mask_both] = mu_c
        sigma_combined[mask_both] = np.sqrt(numerator / denominator)

    return mu_combined, sigma_combined, total_n
