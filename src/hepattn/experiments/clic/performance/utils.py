import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .style_sheet import ALPHAS, COLORS, HISTTYPES, LABEL_LEN, LABELS, LINE_STYLES


def update_stylesheet(stylesheet):
    if stylesheet is None:
        stylesheet = {}
    global COLORS, LABELS, HISTTYPES, ALPHAS, LINE_STYLES, LABEL_LEN
    colors = stylesheet.get("COLORS", COLORS)
    labels = stylesheet.get("LABELS", LABELS)
    histtypes = stylesheet.get("HISTTYPES", HISTTYPES)
    alphas = stylesheet.get("ALPHAS", ALPHAS)
    line_styles = stylesheet.get("LINE_STYLES", LINE_STYLES)
    label_len = stylesheet.get("LABEL_LEN", LABEL_LEN)

    return colors, labels, histtypes, alphas, line_styles, label_len


def deltar(eta1, phi1, eta2, phi2):
    d_eta = eta1 - eta2
    phi1, phi2 = (phi1 + np.pi) % (2 * np.pi) - np.pi, (phi2 + np.pi) % (2 * np.pi) - np.pi
    d_phi = np.minimum(np.abs(phi1 - phi2), 2 * np.pi - np.abs(phi1 - phi2))
    return np.sqrt(d_eta**2 + d_phi**2)


def get_invariant_mass(jets, option="two-jet"):
    if option == "two-jet":
        m = (jets[0].fj_jet + jets[1].fj_jet).m() if len(jets) >= 2 else -1
    elif option == "one-jet":
        m = jets[0].m() if len(jets) >= 1 else -1
    return m


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-(((x - mean) / stddev) ** 2) / 2)


def custom_hist_v1(ax, vals, label_length=-1, metrics="mean std iqr", f=None, **hist_kwargs):  # noqa: ARG001
    if label_length != -1:
        hist_kwargs["label"] = hist_kwargs["label"].ljust(label_length)

    hist_kwargs["label"] += f" (M={np.nanmedian(vals):+.3f},".replace("+", " ")
    iqr = np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)
    hist_kwargs["label"] += f" IQR={iqr:.3f}"
    if f is not None:
        hist_kwargs["label"] += f", $f$={f:.3f})"

    ax.hist(vals, **hist_kwargs)


def custom_hist_v2(ax, vals, label_length=-1, metrics="mean std iqr", f=None, **hist_kwargs):  # noqa: ARG001
    hist_kwargs["label"] += f"\n(M={np.nanmedian(vals):+.3f},".replace("+", " ")
    iqr = np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)
    hist_kwargs["label"] += f" IQR={iqr:.3f}"
    if f is not None:
        hist_kwargs["label"] += f", $f$={f:.3f})"

    ax.hist(vals, **hist_kwargs)


def delta_r(eta1, eta2, phi1, phi2):
    dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
    deta = eta1 - eta2
    return np.sqrt(deta**2 + dphi**2)


def format_number(number):
    if number >= 1e9:
        return f"{number / 1e9:.1f}b"
    if number >= 1e6:
        return f"{number / 1e6:.1f}m"
    if number >= 1e3:
        return f"{number / 1e3:.1f}k"
    return str(number)


def get_cmap(cmap_type="lin_seg"):
    if cmap_type == "lin_seg":
        return LinearSegmentedColormap.from_list("custom_cmap", ["cornflowerblue", "red"])
    raise ValueError(f"Unknown cmap type: {cmap_type}")
