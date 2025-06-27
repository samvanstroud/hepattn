import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .style_sheet import LABELS, COLORS, HISTTYPES, ALPHAS, LINE_STYLES, LABEL_LEN


def update_stylesheet(stylesheet):
    if stylesheet is None:
        stylesheet = {}
    global COLORS, LABELS, HISTTYPES, ALPHAS, LINE_STYLES, LABEL_LEN
    _COLORS = stylesheet.get("COLORS", COLORS)
    _LABELS = stylesheet.get("LABELS", LABELS)
    _HISTTYPES = stylesheet.get("HISTTYPES", HISTTYPES)
    _ALPHAS = stylesheet.get("ALPHAS", ALPHAS)
    _LINE_STYLES = stylesheet.get("LINE_STYLES", LINE_STYLES)
    _LABEL_LEN = stylesheet.get("LABEL_LEN", LABEL_LEN)

    return _COLORS, _LABELS, _HISTTYPES, _ALPHAS, _LINE_STYLES, _LABEL_LEN


def deltaR(eta1, phi1, eta2, phi2):
    d_eta = eta1 - eta2
    phi1, phi2 = (phi1 + np.pi) % (2 * np.pi) - np.pi, (phi2 + np.pi) % (2 * np.pi) - np.pi
    d_phi = np.minimum(np.abs(phi1 - phi2), 2 * np.pi - np.abs(phi1 - phi2))
    dR = np.sqrt(d_eta**2 + d_phi**2)
    return dR


def get_invariant_mass(jets, option="two-jet"):
    if option == "two-jet":
        if len(jets) >= 2:
            m = (jets[0].fj_jet + jets[1].fj_jet).m()
        else:
            m = -1
    elif option == "one-jet":
        if len(jets) >= 1:
            m = jets[0].m()
        else:
            m = -1
    return m


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-(((x - mean) / stddev) ** 2) / 2)


def custom_hist_v1(ax, vals, label_length=-1, metrics="mean std iqr", f=None, **hist_kwargs):
    if label_length != -1:
        hist_kwargs["label"] = hist_kwargs["label"].ljust(label_length)

    hist_kwargs["label"] += f" (M={np.nanmedian(vals):+.3f},".replace("+", " ")
    iqr = np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)
    hist_kwargs["label"] += f" IQR={iqr:.3f}"
    if f is not None:
        hist_kwargs["label"] += f", $f$={f:.3f})"

    ax.hist(vals, **hist_kwargs)


def custom_hist_v2(ax, vals, label_length=-1, metrics="mean std iqr", f=None, **hist_kwargs):
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
        return "{:.1f}b".format(number / 1e9)
    elif number >= 1e6:
        return "{:.1f}m".format(number / 1e6)
    elif number >= 1e3:
        return "{:.1f}k".format(number / 1e3)
    else:
        return str(number)


def get_cmap(type="lin_seg"):
    if type == "lin_seg":
        return LinearSegmentedColormap.from_list("custom_cmap", ["cornflowerblue", "red"])
    else:
        raise ValueError(f"Unknown cmap type: {type}")
