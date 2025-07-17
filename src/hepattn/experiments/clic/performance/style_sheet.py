FIG_W = 12
FIG_H_1ROW = 4.5
FIG_DPI = 200


LABELS = {
    # algorithms
    "truth": "Truth",
    "ppflow": "PPflow",
    "pandora": "Pandora",
    "topo": "TopoJet",
    "proxy": "Proxy",
    "hgpflow": "HGPflow",
    "mpflow": "MPFlow",
    "hgpflow_mini": "HGPflow mini",
    "hgpflow_target": "HGPflow target",
    "mlpf": "MLPF",
    # particles
    0: "ch. had",
    1: r"$e^\pm$",
    2: r"$\mu^\pm$",
    3: "nu. had",
    4: r"$\gamma$",
    5: "resid.",
}

COLORS = {
    # algorithms
    "truth": "black",
    "ppflow": "gray",
    "pandora": "gray",
    "topo": "orange",
    "proxy": "blue",
    "hgpflow": "firebrick",
    "mpflow": "purple",
    "hgpflow_mini": "blue",
    "hgpflow_target": "red",
    "mlpf": "teal",
    # particles
    0: "red",
    1: "orange",
    2: "magenta",
    3: "green",
    4: "blue",
    5: "gray",
}

HISTTYPES = {
    "truth": "step",
    "ppflow": "bar",
    "pandora": "bar",
    "topo": "bar",
    "proxy": "step",
    "hgpflow": "step",
    "mpflow": "step",
    "hgpflow_mini": "step",
    "hgpflow_target": "step",
    "mlpf": "step",
}

ALPHAS = {
    "truth": 1.0,
    "ppflow": 0.5,
    "pandora": 0.5,
    "topo": 0.5,
    "proxy": 1.0,
    "hgpflow": 1.0,
    "mpflow": 1.0,
    "hgpflow_mini": 1.0,
    "hgpflow_target": 1.0,
    "mlpf": 1.0,
}

LINE_STYLES = {
    "truth": "--",
    "ppflow": "-",
    "pandora": "-",
    "topo": "-",
    "proxy": "-",
    "hgpflow": "-",
    "mpflow": "-",
    "hgpflow_mini": "-",
    "hgpflow_target": "--",
    "mlpf": "-",
}

LABEL_LEN = {
    "truth": 10,
    "ppflow": 10,
    "pandora": 7,
    "topo": 10,
    "proxy": 10,
    "hgpflow": 7,
    "mpflow": 7,
    "hgpflow_target": 10,
    "mlpf": 9,
}
