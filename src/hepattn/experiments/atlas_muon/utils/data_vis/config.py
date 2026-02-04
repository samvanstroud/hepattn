"""Configuration and constants for ROOT file-based muon tracking data analysis.

This module provides default configuration values for data visualization
and analysis of ATLAS muon spectrometer tracking data stored in ROOT format.
"""

# =============================================================================
# File paths - These should be overridden via command-line arguments or config
# =============================================================================

# Dictionary mapping configuration keys to ROOT file paths
# Override via CLI arguments in production use
ROOT_FILE_PATHS: dict[str, str] = {}

# =============================================================================
# Tree/Dataset names
# =============================================================================

DEFAULT_TREE_NAME = "MuonHitDump"

# =============================================================================
# Branch name mappings
# =============================================================================

BRANCH_NAMES = {
    "event_number": "eventNumber",
    "truth_link": "spacePoint_truthLink",
    "pos_x": "spacePoint_globPosX",
    "pos_y": "spacePoint_globPosY",
    "pos_z": "spacePoint_globPosZ",
    "position_x": "spacePoint_PositionX",
}

# =============================================================================
# Visualization settings
# =============================================================================

TRACK_COLORS = [
    "#FF0000",  # Red
    "#00FF00",  # Green
    "#0000FF",  # Blue
    "#FF00FF",  # Magenta
    "#00FFFF",  # Cyan
    "#FFFF00",  # Yellow
    "#FF8000",  # Orange
    "#8000FF",  # Purple
]

PLOT_SETTINGS = {
    "figure_size": (18, 6),
    "background_color": "gray",
    "background_alpha": 0.3,
    "background_size": 10,
    "track_alpha": 0.9,
    "track_size": 30,
    "track_marker": "x",
    "track_linewidth": 2,
    "grid_alpha": 0.3,
    "dpi": 300,
}

# =============================================================================
# Analysis settings
# =============================================================================

ANALYSIS_SETTINGS = {
    "max_track_id": 1e6,  # Filter out invalid track IDs above this value
    "min_track_id": 0,  # Filter out background hits (typically -1)
    "histogram_bins": 50,
}

# =============================================================================
# Histogram settings for ROOT branch visualization
# =============================================================================

HISTOGRAM_SETTINGS = {
    "spacePoint_globEdgeHighX": {"bins": 100, "range": (-15000, 15000)},
    "spacePoint_globEdgeHighY": {"bins": 100, "range": (-15000, 15000)},
    "spacePoint_globEdgeHighZ": {"bins": 100, "range": (-25000, 25000)},
    "spacePoint_globEdgeLowX": {"bins": 100, "range": (-15000, 15000)},
    "spacePoint_globEdgeLowY": {"bins": 100, "range": (-15000, 15000)},
    "spacePoint_globEdgeLowZ": {"bins": 100, "range": (-25000, 25000)},
    "spacePoint_time": {"bins": 100, "range": (-1000, 20000)},
    "spacePoint_driftR": {"bins": 100, "range": (0, 15)},
    "spacePoint_readOutSide": {"bins": None, "range": (-2, 2)},
    "spacePoint_covXX": {"bins": 100, "range": (-1000, 10000000)},
    "spacePoint_covXY": {"bins": 100, "range": (-300000, 300000)},
    "spacePoint_covYX": {"bins": 100, "range": (-150000, 150000)},
    "spacePoint_covYY": {"bins": 100, "range": (0, 1300000)},
    "spacePoint_channel": {"bins": 100, "range": (0, 6000)},
    "spacePoint_layer": {"bins": None, "range": (0, 10)},
    "spacePoint_stationPhi": {"bins": None, "range": (0, 50)},
    "spacePoint_stationEta": {"bins": None, "range": (-10, 10)},
    "spacePoint_technology": {"bins": None, "range": (-1, 10)},
    "spacePoint_truthLink": {"bins": None, "range": (-1, 8)},
    "truthMuon_pt": {"bins": 100, "range": (0, 200)},
    "truthMuon_eta": {"bins": 100, "range": (-3, 3)},
    "truthMuon_phi": {"bins": 100, "range": (-3.2, 3.2)},
    "truthMuon_q": {"bins": None, "range": (-2, 2)},
}
