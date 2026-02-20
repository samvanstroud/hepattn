"""Configuration and constants for HDF5-based muon tracking data analysis.

This module provides default configuration values for data visualization
and analysis of ATLAS muon spectrometer tracking data stored in HDF5 format.
"""

# =============================================================================
# File paths - These should be overridden via command-line arguments or config
# =============================================================================

# Default HDF5 data directory (override via CLI or environment variable)
H5_FILEPATH: str | None = None

# Optional evaluation results file for filtered data visualization
HIT_EVAL_FILEPATH: str | None = None

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
# Histogram settings for feature visualization
# =============================================================================

HISTOGRAM_SETTINGS = {
    "hits": {
        "hit_spacePoint_globEdgeHighX": {
            "bins": 100,
            "range": (-15000, 15000),
            "scale_factor": 0.001,
        },
        "hit_spacePoint_globEdgeHighY": {
            "bins": 100,
            "range": (-15000, 15000),
            "scale_factor": 0.001,
        },
        "hit_spacePoint_globEdgeHighZ": {
            "bins": 100,
            "range": (-25000, 25000),
            "scale_factor": 0.001,
        },
        "hit_spacePoint_globEdgeLowX": {
            "bins": 100,
            "range": (-15000, 15000),
            "scale_factor": 0.001,
        },
        "hit_spacePoint_globEdgeLowY": {
            "bins": 100,
            "range": (-15000, 15000),
            "scale_factor": 0.001,
        },
        "hit_spacePoint_globEdgeLowZ": {
            "bins": 100,
            "range": (-25000, 25000),
            "scale_factor": 0.001,
        },
        "hit_spacePoint_time": {
            "bins": 100,
            "range": (-1000, 20000),
            "scale_factor": 0.00001,
        },
        "hit_spacePoint_driftR": {
            "bins": 100,
            "range": (0, 15),
            "scale_factor": 1.0,
        },
        "hit_spacePoint_readOutSide": {
            "bins": None,
            "range": (-2, 2),
            "scale_factor": 1.0,
        },
        "hit_spacePoint_covXX": {
            "bins": 100,
            "range": (-1000, 10000000),
            "scale_factor": 0.000001,
        },
        "hit_spacePoint_covXY": {
            "bins": 100,
            "range": (-300000, 300000),
            "scale_factor": 0.000001,
        },
        "hit_spacePoint_covYX": {
            "bins": 100,
            "range": (-150000, 150000),
            "scale_factor": 0.000001,
        },
        "hit_spacePoint_covYY": {
            "bins": 100,
            "range": (0, 1300000),
            "scale_factor": 0.000001,
        },
        "hit_spacePoint_channel": {
            "bins": 100,
            "range": (0, 6000),
            "scale_factor": 0.001,
        },
        "hit_spacePoint_layer": {
            "bins": None,
            "range": (0, 10),
            "scale_factor": 1.0,
        },
        "hit_spacePoint_stationPhi": {
            "bins": None,
            "range": (0, 50),
            "scale_factor": 1.0,
        },
        "hit_spacePoint_stationEta": {
            "bins": None,
            "range": (-10, 10),
            "scale_factor": 1.0,
        },
        "hit_spacePoint_technology": {
            "bins": None,
            "range": (-1, 10),
            "scale_factor": 1.0,
        },
        "hit_spacePoint_stationIndex": {
            "bins": None,
            "range": (-1000, 1000),
            "scale_factor": 0.1,
        },
        "hit_spacePoint_truthLink": {
            "bins": None,
            "range": (-1, 8),
            "scale_factor": 1.0,
        },
        "hit_r": {
            "bins": 100,
            "range": (0, 15000),
            "scale_factor": 0.001,
        },
        "hit_s": {
            "bins": 100,
            "range": (0, 30000),
            "scale_factor": 0.001,
        },
        "hit_theta": {
            "bins": 100,
            "range": (0, 3.2),
            "scale_factor": 1.0,
        },
        "hit_phi": {
            "bins": 100,
            "range": (-3.2, 3.2),
            "scale_factor": 1.0,
        },
    },
    "targets": {
        "particle_truthMuon_pt": {
            "bins": 100,
            "range": (0, 200),
            "scale_factor": 1.0,
        },
        "particle_truthMuon_eta": {
            "bins": 100,
            "range": (-3, 3),
            "scale_factor": 1.0,
        },
        "particle_truthMuon_phi": {
            "bins": 100,
            "range": (-3.2, 3.2),
            "scale_factor": 1.0,
        },
        "particle_truthMuon_q": {
            "bins": None,
            "range": (-2, 2),
            "scale_factor": 1.0,
        },
        "particle_truthMuon_qpt": {
            "bins": 100,
            "range": (-0.25, 0.25),
            "scale_factor": 1.0,
        },
    },
}
