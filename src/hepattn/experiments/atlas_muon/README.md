# ATLAS Muon Spectrometer Tracking

This experiment implements ML-assisted tracking in the ATLAS Muon Spectrometer for the High Luminosity LHC (HL-LHC).

## Context

The HL-LHC, scheduled to begin operation after 2030, will increase proton-proton collisions per event from ~60 to up to 200. This rise in interaction density substantially elevates occupancy within the ATLAS Muon Spectrometer, necessitating more efficient and robust real-time data processing strategies.

This work explores:
1. **Hit Filtering**: ML models to reject background hits from hadronic punch-throughs and gamma radiation
2. **End-to-End Tracking**: Masked Transformer models (MaskFormer) for muon track reconstruction

For more details, see:
- **Talk**: [ML-Assisted Tracking in the ATLAS Muon Spectrometer](https://indico.cern.ch/event/1499357/contributions/6621917/) - Connecting The Dots 2025
- **Poster**: [ATL-DAQ-SLIDE-2025-693](https://cds.cern.ch/record/2950110/files/ATL-DAQ-SLIDE-2025-693.pdf)

## Setup

```shell
# Clone and enter the repository
git clone git@github.com:samvanstroud/hepattn.git
cd hepattn

# Optional: Use container for libc compatibility
apptainer pull pixi.sif docker://ghcr.io/prefix-dev/pixi:0.54.1-jammy-cuda-12.8.1
apptainer shell --nv pixi.sif

# Install dependencies
pixi install --locked

# Activate environment
pixi shell
```

## Data Pipeline

The training pipeline consists of three stages:

### Stage 1: ROOT → HDF5 (Hit Filtering Dataset)

Convert raw ROOT files from ATLAS simulation to HDF5 format for the hit filtering model:

```shell
python -m hepattn.experiments.atlas_muon.utils.data_prep_root_to_filter \
    --input-dir /path/to/root/files \
    --output-dir /path/to/hdf5/output \
    --expected-num-events-per-file 1000 \
    --num-workers 8 \
    --pt-threshold 1.0 \
    --eta-threshold 2.5
```

### Stage 2: Train Hit Filtering Model

Train the hit filtering model to classify signal vs background hits:

```shell
python run_filtering.py fit --config config/muon_filtering.yaml

# Test the model
python run_filtering.py test --config /path/to/checkpoint/config.yaml
```

### Stage 3: HDF5 → Filtered HDF5 (Tracking Dataset)

Apply the trained hit filter and create a reduced dataset for tracking:

```shell
python -m hepattn.experiments.atlas_muon.utils.data_prep_filter_to_tracking \
    --input-dir /path/to/filtering/hdf5 \
    --eval-file /path/to/hit_filter_predictions.h5 \
    --output-dir /path/to/tracking/hdf5 \
    --working-point 0.99 \
    --max-tracks-per-event 2 \
    --max-hits-per-event 600
```

### Stage 4: Train Tracking Model

Train the MaskFormer tracking model:

```shell
python run_tracking.py fit --config config/muon_tracking.yaml

# Test the model
python run_tracking.py test --config /path/to/checkpoint/config.yaml
```

## Evaluation Scripts

### Hit Filtering Evaluation

Evaluate hit filtering performance with ROC curves and efficiency plots:

```shell
python -m hepattn.experiments.atlas_muon.utils.evaluate_filtering_performance \
    --config /path/to/config.yaml \
    --checkpoint /path/to/checkpoint.ckpt \
    --output-dir ./evaluation_results
```

### Tracking Evaluation

Three evaluation scripts for different tracking tasks:

| Script | Purpose |
|--------|---------|
| `evalute_tracking_hit_track_assignment.py` | Hit-track assignment efficiency/purity |
| `evalute_tracking_track_validity.py` | Track validity classification ROC/efficiency |
| `evalute_tracking_track_regression.py` | Parameter regression residuals (η, φ, pT, q) |

```shell
# Example: Track validity evaluation
python -m hepattn.experiments.atlas_muon.utils.evalute_tracking_track_validity \
    --data-dir /path/to/test/data \
    --predictions /path/to/predictions.h5 \
    --output-dir ./tracking_evaluation
```

## Visualization Tools

Located in `utils/data_vis/`:

| Module | Description |
|--------|-------------|
| `root_analyzer.py` | Analyze ROOT files: hit distributions, track properties, branch histograms |
| `h5_analyzer.py` | Analyze HDF5 files via DataModule: feature histograms, signal/background separation |
| `track_visualizer.py` | 2D projections (X-Y, Z-Y, Z-X) of muon tracks from ROOT files |
| `track_visualizer_MDTGeometry.py` | Track visualization with MDT detector geometry overlay |
| `track_visualizer_h5_MDTGeometry.py` | HDF5-based track visualization with geometry |

```shell
# Example: Analyze HDF5 dataset statistics
python -m hepattn.experiments.atlas_muon.utils.plot_h5_stats \
    --data-dir /path/to/hdf5/data \
    --output-dir ./plots

# Example: Analyze ROOT file statistics
python -m hepattn.experiments.atlas_muon.utils.plot_root_stats \
    --root-file /path/to/file.root \
    --output-dir ./plots
```

## Configuration Files

| Config | Description |
|--------|-------------|
| `config/muon_filtering.yaml` | Hit filtering model configuration |
| `config/muon_tracking.yaml` | Tracking model configuration |

## Directory Structure

```
atlas_muon/
├── config/                     # YAML configuration files
├── data.py                     # AtlasMuonDataModule for data loading
├── run_filtering.py            # Hit filtering training entry point
├── run_tracking.py             # Tracking training entry point
└── utils/
    ├── data_prep_root_to_filter.py      # ROOT → HDF5 conversion
    ├── data_prep_filter_to_tracking.py  # Apply filter, create tracking dataset
    ├── evaluate_filtering_performance.py # Hit filter evaluation
    ├── evalute_tracking_*.py            # Tracking evaluation scripts
    ├── plot_h5_stats.py                 # HDF5 statistics plotting
    ├── plot_root_stats.py               # ROOT statistics plotting
    └── data_vis/                        # Visualization modules
        ├── h5_analyzer.py               # HDF5 data analysis
        ├── root_analyzer.py             # ROOT file analysis
        └── track_visualizer*.py         # Track visualization
```
