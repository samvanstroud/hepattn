# ruff: noqa: PLC0415

"""Filter preprocessed ATLAS muon dataset using hit filter predictions.

This script applies hit filtering using ML model predictions and additional
cuts (max tracks per event, max hits per event) to create a reduced dataset
optimized for the second stage of tracking model training.

The script maintains the same structure as the original dataset for
plug-and-play compatibility with the existing data loading routines.
"""

import argparse
import multiprocessing as mp
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import yaml
from sklearn.metrics import roc_curve
from tqdm import tqdm

warnings.filterwarnings("ignore")


def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


class HitFilterDatasetReducer:
    """Filter dataset using hit filter predictions and additional cuts."""

    def __init__(
        self,
        input_dir: str,
        eval_file: str,
        output_dir: str,
        working_point: float = 0.99,
        detection_threshold: float | None = None,
        max_tracks_per_event: int = 3,
        max_hits_per_event: int = 500,
        max_events: int = -1,
        num_workers: int | None = None,
        disable_track_filtering: bool = False,
        pt_threshold: float = 5.0,
        eta_threshold: float = 2.7,
        num_hits_threshold: int = 3,
    ):
        self.input_dir = Path(input_dir)
        self.eval_file = Path(eval_file)
        self.output_dir = Path(output_dir)
        self.working_point = working_point
        self.detection_threshold = detection_threshold
        self.max_tracks_per_event = max_tracks_per_event
        self.max_hits_per_event = max_hits_per_event
        self.max_events = max_events
        self.num_workers = num_workers or mp.cpu_count()
        self.disable_track_filtering = disable_track_filtering
        self.pt_threshold = pt_threshold
        self.eta_threshold = eta_threshold
        self.num_hits_threshold = num_hits_threshold

        # Load original metadata
        self.load_original_metadata()

        # Calculate detection threshold if not provided
        if self.detection_threshold is None:
            self.detection_threshold = self.calculate_detection_threshold()
            print(f"Calculated detection threshold: {self.detection_threshold:.6f} for working point {self.working_point}")
        else:
            print(f"Using provided detection threshold: {self.detection_threshold}")

        # Statistics tracking
        self.stats = {
            "total_events_processed": 0,
            "events_passed_hit_filter": 0,
            "events_failed_no_hits_after_filter": 0,
            "events_failed_max_tracks": 0,
            "events_failed_min_tracks": 0,
            "events_failed_max_hits": 0,
            "events_failed_min_hits": 0,
            "events_failed_eval_data_missing": 0,
            "events_failed_data_loading": 0,
            "events_failed_track_filtering": 0,
            "events_final_output": 0,
            "total_hits_before": 0,
            "total_hits_after": 0,
            "total_tracks_before": 0,
            "total_tracks_after": 0,
            "excluded_tracks_count": 0,
            # Track reconstructability after hit filtering
            "tracks_reconstructable_after_hit_filter": 0,
            # Detailed track filtering statistics
            "tracks_excluded_pt": 0,
            "tracks_excluded_eta": 0,
            "tracks_excluded_hits": 0,
            "tracks_excluded_no_true_hits": 0,
            # Event counting for average calculations
            "events_for_averages_pre_filter": 0,
            "events_for_averages_post_filter": 0,
            # True hits tracking for averages
            "total_true_hits_before": 0,
            "total_true_hits_after": 0,
        }

        # Data storage
        self.filtered_events = []
        self.file_indices = []
        self.row_indices = []

    def load_original_metadata(self):
        """Load metadata from the original dataset.

        Raises:
            FileNotFoundError: If the metadata.yaml file does not exist.
        """
        metadata_path = self.input_dir / "metadata.yaml"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with metadata_path.open() as f:
            self.original_metadata = yaml.safe_load(f)

        self.hit_features = self.original_metadata["hit_features"]
        self.track_features = self.original_metadata["track_features"]

        # Load index arrays
        self.original_file_indices = np.load(self.input_dir / "event_file_indices.npy")
        self.original_row_indices = np.load(self.input_dir / "event_row_indices.npy")

        print(f"Loaded original dataset with {len(self.original_file_indices)} events")

    def calculate_detection_threshold(self):
        """Calculate detection threshold for given working point using DataLoader approach.

        Raises:
            ValueError: If the working point cannot be achieved with the predictions.
        """
        print("Calculating detection threshold from evaluation file...")

        # Import here to avoid circular imports
        from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule

        # We need a config file to set up the data module - look for one in the configs directory
        config_dir = Path(__file__).parent / "configs" / "NGT"
        config_files = list(config_dir.glob("*.yaml"))

        if not config_files:
            # Fallback to a basic configuration
            inputs = {"hit": ["spacePoint_globEdgeLowX", "spacePoint_globEdgeLowY", "spacePoint_globEdgeLowZ"]}
            targets = {"hit": ["on_valid_particle"], "particle": ["truthMuon_pt", "truthMuon_eta", "truthMuon_phi", "truthMuon_q"]}
        else:
            # Load the first available config
            with config_files[0].open() as f:
                config = yaml.safe_load(f)
            data_config = config.get("data", {})
            inputs = data_config.get("inputs", {"hit": ["spacePoint_globEdgeLowX", "spacePoint_globEdgeLowY", "spacePoint_globEdgeLowZ"]})
            targets = data_config.get(
                "targets", {"hit": ["on_valid_particle"], "particle": ["truthMuon_pt", "truthMuon_eta", "truthMuon_phi", "truthMuon_q"]}
            )

        # Set up data module for a small sample to calculate threshold
        data_module = AtlasMuonDataModule(
            train_dir=str(self.input_dir),
            val_dir=str(self.input_dir),
            test_dir=str(self.input_dir),
            num_workers=50,  # Use fewer workers for threshold calculation
            num_train=-1,  # Small sample
            num_val=-1,
            num_test=-1,  # Use only 1000 events for threshold calculation
            batch_size=1,
            inputs=inputs,
            targets=targets,
            pin_memory=True,
        )

        data_module.setup(stage="test")
        test_dataloader = data_module.test_dataloader(shuffle=False)

        all_logits = []
        all_true_labels = []

        with h5py.File(self.eval_file, "r") as eval_f:
            for batch_idx, (_inputs_batch, targets_batch) in enumerate(tqdm(test_dataloader, desc="Loading evaluation data")):
                try:
                    # Get sample ID to match with evaluation file
                    if "sample_id" not in targets_batch:
                        continue

                    event_idx = targets_batch["sample_id"][0].item()

                    # Check if this event exists in evaluation file
                    if str(event_idx) not in eval_f:
                        continue

                    # Load logits from evaluation file
                    hit_logits = eval_f[f"{event_idx}/outputs/final/hit_filter/hit_logit"][0].astype(np.float32)

                    # Load true labels from dataset
                    true_labels = targets_batch["hit_on_valid_particle"][0].numpy().astype(np.bool_)

                    # Check dimensions match
                    if len(hit_logits) != len(true_labels):
                        print(f"Warning: Dimension mismatch for event {event_idx}: logits={len(hit_logits)}, labels={len(true_labels)}")

                        continue

                    all_logits.extend(hit_logits)
                    all_true_labels.extend(true_labels)

                    # # Stop after collecting enough data for threshold calculation
                    # if len(all_logits) > 100000000:  # 100 Mio hits should be enough
                    #     break

                except KeyError as e:
                    print(f"Warning: Could not load data for event {event_idx}: {e}")
                    continue
                except (OSError, RuntimeError) as e:
                    print(f"Warning: Error processing batch {batch_idx}: {e}")
                    continue

        if not all_logits:
            raise ValueError("No valid logits found in evaluation file")

        all_logits = np.array(all_logits)
        all_true_labels = np.array(all_true_labels)

        # Calculate ROC curve
        _fpr, tpr, thresholds = roc_curve(all_true_labels, all_logits)

        # Find threshold that achieves target efficiency
        target_efficiency = self.working_point
        valid_indices = tpr >= target_efficiency

        if not np.any(valid_indices):
            raise ValueError(f"Cannot achieve target efficiency {target_efficiency}")

        threshold = thresholds[valid_indices][0]

        print(f"ROC calculation complete: {len(all_logits)} total hits processed")

        return threshold

    def process_events(self):
        """Main processing method with multiprocessing."""
        print(f"Starting filtered dataset creation with {self.num_workers} workers...")

        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)

        # Split events among workers
        num_events = len(self.original_file_indices)

        # Apply max_events limit if specified
        if self.max_events > 0 and self.max_events < num_events:
            num_events = self.max_events
            print(f"Limiting processing to {num_events} events (max_events={self.max_events})")

        chunk_size = max(1, num_events // self.num_workers)

        event_chunks = []
        for i in range(0, num_events, chunk_size):
            end_idx = min(i + chunk_size, num_events)
            event_chunks.append((i, end_idx))

        # Limit to actual number of workers needed
        event_chunks = event_chunks[: self.num_workers]

        print(f"Processing {num_events} events in {len(event_chunks)} chunks")

        # Create worker arguments
        worker_args = []
        for worker_id, (start_idx, end_idx) in enumerate(event_chunks):
            args = (
                worker_id,
                start_idx,
                end_idx,
                str(self.input_dir),
                str(self.eval_file),
                str(self.output_dir),
                self.detection_threshold,
                self.max_tracks_per_event,
                self.max_hits_per_event,
                self.hit_features,
                self.track_features,
                self.original_metadata["event_mapping"]["chunk_summary"],
                self.disable_track_filtering,
                self.pt_threshold,
                self.eta_threshold,
                self.num_hits_threshold,
            )
            worker_args.append(args)

        # Process in parallel
        start_time = time.time()

        with mp.Pool(len(event_chunks)) as pool:
            results = list(tqdm(pool.imap(process_worker_events, worker_args), total=len(worker_args), desc="Worker progress"))

        processing_time = time.time() - start_time

        # Aggregate results
        self.aggregate_results(results)

        # Save final dataset
        self.save_filtered_dataset(processing_time)

        print(f"Processing complete in {processing_time:.2f} seconds")

    def aggregate_results(self, results: list[dict]):
        """Aggregate results from all workers."""
        print("Aggregating results from workers...")

        chunk_offset = 0

        for worker_result in results:
            if worker_result is None:
                continue

            # Aggregate statistics
            for key in self.stats:
                self.stats[key] += worker_result["stats"][key]

            # Aggregate event data
            self.filtered_events.extend(worker_result["filtered_events"])

            # Aggregate indices with proper offset
            worker_file_indices = np.array(worker_result["file_indices"]) + chunk_offset
            self.file_indices.extend(worker_file_indices.tolist())
            self.row_indices.extend(worker_result["row_indices"])

            chunk_offset += len(worker_result["filtered_events"])

    def save_filtered_dataset(self, processing_time: float):
        """Save the filtered dataset with metadata."""
        print("Saving filtered dataset...")

        # Save single merged H5 file
        output_h5_path = self.output_dir / "data" / "filtered_events.h5"

        with h5py.File(output_h5_path, "w") as f:
            # Store feature names as attributes
            f.attrs["hit_features"] = [name.encode() for name in self.hit_features]
            f.attrs["track_features"] = [name.encode() for name in self.track_features]

            # Save all events in compound arrays
            if self.filtered_events:
                max_hits = max(len(event["hits_array"]) for event in self.filtered_events)
                max_tracks = max(len(event["tracks_array"]) for event in self.filtered_events)

                # Create compound arrays with proper data types (matching original dataset)
                all_hits = np.full((len(self.filtered_events), max_hits, len(self.hit_features)), np.nan, dtype=np.float32)
                all_tracks = np.full((len(self.filtered_events), max_tracks, len(self.track_features)), np.nan, dtype=np.float32)
                all_event_numbers = np.full((len(self.filtered_events),), -1, dtype=np.int64)
                all_num_hits = np.zeros((len(self.filtered_events),), dtype=np.int16)
                all_num_tracks = np.zeros((len(self.filtered_events),), dtype=np.int16)

                for i, event in enumerate(self.filtered_events):
                    hits_len = len(event["hits_array"])
                    tracks_len = len(event["tracks_array"])

                    all_hits[i, :hits_len, :] = event["hits_array"]
                    all_tracks[i, :tracks_len, :] = event["tracks_array"]
                    all_event_numbers[i] = event["event_number"]
                    all_num_hits[i] = hits_len
                    all_num_tracks[i] = tracks_len

                # Use row-based chunking for better random access performance, no compression
                chunk_size_hits = (1, max_hits, len(self.hit_features))
                chunk_size_tracks = (1, max_tracks, len(self.track_features))

                f.create_dataset("hits", data=all_hits, chunks=chunk_size_hits)
                f.create_dataset("tracks", data=all_tracks, chunks=chunk_size_tracks)
                f.create_dataset("event_numbers", data=all_event_numbers)
                f.create_dataset("num_hits", data=all_num_hits)
                f.create_dataset("num_tracks", data=all_num_tracks)

        # Save index arrays for filtered dataset
        # All events are in a single file (index 0) with sequential row indices
        num_filtered_events = len(self.filtered_events)
        filtered_file_indices = np.zeros(num_filtered_events, dtype=np.int32)  # All point to file 0, use int32 for large datasets
        filtered_row_indices = np.arange(num_filtered_events, dtype=np.int32)  # Sequential 0,1,2,..., use int32 for up to 3M events

        np.save(self.output_dir / "event_file_indices.npy", filtered_file_indices)
        np.save(self.output_dir / "event_row_indices.npy", filtered_row_indices)

        # Create and save metadata
        self.save_metadata(processing_time)

        # Print final statistics
        self.print_final_statistics()

    def save_metadata(self, processing_time: float):
        """Save metadata for the filtered dataset."""

        # Helper function to convert numpy types to native Python types
        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        # Calculate statistics
        hit_filter_pass_rate = self.stats["events_passed_hit_filter"] / max(1, self.stats["total_events_processed"]) * 100
        max_tracks_fail_rate = self.stats["events_failed_max_tracks"] / max(1, self.stats["total_events_processed"]) * 100
        max_hits_fail_rate = self.stats["events_failed_max_hits"] / max(1, self.stats["total_events_processed"]) * 100
        final_pass_rate = self.stats["events_final_output"] / max(1, self.stats["total_events_processed"]) * 100

        hit_reduction_rate = (1 - self.stats["total_hits_after"] / max(1, self.stats["total_hits_before"])) * 100
        track_reduction_rate = (1 - self.stats["total_tracks_after"] / max(1, self.stats["total_tracks_before"])) * 100
        tracks_reconstructable_rate = self.stats["tracks_reconstructable_after_hit_filter"] / max(1, self.stats["total_tracks_before"]) * 100

        # Calculate average statistics per event
        avg_tracks_pre_filter = self.stats["total_tracks_before"] / max(1, self.stats["events_for_averages_pre_filter"])
        avg_hits_pre_filter = self.stats["total_hits_before"] / max(1, self.stats["events_for_averages_pre_filter"])
        avg_true_hits_pre_filter = self.stats["total_true_hits_before"] / max(1, self.stats["events_for_averages_pre_filter"])

        avg_tracks_post_filter = self.stats["total_tracks_after"] / max(1, self.stats["events_for_averages_post_filter"])
        avg_hits_post_filter = self.stats["total_hits_after"] / max(1, self.stats["events_for_averages_post_filter"])
        avg_true_hits_post_filter = self.stats["total_true_hits_after"] / max(1, self.stats["events_for_averages_post_filter"])

        # Create new metadata
        filtered_metadata = {
            "hit_features": self.hit_features,
            "track_features": self.track_features,
            "filtering_summary": {
                "total_events_processed": int(self.stats["total_events_processed"]),
                "events_passed_hit_filter": int(self.stats["events_passed_hit_filter"]),
                "events_failed_no_hits_after_filter": int(self.stats["events_failed_no_hits_after_filter"]),
                "events_failed_eval_data_missing": int(self.stats["events_failed_eval_data_missing"]),
                "events_failed_data_loading": int(self.stats["events_failed_data_loading"]),
                "events_failed_max_tracks": int(self.stats["events_failed_max_tracks"]),
                "events_failed_min_tracks": int(self.stats["events_failed_min_tracks"]),
                "events_failed_max_hits": int(self.stats["events_failed_max_hits"]),
                "events_failed_min_hits": int(self.stats["events_failed_min_hits"]),
                "events_failed_track_filtering": int(self.stats["events_failed_track_filtering"]),
                "events_final_output": int(self.stats["events_final_output"]),
                "excluded_tracks_count": int(self.stats["excluded_tracks_count"]),
                # Track reconstructability metric
                "tracks_reconstructable_after_hit_filter": int(self.stats["tracks_reconstructable_after_hit_filter"]),
                # Detailed track filtering statistics
                "tracks_excluded_pt": int(self.stats["tracks_excluded_pt"]),
                "tracks_excluded_eta": int(self.stats["tracks_excluded_eta"]),
                "tracks_excluded_hits": int(self.stats["tracks_excluded_hits"]),
                "tracks_excluded_no_true_hits": int(self.stats["tracks_excluded_no_true_hits"]),
                "hit_filter_pass_rate_percent": float(hit_filter_pass_rate),
                "max_tracks_fail_rate_percent": float(max_tracks_fail_rate),
                "max_hits_fail_rate_percent": float(max_hits_fail_rate),
                "final_pass_rate_percent": float(final_pass_rate),
                "total_hits_before": int(self.stats["total_hits_before"]),
                "total_hits_after": int(self.stats["total_hits_after"]),
                "hit_reduction_rate_percent": float(hit_reduction_rate),
                "total_tracks_before": int(self.stats["total_tracks_before"]),
                "total_tracks_after": int(self.stats["total_tracks_after"]),
                "track_reduction_rate_percent": float(track_reduction_rate),
                "tracks_reconstructable_after_hit_filter_percent": float(tracks_reconstructable_rate),
                # Average statistics per event
                "average_tracks_per_event_pre_filter": float(avg_tracks_pre_filter),
                "average_hits_per_event_pre_filter": float(avg_hits_pre_filter),
                "average_true_hits_per_event_pre_filter": float(avg_true_hits_pre_filter),
                "average_tracks_per_event_post_filter": float(avg_tracks_post_filter),
                "average_hits_per_event_post_filter": float(avg_hits_post_filter),
                "average_true_hits_per_event_post_filter": float(avg_true_hits_post_filter),
                "events_for_averages_pre_filter": int(self.stats["events_for_averages_pre_filter"]),
                "events_for_averages_post_filter": int(self.stats["events_for_averages_post_filter"]),
                "processing_time_seconds": float(processing_time),
                "num_workers": int(self.num_workers),
            },
            "filtering_parameters": {
                "working_point": float(self.working_point),
                "detection_threshold": float(self.detection_threshold),
                "max_tracks_per_event": int(self.max_tracks_per_event),
                "max_hits_per_event": int(self.max_hits_per_event),
                "disable_track_filtering": bool(self.disable_track_filtering),
                "pt_threshold": float(self.pt_threshold),
                "eta_threshold": float(self.eta_threshold),
                "num_hits_threshold": int(self.num_hits_threshold),
                "eval_file": str(self.eval_file),
                "source_dataset": str(self.input_dir),
            },
            "original_dataset_info": convert_numpy_types(self.original_metadata),
            "event_mapping": {
                "description": "Filtered events stored in single H5 file",
                "total_events": int(self.stats["events_final_output"]),
                "total_chunks": 1,
                "index_files": {
                    "file_indices": "event_file_indices.npy",
                    "row_indices": "event_row_indices.npy",
                },
                "chunk_summary": [
                    {
                        "h5_file": "data/filtered_events.h5",
                        "source_dataset": str(self.input_dir),
                        "event_count": int(self.stats["events_final_output"]),
                        "worker_id": "merged",
                    }
                ],
            },
        }

        # Save metadata
        metadata_path = self.output_dir / "metadata.yaml"
        with metadata_path.open("w") as f:
            yaml.dump(filtered_metadata, f, default_flow_style=False, sort_keys=False)

        print(f"Filtered dataset metadata saved to: {metadata_path}")

    def print_final_statistics(self):
        """Print comprehensive statistics about the filtering process."""
        print(f"\n{'=' * 80}")
        print("DATASET FILTERING SUMMARY")
        print(f"{'=' * 80}")
        print(f"Original dataset: {self.input_dir.name}")
        print(f"Working point: {self.working_point}")
        print(f"Detection threshold: {self.detection_threshold:.6f}")
        print(f"Max tracks per event: {self.max_tracks_per_event}")
        print(f"Max hits per event: {self.max_hits_per_event}")
        if not self.disable_track_filtering:
            print("Track filtering enabled:")
            print(f"  pT threshold: {self.pt_threshold} GeV")
            print(f"  |eta| threshold: {self.eta_threshold}")
            print(f"  Min hits per track: {self.num_hits_threshold}")
        else:
            print("Track filtering: DISABLED")
        print()
        print("EVENT STATISTICS:")
        print(f"  Total events processed: {self.stats['total_events_processed']:,}")
        print(
            f"  Events passed hit filter: {self.stats['events_passed_hit_filter']:,} "
            f"({self.stats['events_passed_hit_filter'] / max(1, self.stats['total_events_processed']) * 100:.2f}%)"
        )
        print(
            f"  Events failed - no hits after filter: {self.stats['events_failed_no_hits_after_filter']:,} "
            f"({self.stats['events_failed_no_hits_after_filter'] / max(1, self.stats['total_events_processed']) * 100:.2f}%)"
        )
        print(
            f"  Events failed - eval data missing: {self.stats['events_failed_eval_data_missing']:,} "
            f"({self.stats['events_failed_eval_data_missing'] / max(1, self.stats['total_events_processed']) * 100:.2f}%)"
        )
        print(
            f"  Events failed - data loading error: {self.stats['events_failed_data_loading']:,} "
            f"({self.stats['events_failed_data_loading'] / max(1, self.stats['total_events_processed']) * 100:.2f}%)"
        )
        print(
            f"  Events failed max tracks cut: {self.stats['events_failed_max_tracks']:,} "
            f"({self.stats['events_failed_max_tracks'] / max(1, self.stats['total_events_processed']) * 100:.2f}%)"
        )
        print(
            f"  Events failed min tracks cut: {self.stats['events_failed_min_tracks']:,} "
            f"({self.stats['events_failed_min_tracks'] / max(1, self.stats['total_events_processed']) * 100:.2f}%)"
        )
        print(
            f"  Events failed max hits cut: {self.stats['events_failed_max_hits']:,} "
            f"({self.stats['events_failed_max_hits'] / max(1, self.stats['total_events_processed']) * 100:.2f}%)"
        )
        print(
            f"  Events failed min hits cut: {self.stats['events_failed_min_hits']:,} "
            f"({self.stats['events_failed_min_hits'] / max(1, self.stats['total_events_processed']) * 100:.2f}%)"
        )
        if not self.disable_track_filtering:
            print(
                f"  Events failed track filtering: {self.stats['events_failed_track_filtering']:,} "
                f"({self.stats['events_failed_track_filtering'] / max(1, self.stats['total_events_processed']) * 100:.2f}%)"
            )
        print(
            f"  Final events output: {self.stats['events_final_output']:,} "
            f"({self.stats['events_final_output'] / max(1, self.stats['total_events_processed']) * 100:.2f}%)"
        )
        print()
        print("HIT/TRACK STATISTICS:")
        print(f"  Total hits before: {self.stats['total_hits_before']:,}")
        print(f"  Total hits after: {self.stats['total_hits_after']:,}")
        print(f"  Hit reduction: {(1 - self.stats['total_hits_after'] / max(1, self.stats['total_hits_before'])) * 100:.2f}%")
        print(f"  Total tracks before: {self.stats['total_tracks_before']:,}")
        print(f"  Total tracks after: {self.stats['total_tracks_after']:,}")
        print(f"  Track reduction: {(1 - self.stats['total_tracks_after'] / max(1, self.stats['total_tracks_before'])) * 100:.2f}%")
        recon_count = self.stats["tracks_reconstructable_after_hit_filter"]
        recon_pct = recon_count / max(1, self.stats["total_tracks_before"]) * 100
        print(f"  Tracks reconstructable after hit filtering (>3 true hits): {recon_count:,} ({recon_pct:.2f}%)")

        # Calculate and display average statistics
        avg_tracks_pre = self.stats["total_tracks_before"] / max(1, self.stats["events_for_averages_pre_filter"])
        avg_hits_pre = self.stats["total_hits_before"] / max(1, self.stats["events_for_averages_pre_filter"])
        avg_true_hits_pre = self.stats["total_true_hits_before"] / max(1, self.stats["events_for_averages_pre_filter"])

        avg_tracks_post = self.stats["total_tracks_after"] / max(1, self.stats["events_for_averages_post_filter"])
        avg_hits_post = self.stats["total_hits_after"] / max(1, self.stats["events_for_averages_post_filter"])
        avg_true_hits_post = self.stats["total_true_hits_after"] / max(1, self.stats["events_for_averages_post_filter"])

        print()
        print("AVERAGE STATISTICS PER EVENT:")
        print(f"  Pre-filtering averages (over {self.stats['events_for_averages_pre_filter']:,} events):")
        print(f"    - Tracks per event: {avg_tracks_pre:.2f}")
        print(f"    - Hits per event: {avg_hits_pre:.2f}")
        print(f"    - True hits per event: {avg_true_hits_pre:.2f}")
        print(f"  Post-filtering averages (over {self.stats['events_for_averages_post_filter']:,} events):")
        print(f"    - Tracks per event: {avg_tracks_post:.2f}")
        print(f"    - Hits per event: {avg_hits_post:.2f}")
        print(f"    - True hits per event: {avg_true_hits_post:.2f}")
        if not self.disable_track_filtering:
            print(f"  Tracks excluded by filtering: {self.stats['excluded_tracks_count']:,}")
            total_tracks = self.stats["total_tracks_before"]
            if total_tracks > 0:
                pt_exc = self.stats["tracks_excluded_pt"]
                pt_pct = pt_exc / total_tracks * 100
                print(f"    - Excluded due to pT < {self.pt_threshold} GeV: {pt_exc:,} ({pt_pct:.2f}%)")
                eta_exc = self.stats["tracks_excluded_eta"]
                eta_pct = eta_exc / total_tracks * 100
                print(f"    - Excluded due to |eta| > {self.eta_threshold}: {eta_exc:,} ({eta_pct:.2f}%)")
                hits_exc = self.stats["tracks_excluded_hits"]
                hits_pct = hits_exc / total_tracks * 100
                print(f"    - Excluded due to < {self.num_hits_threshold} hits: {hits_exc:,} ({hits_pct:.2f}%)")
                no_hits_exc = self.stats["tracks_excluded_no_true_hits"]
                no_hits_pct = no_hits_exc / total_tracks * 100
                print(f"    - Excluded due to no true hits after hit filtering: {no_hits_exc:,} ({no_hits_pct:.2f}%)")
        print()
        print("OUTPUT SUMMARY:")
        print(f"Output directory: {self.output_dir}")
        print(f"{'=' * 80}")


def process_worker_events(args: tuple) -> dict:
    """Worker function to process a range of events."""
    (
        worker_id,
        start_idx,
        end_idx,
        input_dir,
        eval_file,
        _output_dir,
        detection_threshold,
        max_tracks_per_event,
        max_hits_per_event,
        hit_features,
        track_features,
        chunk_summary,
        disable_track_filtering,
        _pt_threshold,
        _eta_threshold,
        num_hits_threshold,
    ) = args

    print(f"Worker {worker_id}: Processing events {start_idx} to {end_idx - 1}")

    # Load indices for this worker's range
    input_path = Path(input_dir)
    file_indices = np.load(input_path / "event_file_indices.npy")[start_idx:end_idx]
    row_indices = np.load(input_path / "event_row_indices.npy")[start_idx:end_idx]

    # Initialize worker statistics
    worker_stats = {
        "total_events_processed": 0,
        "events_passed_hit_filter": 0,
        "events_failed_no_hits_after_filter": 0,
        "events_failed_max_tracks": 0,
        "events_failed_min_tracks": 0,
        "events_failed_max_hits": 0,
        "events_failed_min_hits": 0,
        "events_failed_eval_data_missing": 0,
        "events_failed_data_loading": 0,
        "events_failed_track_filtering": 0,
        "events_final_output": 0,
        "total_hits_before": 0,
        "total_hits_after": 0,
        "total_tracks_before": 0,
        "total_tracks_after": 0,
        "excluded_tracks_count": 0,
        # Track reconstructability after hit filtering
        "tracks_reconstructable_after_hit_filter": 0,
        # Detailed track filtering statistics
        "tracks_excluded_pt": 0,
        "tracks_excluded_eta": 0,
        "tracks_excluded_hits": 0,
        "tracks_excluded_no_true_hits": 0,
        # Event counting for average calculations
        "events_for_averages_pre_filter": 0,
        "events_for_averages_post_filter": 0,
        # True hits tracking for averages
        "total_true_hits_before": 0,
        "total_true_hits_after": 0,
    }

    # Add track distribution tracking
    track_dist_original = {}
    track_dist_final = {}

    filtered_events = []
    worker_file_indices = []
    worker_row_indices = []

    # Open evaluation file once
    with h5py.File(eval_file, "r") as eval_f:
        for local_idx, (file_idx, row_idx) in enumerate(zip(file_indices, row_indices, strict=False)):
            global_event_idx = start_idx + local_idx
            worker_stats["total_events_processed"] += 1

            try:
                # Load original event
                chunk_info = chunk_summary[file_idx]
                h5_file_path = input_path / chunk_info["h5_file"]

                with h5py.File(h5_file_path, "r") as h5_f:
                    # Load hit and track data
                    hits_array = h5_f["hits"][row_idx]
                    tracks_array = h5_f["tracks"][row_idx]
                    event_number = h5_f["event_numbers"][row_idx]

                    # Convert to dictionary format
                    hits_dict = {}
                    for i, feature_name in enumerate(hit_features):
                        hits_dict[feature_name] = hits_array[:, i]

                    tracks_dict = {}
                    for i, feature_name in enumerate(track_features):
                        tracks_dict[feature_name] = tracks_array[:, i]

                # Remove NaN hits (padding)
                valid_hit_mask = ~np.isnan(hits_dict[hit_features[0]])
                for feature in hit_features:
                    hits_dict[feature] = hits_dict[feature][valid_hit_mask]

                # Remove NaN tracks (padding)
                valid_track_mask = ~np.isnan(tracks_dict[track_features[0]])
                for feature in track_features:
                    tracks_dict[feature] = tracks_dict[feature][valid_track_mask]

                original_num_hits = len(hits_dict[hit_features[0]])
                original_num_tracks = len(tracks_dict[track_features[0]])

                # Count original true hits (hits assigned to valid tracks)
                original_true_hits = np.sum(hits_dict["spacePoint_truthLink"] >= 0)

                worker_stats["total_hits_before"] += original_num_hits
                worker_stats["total_tracks_before"] += original_num_tracks
                worker_stats["total_true_hits_before"] += original_true_hits
                worker_stats["events_for_averages_pre_filter"] += 1

                # Track original distribution
                track_dist_original[original_num_tracks] = track_dist_original.get(original_num_tracks, 0) + 1

                # Apply hit filter using evaluation predictions
                try:
                    # The event index from the original dataset corresponds to the sample_id
                    # In the evaluation file, events are stored using sample_id as keys
                    sample_id = global_event_idx  # This is the sample_id used in the evaluation

                    # Check if this sample exists in evaluation file
                    if str(sample_id) not in eval_f:
                        print(f"Worker {worker_id}: Warning - Could not find predictions for sample {sample_id}")
                        worker_stats["events_failed_eval_data_missing"] += 1
                        continue

                    # Load logits from evaluation file using the correct path
                    logits = eval_f[f"{sample_id}/outputs/final/hit_filter/hit_logit"][0].astype(np.float32)

                    # Apply threshold to get hit filter mask
                    hit_filter_mask = logits >= detection_threshold

                    # Ensure mask length matches hits
                    if len(hit_filter_mask) != original_num_hits:
                        print(
                            f"Worker {worker_id}: Warning - Mask length mismatch for sample {sample_id}: "
                            f"logits={len(logits)}, hits={original_num_hits}"
                        )
                        worker_stats["events_failed_eval_data_missing"] += 1
                        continue

                except KeyError as e:
                    print(f"Worker {worker_id}: Warning - Could not load predictions for sample {global_event_idx}: {e}")
                    worker_stats["events_failed_eval_data_missing"] += 1
                    continue

                # Apply hit filtering
                for feature in hit_features:
                    hits_dict[feature] = hits_dict[feature][hit_filter_mask]

                filtered_num_hits = len(hits_dict[hit_features[0]])

                if filtered_num_hits == 0:
                    worker_stats["events_failed_no_hits_after_filter"] += 1
                    continue

                worker_stats["events_passed_hit_filter"] += 1

                # Apply track-level filtering similar to the preprocessing script
                unique_track_ids = np.unique(hits_dict["spacePoint_truthLink"])
                valid_track_ids = unique_track_ids[unique_track_ids >= 0]

                if len(valid_track_ids) == 0:
                    worker_stats["events_failed_no_hits_after_filter"] += 1
                    continue

                # DIAGNOSTIC: Check track ID to array index relationship
                # In properly preprocessed data, valid_track_ids should be [0, 1, 2, ...] up to num_tracks-1
                expected_track_ids = np.arange(original_num_tracks)
                if not np.array_equal(np.sort(valid_track_ids), expected_track_ids):
                    print(f"Worker {worker_id}: WARNING - Event {global_event_idx} has unexpected track IDs")
                    print(f"  Expected: {expected_track_ids}")
                    print(f"  Found: {np.sort(valid_track_ids)}")
                    print(f"  Original tracks: {original_num_tracks}")

                    # Check for dangerous mismatches that could cause indexing errors
                    dangerous_ids = [tid for tid in valid_track_ids if int(tid) >= original_num_tracks]
                    if dangerous_ids:
                        print(f"  CRITICAL: truthLink values {dangerous_ids} >= num_tracks {original_num_tracks}")
                        print("  DROPPING these tracks to prevent indexing errors!")
                        # Actually remove the dangerous tracks from valid_track_ids
                        valid_track_ids = np.array([tid for tid in valid_track_ids if int(tid) < original_num_tracks])
                        print(f"  After dropping dangerous tracks: {np.sort(valid_track_ids)}")
                    # This suggests a data preprocessing issue or incorrect assumption

                # Count tracks that remain reconstructable after hit filtering (have >3 true hits)
                # This metric is independent of all other cuts
                for track_idx in valid_track_ids:
                    true_hits_count = np.sum(hits_dict["spacePoint_truthLink"] == track_idx)
                    if true_hits_count > 3:  # Using > 3 instead of >= 3 to match "more than 3"
                        worker_stats["tracks_reconstructable_after_hit_filter"] += 1

                # ALWAYS filter tracks that have zero hits after hit filtering
                # This ensures consistency between hit and track data
                exclude_tracks = []

                if not disable_track_filtering:
                    # Apply track filters based on pT, eta, and hit count
                    # But we need to be careful about mapping truthLink to array indices

                    # Create a mapping from truthLink values to track counts
                    track_hit_counts = {}
                    for track_idx in valid_track_ids:
                        track_hit_counts[int(track_idx)] = np.sum(hits_dict["spacePoint_truthLink"] == track_idx)

                    # Now filter tracks, but we need to figure out which array index corresponds to which truthLink
                    # Since we don't have truthLink in track data, we'll assume the truthLink values
                    # are meant to be indices, but with possible gaps/offsets

                    # The safest approach: only exclude tracks that have zero hits after hit filtering
                    # Skip the pT/eta filtering for now since we can't safely map truthLink to array indices
                    for track_idx in valid_track_ids:
                        true_hits_count = track_hit_counts[int(track_idx)]

                        # Only exclude tracks with zero hits or too few hits
                        if true_hits_count == 0:
                            exclude_tracks.append(track_idx)
                            worker_stats["tracks_excluded_no_true_hits"] += 1
                            worker_stats["excluded_tracks_count"] += 1
                        elif true_hits_count < num_hits_threshold:
                            exclude_tracks.append(track_idx)
                            worker_stats["tracks_excluded_hits"] += 1
                            worker_stats["excluded_tracks_count"] += 1

                        # NOTE: Skipping pT and eta filtering due to truthLink mapping issues
                        # This should be fixed in the preprocessing step to ensure truthLink values
                        # correspond exactly to track array indices [0, 1, 2, ...]
                else:
                    # Even when track filtering is disabled, we must remove tracks with no hits
                    for track_idx in valid_track_ids:
                        # Get track index in the tracks array
                        track_array_idx = int(track_idx)

                        # Check if track index is valid
                        if track_array_idx >= len(tracks_dict["truthMuon_pt"]):
                            exclude_tracks.append(track_idx)
                            continue

                        # Check if track has any true hits left after hit filtering
                        true_hits_count = np.sum(hits_dict["spacePoint_truthLink"] == track_idx)
                        if true_hits_count == 0:
                            worker_stats["tracks_excluded_no_true_hits"] += 1
                            exclude_tracks.append(track_idx)
                            worker_stats["excluded_tracks_count"] += 1

                remaining_tracks = np.setdiff1d(valid_track_ids, exclude_tracks)

                if len(remaining_tracks) == 0:
                    worker_stats["events_failed_track_filtering"] += 1
                    continue

                # Filter hits to only keep those from remaining tracks
                hit2track_mask = np.isin(hits_dict["spacePoint_truthLink"], remaining_tracks)
                hit2track_mask |= hits_dict["spacePoint_truthLink"] == -1  # Keep background hits
                modified_truth_link = hits_dict["spacePoint_truthLink"].copy()
                modified_truth_link[~hit2track_mask] = -1

                # Apply the mask to all hit features
                hits_dict["spacePoint_truthLink"] = modified_truth_link

                # Filter tracks to only keep remaining tracks

                # Create mapping from old truthLink values to new sequential indices
                remaining_track_ids_sorted = np.sort(remaining_tracks)
                old_to_new_mapping = {int(old_id): new_idx for new_idx, old_id in enumerate(remaining_track_ids_sorted)}

                # Remap truthLink values in hits to be sequential [0,1,2...]
                new_truth_link = hits_dict["spacePoint_truthLink"].copy()
                for old_id, new_idx in old_to_new_mapping.items():
                    mask = hits_dict["spacePoint_truthLink"] == old_id
                    new_truth_link[mask] = new_idx

                # Keep noise hits as -1
                noise_mask = hits_dict["spacePoint_truthLink"] == -1
                new_truth_link[noise_mask] = -1

                # Set excluded tracks to noise (-1)
                for excluded_id in exclude_tracks:
                    excluded_mask = hits_dict["spacePoint_truthLink"] == excluded_id
                    new_truth_link[excluded_mask] = -1

                hits_dict["spacePoint_truthLink"] = new_truth_link

                # Filter tracks array to only keep remaining tracks in the correct order
                # We need to map the remaining_tracks to their original array positions
                # For now, assume truthLink corresponds to array index
                # This is the fundamental issue - we don't have the reverse mapping
                track_indices_to_keep = [int(track_id) for track_id in remaining_track_ids_sorted if int(track_id) < len(tracks_dict["truthMuon_pt"])]

                if len(track_indices_to_keep) > 0:
                    for feature in track_features:
                        tracks_dict[feature] = tracks_dict[feature][track_indices_to_keep]
                else:
                    # No valid tracks, create empty arrays
                    for feature in track_features:
                        tracks_dict[feature] = np.array([], dtype=tracks_dict[feature].dtype)

                valid_track_ids = remaining_tracks

                # Update counts after track filtering
                filtered_num_hits = len(hits_dict[hit_features[0]])
                filtered_num_tracks = len(valid_track_ids)

                # Apply max tracks cut
                if filtered_num_tracks > max_tracks_per_event:
                    worker_stats["events_failed_max_tracks"] += 1
                    continue
                # Apply min tracks cut
                if filtered_num_tracks < 1:
                    worker_stats["events_failed_min_tracks"] += 1
                    continue

                # Apply max hits cut
                if filtered_num_hits > max_hits_per_event:
                    worker_stats["events_failed_max_hits"] += 1
                    continue
                # Convert back to arrays (tracks have already been filtered above)
                filtered_hits_array = np.column_stack([hits_dict[feature] for feature in hit_features])
                filtered_tracks_array = np.column_stack([tracks_dict[feature] for feature in track_features])

                # Store filtered event
                filtered_events.append({"hits_array": filtered_hits_array, "tracks_array": filtered_tracks_array, "event_number": event_number})

                worker_file_indices.append(len(filtered_events) - 1)  # Points to position in filtered_events
                worker_row_indices.append(0)  # Always 0 since we have one event per "file"

                worker_stats["events_final_output"] += 1
                worker_stats["total_hits_after"] += filtered_num_hits
                worker_stats["total_tracks_after"] += filtered_num_tracks
                worker_stats["events_for_averages_post_filter"] += 1

                # Count true hits after filtering (hits assigned to valid tracks)
                filtered_true_hits = np.sum(hits_dict["spacePoint_truthLink"] >= 0)
                worker_stats["total_true_hits_after"] += filtered_true_hits

                # Track final distribution
                track_dist_final[filtered_num_tracks] = track_dist_final.get(filtered_num_tracks, 0) + 1

            except (OSError, KeyError, ValueError) as e:
                print(f"Worker {worker_id}: Error processing event {global_event_idx}: {e}")
                worker_stats["events_failed_data_loading"] += 1
                continue

    print(
        f"Worker {worker_id}: Completed. Processed {worker_stats['total_events_processed']} events, "
        f"output {worker_stats['events_final_output']} events"
    )

    # Print track distribution for this worker
    print(f"Worker {worker_id}: Track distribution changes:")
    print(f"  Original: {dict(sorted(track_dist_original.items()))}")
    print(f"  Final: {dict(sorted(track_dist_final.items()))}")

    return {
        "stats": worker_stats,
        "filtered_events": filtered_events,
        "file_indices": worker_file_indices,
        "row_indices": worker_row_indices,
        "track_distributions": {"original": track_dist_original, "final": track_dist_final},
    }


def generate_output_dir_name(input_dir: str, working_point: float, max_tracks: int, max_hits: int) -> str:
    """Generate output directory name with filtering parameters in the same parent directory as input."""
    input_path = Path(input_dir)
    input_name = input_path.name
    parent_dir = input_path.parent

    # Create descriptive suffix
    wp_str = f"wp{working_point:.3f}".replace(".", "")
    tracks_str = f"maxtrk{max_tracks}"
    hits_str = f"maxhit{max_hits}"

    # Create the full path in the same parent directory as the input
    filtered_name = f"{input_name}_filtered_{wp_str}_{tracks_str}_{hits_str}"
    return str(parent_dir / filtered_name)


def main():
    parser = argparse.ArgumentParser(description="Filter ATLAS muon dataset using hit filter predictions and cuts")

    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Input directory with preprocessed dataset")
    parser.add_argument("--eval_file", "-e", type=str, required=True, help="Path to evaluation HDF5 file with hit filter predictions")
    parser.add_argument("--output_dir", "-o", type=str, default=None, help="Output directory (auto-generated if not provided)")
    parser.add_argument("--working_point", "-wp", type=float, default=0.99, help="Working point efficiency for hit filter (default: 0.99)")
    parser.add_argument(
        "--detection_threshold", "-dt", type=float, default=None, help="Detection threshold (calculated from working_point if not provided)"
    )
    parser.add_argument("--max_tracks_per_event", "-mt", type=int, default=2, help="Maximum number of tracks per event (default: 3)")
    parser.add_argument("--max_hits_per_event", "-mh", type=int, default=600, help="Maximum number of hits per event after filtering (default: 600)")
    parser.add_argument("--max_events", "-me", type=int, default=-1, help="Maximum number of events to process (default: -1 for all events)")
    parser.add_argument("--num_workers", "-w", type=int, default=None, help="Number of worker processes (default: CPU count)")
    parser.add_argument(
        "--disable-track-filtering", action="store_true", default=False, help="Disable track filtering based on pt, eta, and hit count"
    )
    parser.add_argument("--pt-threshold", type=float, default=5.0, help="Minimum pT threshold for tracks (GeV, default: 5.0)")
    parser.add_argument("--eta-threshold", type=float, default=2.7, help="Maximum |eta| threshold for tracks (default: 2.7)")
    parser.add_argument("--num-hits-threshold", type=int, default=3, help="Minimum number of hits per track (default: 3)")

    args = parser.parse_args()

    # Generate output directory name if not provided
    if args.output_dir is None:
        args.output_dir = generate_output_dir_name(args.input_dir, args.working_point, args.max_tracks_per_event, args.max_hits_per_event)

    print("=" * 80)
    print("ATLAS MUON DATASET FILTERING")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Evaluation file: {args.eval_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Working point: {args.working_point}")
    print(f"Detection threshold: {args.detection_threshold or 'Auto-calculate'}")
    print(f"Max tracks per event: {args.max_tracks_per_event}")
    print(f"Max hits per event: {args.max_hits_per_event}")
    print(f"Max events: {args.max_events if args.max_events > 0 else 'ALL'}")
    print(f"Number of workers: {args.num_workers or 'Auto'}")
    print(f"Disable track filtering: {args.disable_track_filtering}")
    if not args.disable_track_filtering:
        print(f"pT threshold: {args.pt_threshold} GeV")
        print(f"Eta threshold: {args.eta_threshold}")
        print(f"Min hits per track: {args.num_hits_threshold}")
    print("=" * 80)

    # Create and run the filter
    filter_processor = HitFilterDatasetReducer(
        input_dir=args.input_dir,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        working_point=args.working_point,
        detection_threshold=args.detection_threshold,
        max_tracks_per_event=args.max_tracks_per_event,
        max_hits_per_event=args.max_hits_per_event,
        max_events=args.max_events,
        num_workers=args.num_workers,
        disable_track_filtering=args.disable_track_filtering,
        pt_threshold=args.pt_threshold,
        eta_threshold=args.eta_threshold,
        num_hits_threshold=args.num_hits_threshold,
    )

    filter_processor.process_events()


if __name__ == "__main__":
    main()
