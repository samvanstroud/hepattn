# ruff: noqa: N803, E501, A001

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import uproot
import yaml


def is_valid_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


class ParallelRootFilter:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        expected_num_events_per_file: int,
        max_events: int = -1,
        num_workers: int | None = None,
        no_NSW: bool = False,
        no_rpc: bool = False,
        disable_track_filtering: bool = False,
        pt_threshold: float = 1.0,
        eta_threshold: float = 2.5,
        num_hits_threshold: int = 5,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.expected_num_events_per_file = expected_num_events_per_file
        self.max_events = max_events
        self.num_workers = num_workers or mp.cpu_count()
        self.no_NSW = no_NSW
        self.no_rpc = no_rpc
        self.disable_track_filtering = disable_track_filtering
        self.pt_threshold = pt_threshold
        self.eta_threshold = eta_threshold
        self.num_hits_threshold = num_hits_threshold

        # Global counters (will be aggregated from workers)
        self.excluded_tracks_count = 0
        self.excluded_events_count = 0
        self.valid_events_count = 0
        self.valid_tracks_count = 0

        # Detailed filtering statistics
        self.tracks_excluded_pt = 0
        self.tracks_excluded_eta = 0
        self.tracks_excluded_hits = 0
        self.events_excluded_no_tracks_after_filtering = 0
        self.events_excluded_technology_filtering = 0
        self.events_excluded_no_hits_after_technology = 0

        self.event_mapping = []
        self.file_indices = []
        self.row_indices = []
        self.num_hits_per_event = []
        self.num_tracks_per_event = []

        self.files = self._get_files()

        self.hit_features = [
            "spacePoint_globEdgeHighX",
            "spacePoint_globEdgeHighY",
            "spacePoint_globEdgeHighZ",
            "spacePoint_globEdgeLowX",
            "spacePoint_globEdgeLowY",
            "spacePoint_globEdgeLowZ",
            "spacePoint_time",
            "spacePoint_driftR",
            "spacePoint_readOutSide",
            "spacePoint_covXX",
            "spacePoint_covXY",
            "spacePoint_covYX",
            "spacePoint_covYY",
            "spacePoint_channel",
            "spacePoint_layer",
            "spacePoint_stationPhi",
            "spacePoint_stationEta",
            "spacePoint_stationIndex",
            "spacePoint_technology",
            "spacePoint_truthLink",
        ]

        self.track_features = ["truthMuon_pt", "truthMuon_eta", "truthMuon_phi", "truthMuon_q"]

    def _get_files(self) -> list[Path]:
        dirpath = Path(self.input_dir)
        files = list(dirpath.glob("*.root"))
        if not files:
            raise FileNotFoundError(f"No ROOT files found in {dirpath}")

        valid_files = [f for f in files if is_valid_file(f)]
        if not valid_files:
            raise FileNotFoundError(f"No valid ROOT files found in {dirpath}")
        return sorted(valid_files)

    def _split_files(self) -> list[list[Path]]:
        """Split files into chunks for parallel processing."""
        files_per_worker = len(self.files) // self.num_workers
        remainder = len(self.files) % self.num_workers

        file_chunks = []
        start_idx = 0

        for i in range(self.num_workers):
            # Distribute remainder files among first workers
            chunk_size = files_per_worker + (1 if i < remainder else 0)
            end_idx = start_idx + chunk_size

            if start_idx < len(self.files):
                file_chunks.append(self.files[start_idx:end_idx])
            else:
                file_chunks.append([])  # Empty chunk for excess workers

            start_idx = end_idx
        # print("len(self.files)", len(self.files))
        # non_empty_chunks = sum(1 for chunk in file_chunks if chunk)
        # print(f"Number of non-empty chunks: {non_empty_chunks}")
        return file_chunks

    def process_events(self):
        """Main method to process events in parallel."""
        print(f"Starting parallel processing with {self.num_workers} workers...")
        print(f"Total files to process: {len(self.files)}")

        # Create output directory structure
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        (Path(self.output_dir) / "data").mkdir(exist_ok=True, parents=True)

        # Split files among workers
        file_chunks = self._split_files()

        # Print worker assignment
        for i, chunk in enumerate(file_chunks):
            print(f"Worker {i}: {len(chunk)} files")

        # Create worker arguments
        worker_args = []
        for worker_id, file_chunk in enumerate(file_chunks):
            if file_chunk:  # Only create args for workers with files
                args = (
                    worker_id,
                    file_chunk,
                    self.output_dir,
                    self.expected_num_events_per_file,
                    self.max_events,
                    self.hit_features,
                    self.track_features,
                    self.no_NSW,
                    self.no_rpc,
                    self.disable_track_filtering,
                    self.pt_threshold,
                    self.eta_threshold,
                    self.num_hits_threshold,
                )
                worker_args.append(args)

        # Process in parallel
        start_time = time.time()
        with mp.Pool(len(worker_args)) as pool:
            results = pool.map(process_worker_files, worker_args)

        processing_time = time.time() - start_time
        print(f"Parallel processing completed in {processing_time:.2f} seconds")

        # Aggregate results from all workers
        self._aggregate_results(results)

        self._save_final_metadata(processing_time)

    def _aggregate_results(self, results: list[dict]):
        """Aggregate results from all worker processes."""
        print("Aggregating results from workers...")

        total_chunk_offset = 0
        total_events_seen = 0  # Add this

        for worker_result in results:
            if worker_result is None:
                continue

            # Aggregate counters
            self.excluded_tracks_count += worker_result["excluded_tracks_count"]
            self.excluded_events_count += worker_result["excluded_events_count"]
            self.valid_events_count += worker_result["valid_events_count"]
            self.valid_tracks_count += worker_result["valid_tracks_count"]

            # Aggregate detailed filtering statistics
            self.tracks_excluded_pt += worker_result["tracks_excluded_pt"]
            self.tracks_excluded_eta += worker_result["tracks_excluded_eta"]
            self.tracks_excluded_hits += worker_result["tracks_excluded_hits"]
            self.events_excluded_no_tracks_after_filtering += worker_result["events_excluded_no_tracks_after_filtering"]
            self.events_excluded_technology_filtering += worker_result["events_excluded_technology_filtering"]
            self.events_excluded_no_hits_after_technology += worker_result["events_excluded_no_hits_after_technology"]

            # total_events_seen += worker_result.get('total_events_seen', 0)  # Add this
            total_events_seen += worker_result["total_events_seen"]  # Add this

            # Aggregate event mapping
            self.event_mapping.extend(worker_result["event_mapping"])

            # Aggregate indices with proper offsets
            worker_file_indices = np.array(worker_result["file_indices"]) + total_chunk_offset
            # print("Worker file indices:", worker_file_indices)
            self.file_indices.extend(worker_file_indices.tolist())
            # print("File indices:", self.file_indices)
            self.row_indices.extend(worker_result["row_indices"])
            self.num_hits_per_event.extend(worker_result["num_hits_per_event"])
            self.num_tracks_per_event.extend(worker_result["num_tracks_per_event"])

            total_chunk_offset += len(worker_result["event_mapping"])

        # Store total events seen
        self.total_events_seen = total_events_seen

    def _save_final_metadata(self, processing_time: float):
        """Save aggregated metadata and index arrays."""
        print("Saving final metadata...")

        # Save index arrays
        np.save(Path(self.output_dir) / "event_file_indices.npy", np.array(self.file_indices))
        np.save(Path(self.output_dir) / "event_row_indices.npy", np.array(self.row_indices))

        # Calculate summary statistics
        total_tracks = self.valid_tracks_count + self.excluded_tracks_count
        total_events = self.valid_events_count + self.excluded_events_count
        excluded_tracks_percent = (self.excluded_tracks_count / total_tracks * 100) if total_tracks > 0 else 0
        excluded_events_percent = (self.excluded_events_count / total_events * 100) if total_events > 0 else 0
        avg_tracks_per_event = (self.valid_tracks_count / self.valid_events_count) if self.valid_events_count > 0 else 0

        # Count number of workers that actually processed files
        num_active_workers = len(self.event_mapping)

        # Create metadata
        dataset_info = {
            "hit_features": self.hit_features,
            "track_features": self.track_features,
            "processing_summary": {
                "total_excluded_tracks": self.excluded_tracks_count,
                "total_tracks_processed": total_tracks,
                "excluded_tracks_percentage": excluded_tracks_percent,
                "tracks_excluded_pt": self.tracks_excluded_pt,
                "tracks_excluded_eta": self.tracks_excluded_eta,
                "tracks_excluded_hits": self.tracks_excluded_hits,
                "total_excluded_events": self.excluded_events_count,
                "total_events_processed": total_events,
                "total_events_seen": self.total_events_seen,
                "excluded_events_percentage": excluded_events_percent,
                "events_excluded_no_hits_after_technology": self.events_excluded_no_hits_after_technology,
                "events_excluded_no_tracks_after_filtering": self.events_excluded_no_tracks_after_filtering,
                "valid_events": self.valid_events_count,
                "valid_tracks": self.valid_tracks_count,
                "average_tracks_per_event": avg_tracks_per_event,
                "processing_time_seconds": processing_time,
                "num_workers": self.num_workers,
                "num_root_files": len(self.files),
                "processing_status": "Complete",
            },
            "processing_parameters": {
                "expected_number_of_events_per_file": self.expected_num_events_per_file,
                "max_events": self.max_events,
                "no_NSW": self.no_NSW,
                "no_rpc": self.no_rpc,
                "disable_track_filtering": self.disable_track_filtering,
                "pt_threshold": self.pt_threshold,
                "eta_threshold": self.eta_threshold,
                "num_hits_threshold": self.num_hits_threshold,
            },
            "processed_files": [str(file_path) for file_path in self.files],
            "event_mapping": {
                "description": "Event indices stored in separate numpy files for efficient access",
                "total_events": self.valid_events_count,
                "total_chunks": len(self.event_mapping),
                "index_files": {
                    "file_indices": "event_file_indices.npy",
                    "row_indices": "event_row_indices.npy",
                },
                "chunk_summary": [
                    {
                        "h5_file": chunk["h5_file"],
                        "source_root_file": chunk["source_root_file"],
                        "event_count": chunk["event_range"]["count"],
                        "worker_id": chunk["worker_id"],
                    }
                    for chunk in self.event_mapping
                ],
            },
        }

        # Save metadata
        dataset_info_file = Path(self.output_dir) / "metadata.yaml"
        with dataset_info_file.open("w") as f:
            yaml.dump(dataset_info, f, default_flow_style=False, sort_keys=False)

        # Print summary
        print(f"\n{'=' * 60}")
        print("PROCESSING SUMMARY")
        print(f"{'=' * 60}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Workers used: {num_active_workers}")
        print(f"Total events seen: {self.total_events_seen:,}")
        print()
        print("TRACK FILTERING STATISTICS:")
        if not self.disable_track_filtering:
            print(f"  Total excluded tracks: {self.excluded_tracks_count:,} out of {total_tracks:,} ({excluded_tracks_percent:.2f}%)")
            print(
                f"    - Excluded due to pT < {self.pt_threshold} GeV: {self.tracks_excluded_pt:,} ({self.tracks_excluded_pt / max(1, total_tracks) * 100:.2f}%)"
            )
            print(
                f"    - Excluded due to |eta| > {self.eta_threshold}: {self.tracks_excluded_eta:,} ({self.tracks_excluded_eta / max(1, total_tracks) * 100:.2f}%)"
            )
            print(
                f"    - Excluded due to < {self.num_hits_threshold} hits: {self.tracks_excluded_hits:,} ({self.tracks_excluded_hits / max(1, total_tracks) * 100:.2f}%)"
            )
        else:
            print("  Track filtering: DISABLED")
        print(f"  Valid tracks: {self.valid_tracks_count:,}")
        print()
        print("EVENT FILTERING STATISTICS:")
        print(f"  Total excluded events: {self.excluded_events_count:,} out of {total_events:,} ({excluded_events_percent:.2f}%)")
        print(
            f"    - Events with no hits after technology filtering: {self.events_excluded_no_hits_after_technology:,} ({self.events_excluded_no_hits_after_technology / max(1, total_events) * 100:.2f}%)"
        )
        if not self.disable_track_filtering:
            print(
                f"    - Events with no tracks after filtering: {self.events_excluded_no_tracks_after_filtering:,} ({self.events_excluded_no_tracks_after_filtering / max(1, total_events) * 100:.2f}%)"
            )
        print(f"  Valid events: {self.valid_events_count:,}")
        print(f"  Average tracks per event: {avg_tracks_per_event:.2f}")
        print()
        print("OUTPUT SUMMARY:")
        print(f"  Total chunks created: {len(self.event_mapping)}")
        print(f"  Dataset metadata saved to: {dataset_info_file}")
        print(f"{'=' * 60}")


def process_worker_files(args: tuple) -> dict:
    """Worker function to process a subset of files."""
    (
        worker_id,
        file_chunk,
        output_dir,
        expected_num_events_per_file,
        max_events,
        hit_features,
        track_features,
        no_NSW,
        no_rpc,
        disable_track_filtering,
        pt_threshold,
        eta_threshold,
        num_hits_threshold,
    ) = args

    if not file_chunk:
        return None

    print(f"Worker {worker_id}: Starting processing of {len(file_chunk)} files")
    sys.stdout.flush()  # Force immediate output to file

    # Initialize worker-specific counters and data structures
    excluded_tracks_count = 0
    excluded_events_count = 0
    valid_tracks_count = 0

    # Detailed filtering statistics
    tracks_excluded_pt = 0
    tracks_excluded_eta = 0
    tracks_excluded_hits = 0
    events_excluded_no_tracks_after_filtering = 0
    events_excluded_technology_filtering = 0
    events_excluded_no_hits_after_technology = 0

    event_mapping = []
    file_indices = []
    row_indices = []
    num_hits_per_event = []
    num_tracks_per_event = []
    total_valid_events = 0
    total_events_seen = 0  # Add this counter

    for root_file in file_chunk:
        # Initialize per-file variables
        hits_chunk = []
        tracks_chunk = []
        event_numbers_chunk = []

        try:
            with uproot.open(root_file) as rf:
                tree_keys = [key for key in rf if ";" in key]
                if not tree_keys:
                    print(f"Worker {worker_id}: No tree found in {root_file.name}")
                    continue

                tree = tree_keys[0].split(";")[0]
                num_events = rf[tree].num_entries
                chunk_size = 50

                for chunk_start in range(0, num_events, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, num_events)
                    print(f"print worker {worker_id} Loading chunk data.")
                    sys.stdout.flush()  # Force immediate output to file
                    # Load chunk data
                    hit_features_chunk = {}
                    for feature in hit_features:
                        hit_features_chunk[feature] = rf[tree][feature].array(entry_start=chunk_start, entry_stop=chunk_end, library="np")

                    track_features_chunk = {}
                    for feature in track_features:
                        track_features_chunk[feature] = rf[tree][feature].array(entry_start=chunk_start, entry_stop=chunk_end, library="np")

                    event_numbers_array = rf[tree]["eventNumber"].array(entry_start=chunk_start, entry_stop=chunk_end, library="np")

                    # Process each event in chunk
                    for event_idx_in_chunk in range(chunk_end - chunk_start):
                        total_events_seen += 1  # Count every event we see
                        print(f"Worker {worker_id}: Processing event {event_idx_in_chunk + 1}/{chunk_end - chunk_start}")
                        sys.stdout.flush()  # Force immediate output to file
                        # Check if we've reached the global limit (approximate)
                        if max_events > 0 and total_valid_events >= max_events:
                            print(f"Worker {worker_id}: Reached approximate max_events limit ({max_events})")
                            sys.stdout.flush()  # Force immediate output to file
                            # Count remaining events as excluded
                            (chunk_end - chunk_start) - event_idx_in_chunk
                            remaining_events_in_file = num_events - (chunk_start + event_idx_in_chunk)
                            excluded_events_count += remaining_events_in_file
                            total_events_seen += remaining_events_in_file - 1  # -1 because we already counted current event
                            break

                        # Get all tracks for this event (including -1 for unassigned hits)

                        # Apply technology filtering if requested
                        hits = {branch: hit_features_chunk[branch][event_idx_in_chunk].copy() for branch in hit_features}
                        technology_values = hits["spacePoint_technology"]
                        np.unique(hits["spacePoint_truthLink"])

                        # Create mask for hits to keep
                        keep_mask = np.ones(len(technology_values), dtype=bool)

                        if no_NSW:
                            # Remove STGC (4) and MM (5) hits
                            keep_mask &= ~np.isin(technology_values, [4, 5])

                        # TODO: integrate old filtering again check status of that file in GitHub

                        if no_rpc:
                            # Remove RPC (2) hits
                            keep_mask &= technology_values != 2

                        # Filter all hit features based on the keep mask
                        for branch in hit_features:
                            hits[branch] = hits[branch][keep_mask]

                        # Check if we have any hits left after filtering
                        if len(hits["spacePoint_time"]) == 0:
                            # Skip events with no hits after technology filtering
                            events_excluded_no_hits_after_technology += 1
                            excluded_events_count += 1
                            continue

                        # TODO: Filter out tracks below for the true tracks as well
                        unique_tracks = np.unique(hits["spacePoint_truthLink"])
                        valid_tracks = unique_tracks[unique_tracks != -1]

                        if len(valid_tracks) == 0:
                            excluded_events_count += 1
                            continue

                        if not disable_track_filtering:
                            # Apply track filters
                            exclude_tracks = []
                            for track_idx in valid_tracks:
                                exclude_reasons = []

                                # Check pT threshold
                                if track_features_chunk["truthMuon_pt"][event_idx_in_chunk][track_idx] < pt_threshold:
                                    exclude_reasons.append("pt")
                                    tracks_excluded_pt += 1
                                    exclude_tracks.append(track_idx)
                                    excluded_tracks_count += 1

                                # Check eta threshold
                                if abs(track_features_chunk["truthMuon_eta"][event_idx_in_chunk][track_idx]) > eta_threshold:
                                    exclude_reasons.append("eta")
                                    tracks_excluded_eta += 1

                                    exclude_tracks.append(track_idx)
                                    excluded_tracks_count += 1

                                # Check minimum hits threshold
                                if np.sum(hits["spacePoint_truthLink"] == track_idx) < num_hits_threshold:
                                    exclude_reasons.append("hits")
                                    tracks_excluded_hits += 1
                                    exclude_tracks.append(track_idx)
                                    excluded_tracks_count += 1

                            remaining_tracks = np.setdiff1d(valid_tracks, exclude_tracks)

                            if len(remaining_tracks) == 0:
                                events_excluded_no_tracks_after_filtering += 1
                                excluded_events_count += 1
                                continue

                            valid_tracks_count += len(remaining_tracks)
                            total_valid_events += 1  # Count this as a valid event

                            # Build event data
                            hit2track_mask = np.isin(hits["spacePoint_truthLink"], remaining_tracks)
                            modified_truth_link = hits["spacePoint_truthLink"].copy()
                            modified_truth_link[~hit2track_mask] = -1

                            # CRITICAL FIX: Renormalize truthLink values to sequential [0,1,2,...]
                            # to match track array indices
                            remaining_tracks_sorted = np.sort(remaining_tracks)
                            for new_idx, original_track_id in enumerate(remaining_tracks_sorted):
                                track_mask = hits["spacePoint_truthLink"] == original_track_id
                                modified_truth_link[track_mask] = new_idx

                            hits["spacePoint_truthLink"] = modified_truth_link

                            # Create track data - we need to reorder tracks to match the new truthLink indices
                            # The tracks should be ordered by the remaining_tracks_sorted order
                            all_tracks_for_event = {}
                            for branch in track_features:
                                all_tracks_for_event[branch] = track_features_chunk[branch][event_idx_in_chunk]

                            # Keep only tracks that correspond to remaining_tracks_sorted, in order
                            # FIXED: Use original track IDs directly as indices into the track array
                            track_data_ordered = {branch: [] for branch in track_features}
                            for original_track_id in remaining_tracks_sorted:
                                # Use original_track_id directly as index into the track array
                                # Add this track's data to our ordered list
                                for branch in track_features:
                                    track_data_ordered[branch].append(all_tracks_for_event[branch][original_track_id])

                            # Convert lists to numpy arrays
                            tracks = {}
                            for branch in track_features:
                                if len(track_data_ordered[branch]) > 0:
                                    tracks[branch] = np.array(track_data_ordered[branch])
                                else:
                                    tracks[branch] = np.array([], dtype=np.float32)

                        else:
                            # No track filtering - use all valid tracks
                            valid_tracks_count += len(valid_tracks)
                            total_valid_events += 1  # Count this as a valid event

                            valid_tracks_sorted = np.sort(valid_tracks)
                            modified_truth_link = hits["spacePoint_truthLink"].copy()

                            # Renormalize truthLink values to [0,1,2,...]
                            for new_idx, original_track_id in enumerate(valid_tracks_sorted):
                                track_mask = hits["spacePoint_truthLink"] == original_track_id
                                modified_truth_link[track_mask] = new_idx

                            hits["spacePoint_truthLink"] = modified_truth_link

                            # Create track data - reorder tracks to match valid_tracks_sorted
                            all_tracks_for_event = {}
                            for branch in track_features:
                                all_tracks_for_event[branch] = track_features_chunk[branch][event_idx_in_chunk]

                            # Keep only tracks that correspond to valid_tracks_sorted, in order

                            track_data_ordered = {branch: [] for branch in track_features}
                            for original_track_id in valid_tracks_sorted:
                                # Use original_track_id directly as index into the track array
                                # Add this track's data to our ordered list
                                for branch in track_features:
                                    track_data_ordered[branch].append(all_tracks_for_event[branch][original_track_id])

                            # Convert lists to numpy arrays
                            tracks = {}
                            for branch in track_features:
                                if len(track_data_ordered[branch]) > 0:
                                    tracks[branch] = np.array(track_data_ordered[branch])
                                else:
                                    tracks[branch] = np.array([], dtype=np.float32)

                        hits_chunk.append(hits)
                        tracks_chunk.append(tracks)
                        event_numbers_chunk.append(event_numbers_array[event_idx_in_chunk])

                    # Break out of chunk loop if we hit the limit
                    if max_events > 0 and total_valid_events >= max_events:
                        break

        except (OSError, ValueError, KeyError) as e:
            print(f"Worker {worker_id}: Error processing file {root_file}: {e}")
            sys.stdout.flush()  # Force immediate output to file
            excluded_events_count += expected_num_events_per_file  # Count this file as an excluded event
            continue

        # Save file data if we have any valid events
        if len(hits_chunk) > 0:
            file_valid_events = len(hits_chunk)
            print(f"Worker {worker_id}: Saving {file_valid_events} events from {root_file.name}")
            sys.stdout.flush()  # Force immediate output to file

            chunk_info = save_worker_chunk_to_hdf5(
                hits_chunk, tracks_chunk, event_numbers_chunk, output_dir, worker_id, root_file, file_valid_events, hit_features, track_features
            )

            # Update tracking data
            current_chunk_idx = len(event_mapping)
            for i in range(len(hits_chunk)):
                file_indices.append(current_chunk_idx)
                row_indices.append(i)
                num_hits_per_event.append(len(hits_chunk[i]["spacePoint_time"]))
                num_tracks_per_event.append(len(tracks_chunk[i]["truthMuon_pt"]))

            event_mapping.append(chunk_info)
        else:
            print(f"Worker {worker_id}: No valid events found in {root_file.name}")
            sys.stdout.flush()  # Force immediate output to file

        # Stop processing more files if we've reached the limit
        if max_events > 0 and total_valid_events >= max_events:
            print(f"Worker {worker_id}: Reached max_events limit, stopping file processing")
            sys.stdout.flush()  # Force immediate output to file
            break

    print(
        f"Worker {worker_id}: Completed processing. Valid events: {total_valid_events}, Number of excluded events {excluded_events_count}, Total events seen: {total_events_seen}"
    )
    sys.stdout.flush()  # Force immediate output to file
    # print("These are the file indices: ", file_indices)
    return {
        "excluded_tracks_count": excluded_tracks_count,
        "excluded_events_count": excluded_events_count,
        "valid_events_count": total_valid_events,
        "valid_tracks_count": valid_tracks_count,
        "total_events_seen": total_events_seen,  # Add this
        "tracks_excluded_pt": tracks_excluded_pt,
        "tracks_excluded_eta": tracks_excluded_eta,
        "tracks_excluded_hits": tracks_excluded_hits,
        "events_excluded_no_tracks_after_filtering": events_excluded_no_tracks_after_filtering,
        "events_excluded_technology_filtering": events_excluded_technology_filtering,
        "events_excluded_no_hits_after_technology": events_excluded_no_hits_after_technology,
        "event_mapping": event_mapping,
        "file_indices": file_indices,
        "row_indices": row_indices,
        "num_hits_per_event": num_hits_per_event,
        "num_tracks_per_event": num_tracks_per_event,
    }


def save_worker_chunk_to_hdf5(
    hits_chunk, tracks_chunk, event_numbers_chunk, output_dir, worker_id, root_file, valid_events_count, hit_features, track_features
):
    """Save a chunk of data to HDF5 file."""
    data_dir = Path(output_dir) / "data"

    # Extract root file name without extension and add event count
    root_file_stem = Path(root_file).stem
    h5_filename = f"{root_file_stem}_{valid_events_count}events.h5"
    h5_file = data_dir / h5_filename

    chunk_info = {
        "h5_file": f"data/{h5_filename}",
        "source_root_file": str(root_file),
        "worker_id": worker_id,
        "event_range": {"count": len(hits_chunk)},
    }

    with h5py.File(h5_file, "w") as f:
        num_events = len(hits_chunk)
        max_hits = max(len(hits["spacePoint_time"]) for hits in hits_chunk)
        max_tracks = max(len(tracks["truthMuon_pt"]) for tracks in tracks_chunk)
        num_hit_features = len(hit_features)
        num_track_features = len(track_features)

        # Create and fill hit arrays
        hits_array = np.full((num_events, max_hits, num_hit_features), np.nan, dtype=np.float32)
        for event_idx, hits_dict in enumerate(hits_chunk):
            num_hits = len(hits_dict["spacePoint_time"])
            for feat_idx, feature in enumerate(hit_features):
                hits_array[event_idx, :num_hits, feat_idx] = hits_dict[feature]

        # Create and fill track arrays
        tracks_array = np.full((num_events, max_tracks, num_track_features), np.nan, dtype=np.float32)
        for event_idx, tracks_dict in enumerate(tracks_chunk):
            num_tracks = len(tracks_dict["truthMuon_pt"])
            for feat_idx, feature in enumerate(track_features):
                tracks_array[event_idx, :num_tracks, feat_idx] = tracks_dict[feature]

        # Save datasets
        f.create_dataset("hits", data=hits_array, compression="gzip", compression_opts=6, shuffle=True, fletcher32=True)
        f.create_dataset("tracks", data=tracks_array, compression="gzip", compression_opts=6, shuffle=True, fletcher32=True)

        event_num_hits = np.array([len(hits["spacePoint_time"]) for hits in hits_chunk], dtype=np.int16)
        event_num_tracks = np.array([len(tracks["truthMuon_pt"]) for tracks in tracks_chunk], dtype=np.int16)

        f.create_dataset("num_hits", data=event_num_hits, compression="gzip", compression_opts=6)
        f.create_dataset("num_tracks", data=event_num_tracks, compression="gzip", compression_opts=6)
        f.create_dataset("event_numbers", data=np.array(event_numbers_chunk, dtype=np.int32), compression="gzip", compression_opts=6)

        f.attrs["num_events"] = num_events
        f.attrs["source_root_file"] = str(root_file)
        f.attrs["worker_id"] = worker_id
        f.attrs["valid_events_count"] = valid_events_count

    return chunk_info


def main():
    parser = argparse.ArgumentParser(description="Prefilter ATLAS muon events into batched HDF5 files using parallel processing.")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Directory containing input root files")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to save output HDF5 files")
    parser.add_argument("-n", "--expected_num_events_per_file", type=int, default=2000, help="Expected number of events per root file")
    parser.add_argument("-max", "--max_events", type=int, default=-1, help="Maximum number of valid events each worker is allowed to process")
    parser.add_argument("-w", "--num_workers", type=int, default=None, help="Number of worker processes (default: 10)")
    parser.add_argument("--no-NSW", action="store_true", default=False, help="Remove STGC and MM technology hits from dataset")
    parser.add_argument("--no-RPC", action="store_true", default=False, help="Remove RPC technology hits from dataset")
    parser.add_argument(
        "--disable-track-filtering", action="store_true", default=False, help="Disable track filtering based on pt, eta, and hit count"
    )
    parser.add_argument("--pt-threshold", type=float, default=5.0, help="Minimum pT threshold for tracks (GeV, default: 5.0)")
    parser.add_argument("--eta-threshold", type=float, default=2.7, help="Maximum |eta| threshold for tracks (default: 2.7)")
    parser.add_argument("--num-hits-threshold", type=int, default=3, help="Minimum number of hits per track (default: 3)")

    args = parser.parse_args()

    # Modify output directory path based on technology filtering flags
    if getattr(args, "no_NSW", False) or getattr(args, "no_RPC", False):
        base_output_dir = args.output_dir
        suffix = ""
        if getattr(args, "no_NSW", False):
            suffix += "_no-NSW"
        if getattr(args, "no_RPC", False):
            suffix += "_no-RPC"
        args.output_dir = base_output_dir + suffix

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    filter = ParallelRootFilter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        expected_num_events_per_file=args.expected_num_events_per_file,
        max_events=args.max_events,
        num_workers=args.num_workers,
        no_NSW=getattr(args, "no_NSW", False),
        no_rpc=getattr(args, "no_RPC", False),
        disable_track_filtering=getattr(args, "disable_track_filtering", False),
        pt_threshold=getattr(args, "pt_threshold", 1.0),
        eta_threshold=getattr(args, "eta_threshold", 2.5),
        num_hits_threshold=getattr(args, "num_hits_threshold", 5),
    )

    filter.process_events()


if __name__ == "__main__":
    main()
