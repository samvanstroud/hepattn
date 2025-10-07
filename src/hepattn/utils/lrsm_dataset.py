import random
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule, seed_everything
from numpy import ndarray
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from hepattn.utils.tensor_utils import pad_to_size


class LRSMDataset(IterableDataset):
    def __init__(
        self,
        dirpath: str,
        num_samples: int,
        inputs: dict[str, dict[str, list[str]]],
        targets: dict[str, dict[str, list[str]]],
        input_dtype: str = "float32",
        target_dtype: str = "float32",
        input_pad_value: float = 0.0,
        target_pad_value: float = 0.0,
        force_pad_sizes: dict[str, int] | None = None,
        skip_pad_items: list[str] | None = None,
        sampling_seed: int = 42,
        sample_reject_warn_limit: int = 10,
        verbose: bool = False,
    ):
        """A PyTorch Dataset that does lazy rejection sampling with memoisation.
        Samples are read lazily, and evaluated on a selection criteria. The result of the
        selection criteria is memoised for speedup after the first epoch.

        Hndles dynamic sample selection, lazy evaluation, padding, and batching for
        data involving variable-sized inputs.

        Parameters
        ----------
        dirpath : str
            Path to the directory containing the dataset samples.
        num_samples : int
            Number of samples to draw from the dataset.
        inputs : dict[dict[str, list]]
            Dictionary mapping input names to lists of feature names.
        targets : dict[dict[str, list]]
            Dictionary mapping target names to lists of target feature names.
        input_dtype : str
            Floating point type used for the inputs
        target_dtype : str
            Floating point type used for the targets
        input_pad_value : float, optional
            Padding value used for missing input values (default is 0.0).
        target_pad_value : float, optional
            Padding value used for missing target values (default is 0.0).
        force_pad_sizes : dict[str, int], optional
            Specifies a size to which objects / hits will be padded to.
            If not specified, an object is padded to ist largest size in the batch.
        skip_pad_items : list[str]
            Specifies a list of objects which should not be padded,
            e.g. per-sample / event quantites.
        sampling_seed : int, optional
            Random seed used for sampling and reproducibility (default is 42).
        sample_reject_warn_limit : int, optional
            Number of failed sampling attempts before emitting a warning (default is 10).
        """
        if force_pad_sizes is None:
            force_pad_sizes = {}

        if skip_pad_items is None:
            skip_pad_items = []

        super().__init__()

        self.sampling_seed = sampling_seed
        self.dirpath = Path(dirpath)
        self.num_samples = num_samples
        self.inputs = inputs
        self.targets = targets
        self.force_pad_sizes = force_pad_sizes
        self.skip_pad_items = skip_pad_items
        self.input_pad_value = input_pad_value
        self.target_pad_value = target_pad_value
        self.sampling_seed = sampling_seed
        self.sample_reject_warn_limit = sample_reject_warn_limit
        self.dirpath = Path(self.dirpath)
        self.verbose = verbose

        # Setup input and target datatypes
        dtypes = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }

        self.input_dtype = dtypes[input_dtype]
        self.target_dtype = dtypes[target_dtype]

        # Setup random number generators and seeds
        seed_everything(sampling_seed, workers=False)
        self.rng = np.random.default_rng()

        self.sample_ids = None
        self.rejected_sample_ids = set()

        # At the start, we load enough files so that we will have enough for the requested sample size
        # As we read through the dataset, some IDs will be rejected, and so we will load more
        # We will load more samples on demand as needed

    @abstractmethod
    def load_sample(self, sample_id: int) -> dict[str:ndarray] | None:
        """Attempts to load a single sample from disk and validate it against selection criteria.

        Parameters
        ----------
        sample_id : int
            Unique identifier of the sample to load.

        Returns:
        -------
        dict[str, np.ndarray] or None
            Dictionary of input and target arrays if sample is
            valid and passes the selection, otherwise None.
        """

    def register_new_samples(self) -> None:
        """Loads additional samples into the evaluation pool.

        This method should be overrided so that it populates `self.unevaluated_sample_ids` by
        scanning for new samples not yet marked as accepted or rejected.
        """

    def __len__(self) -> int:
        """Returns the number of samples the dataset should contain.

        Returns:
        -------
        int
            Total number of samples.
        """
        return int(self.num_samples)

    def prep_sample(self, sample: dict[str, np.ndarray]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        for item_name, fields in (self.inputs | self.targets).items():
            k = f"{item_name}_valid"
            sample[k] = torch.from_numpy(sample[k])

            for field in fields:
                k = f"{item_name}_{field}"
                sample[k] = torch.from_numpy(sample[k])

        # Apply any manual padding now, need to do it before collate stage incase no batching / collation is used
        for item_name, fields in (self.inputs | self.targets).items():
            # If this item doesnt have any features that are being force padded, continue
            if not any(k in self.force_pad_sizes for k in item_name.split("_")):
                continue

            target_size = tuple(self.force_pad_sizes.get(k, -1) for k in item_name.split("_"))

            # Pad the valid mask, remember that we have a leading singleton batch dim
            k = f"{item_name}_valid"
            sample[k] = pad_to_size(sample[k], target_size, False)

            for field in fields:
                k = f"{item_name}_{field}"

                # Handle the case where the field is a vector by adjusting the target shape accordingly
                if sample[k].dim() - 1 > len(target_size):
                    target_size = (*target_size, sample[k].shape[-1])

                # Check whether this is an input or target so we know the pad value
                pad_value = self.input_pad_value if item_name in self.inputs else self.target_pad_value

                # Now apply the padding
                sample[k] = pad_to_size(sample[k], target_size, pad_value)

        # Convert to a torch tensor of the correct dtype and add the batch dimension
        inputs = {}
        targets = {}
        for input_name, fields in self.inputs.items():
            inputs[f"{input_name}_valid"] = sample[f"{input_name}_valid"].bool().unsqueeze(0)
            # Some tasks might require to know hit padding info for loss masking
            targets[f"{input_name}_valid"] = inputs[f"{input_name}_valid"]
            for field in fields:
                inputs[f"{input_name}_{field}"] = sample[f"{input_name}_{field}"].to(self.input_dtype).unsqueeze(0)

        # Convert the targets
        for target_name, fields in self.targets.items():
            targets[f"{target_name}_valid"] = sample[f"{target_name}_valid"].bool().unsqueeze(0)
            for field in fields:
                targets[f"{target_name}_{field}"] = sample[f"{target_name}_{field}"].to(self.target_dtype).unsqueeze(0)

        return inputs, targets

    def __iter__(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        # Shuffle using worker ID as seed
        sample_ids = list(self.sample_ids)
        random.Random(worker_id + self.sampling_seed).shuffle(sample_ids)

        for idx, sample_id in enumerate(self.sample_ids):
            # Check that this sample_id has been assigned to this worker
            if idx % num_workers != worker_id:
                continue

            # Check if we have already rejected this sample_id
            if sample_id in self.rejected_sample_ids:
                continue

            # Attempt to load the sample with this sample_id
            sample = self.load_sample(sample_id)

            # If the sample was not rejected, prepare it and return it to the iterator
            if sample is not None:
                # Convert dict of numpy arrays into tuple of dict of tensors
                inputs, targets = self.prep_sample(sample)

                # Convert the metedata
                targets["sample_id"] = torch.tensor(sample_id, dtype=torch.int64)

                yield inputs, targets

            # If this sample was rejected, keep a log of it so we don't have to evaluate it again
            else:
                self.rejected_sample_ids.add(sample_id)

    def collate_fn(self, batch: list[tuple[dict[str, Tensor], dict[str, Tensor]]]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Collates a list of samples into a single batched sample.

        Handles variable-length inputs by dynamically padding each input and target field
        to the maximum size found in the batch.

        Parameters
        ----------
        batch : list of tuples
            List where each element is a tuple (inputs, targets), as returned by `__getitem__`.

        Returns:
        -------
        batched_inputs : dict[str, torch.Tensor]
            Batched and padded input tensors.
        batched_targets : dict[str, torch.Tensor]
            Batched and padded target tensors including 'sample_id'.
        """
        # First unpack the batch, merge the inputs and targets back into one dict
        samples = [inputs | targets for inputs, targets in batch]

        # We first go through all of the items and determine the max size of each item within the batch
        pad_sizes = {}
        for item_name in self.inputs | self.targets:
            # Skip features that have multiple dimensions
            if len(item_name.split("_")) > 1:
                continue

            # Skip features marked as ignored for padding
            if item_name in self.skip_pad_items:
                continue

            # Skip features that were manually padded
            if item_name in self.force_pad_sizes:
                continue

            # If no explicit pad size is given for this item, set the pad size to be the max size
            pad_sizes[item_name] = max(sample[f"{item_name}_valid"].shape[1] for sample in samples)

        # Now we apply the padding and the concat
        batched_inputs = {}
        batched_targets = {}
        for item_name, fields in (self.inputs | self.targets).items():
            for sample in samples:
                # If the item was specified as not needing padding, we assume we can just go ahead and concat
                if item_name in self.skip_pad_items:
                    continue

                # The target padding size, ignoring the batch dimension and any field dims
                target_size = tuple(pad_sizes.get(k, -1) for k in item_name.split("_"))

                # Pad the valid mask, remember that we have a leading singleton batch dim
                k = f"{item_name}_valid"
                sample[k] = pad_to_size(sample[k], (1, *target_size), False)

                for field in fields:
                    k = f"{item_name}_{field}"

                    # Handle the case where the field is a vector by adjusting the target shape accordingly
                    if sample[k].dim() - 1 > len(target_size):
                        target_size = (*target_size, sample[k].shape[-1])

                    # Check whether this is an input or target so we know the pad value
                    pad_value = self.input_pad_value if item_name in self.inputs else self.target_pad_value

                    # Now apply the padding
                    sample[k] = pad_to_size(sample[k], (1, *target_size), pad_value)

            # Now we can concatenate the samples into one tensor
            k = f"{item_name}_valid"
            batched_inputs[k] = torch.cat([sample[k] for sample in samples], dim=0)
            batched_targets[k] = batched_inputs[k]

            for field in fields:
                k = f"{item_name}_{field}"
                padded = torch.cat([sample[k] for sample in samples], dim=0)

                if item_name in self.inputs and field in self.inputs[item_name]:
                    batched_inputs[k] = padded

                if item_name in self.targets and field in self.targets[item_name]:
                    batched_targets[k] = padded

        # Now batch the sample IDs
        batched_targets["sample_id"] = torch.stack([sample["sample_id"] for sample in samples], dim=0)

        return batched_inputs, batched_targets


class LRSMDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_class: LRSMDataset,
        batch_size: int,
        train_dir: str,
        val_dir: str,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        test_dir: str | None = None,
        **kwargs,
    ):
        """Lightning data module which wraps together test/train/val LRSM datasets.

        Parameters
        ----------
        dataset_class : LRSMDataset

        batch_size : int
            Number of samples to read in a minibatch.
        train_dir : str
            Training data directory.
        val_dir : str
            Validation data directory.
        test_dir : str
            Test data directory.
        num_workers : int
            Number of workers / threads too use to read batches.
        num_train : int
            Target number of training samples to load.
        num_val : int
            Target number of validation samples to load.
        num_test : int
            Target number of test samples to load.
        """
        super().__init__()

        self.dataset_class = dataset_class
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_workers = num_workers
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.kwargs = kwargs

    def setup(self, stage: str):
        # Create training and validation datasets
        if stage == "fit":
            self.train_dataset = self.dataset_class(dirpath=self.train_dir, num_samples=self.num_train, **self.kwargs)
            self.val_dataset = self.dataset_class(dirpath=self.val_dir, num_samples=self.num_val, **self.kwargs)

        # Only print train/val dataset details when actually training
        if stage == "fit":
            print(f"Created training dataset with {len(self.train_dataset)} samples")
            print(f"Created validation dataset with {len(self.val_dataset)} samples")

        if stage == "test":
            assert self.test_dir is not None, "No test file specified, see --data.test_dir"
            self.test_dataset = self.dataset_class(dirpath=self.test_dir, num_samples=self.num_test, **self.kwargs)
            print(f"Created test dataset with {len(self.test_dataset)} samples")

    def get_dataloader(self, stage: str, dataset: LRSMDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate_fn,
            sampler=None,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        return self.get_dataloader(dataset=self.train_dataset, stage="fit", shuffle=False)

    def val_dataloader(self):
        return self.get_dataloader(dataset=self.val_dataset, stage="test", shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(dataset=self.test_dataset, stage="test", shuffle=False)
