import numpy as np
import pytest
import torch

from hepattn.utils.lrsm_dataset import LRSMDataModule, LRSMDataset

# ----- Test helpers -----


class DummyDataset(LRSMDataset):
    """Minimal concrete subclass so we can call prep_sample/collate_fn without
    depending on filesystem. You can set .sample_ids if you want to test __iter__.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Avoid __iter__ paths unless explicitly tested
        self.sample_ids = []

    def load_sample(self, sample_id: int):
        # Not used by most tests (we build samples manually via prep_sample)
        raise NotImplementedError("Not used in these unit tests.")


def make_sample(
    n_hits: int,
    vector_dim: int = 3,
    include_event=False,
    target_kind="labels",
):
    sample = {
        "hits_valid": np.ones((n_hits,), dtype=bool),
        "hits_px": np.arange(n_hits, dtype=np.float32),
        "hits_feat": np.ones((n_hits, vector_dim), dtype=np.float32),
    }

    if target_kind == "labels":
        sample.update({
            "labels_valid": np.ones((1,), dtype=bool),
            "labels_y": np.array([1.0], dtype=np.float32),
        })
    elif target_kind == "truth":
        sample.update({
            "truth_valid": np.ones((n_hits,), dtype=bool),
            "truth_y": np.arange(n_hits, dtype=np.float32),
        })
    else:
        raise ValueError("Unknown target_kind")

    if include_event:
        sample.update({
            "event_valid": np.ones((1,), dtype=bool),
            "event_z": np.array([42.0], dtype=np.float32),
        })

    return sample


# ----- Fixtures -----


@pytest.fixture
def io_specs_basic():
    inputs = {"hits": ["px", "feat"]}  # NOTE: list of field names
    targets = {"labels": ["y"]}
    return inputs, targets


@pytest.fixture
def io_specs_truth_target():
    inputs = {"hits": ["px", "feat"]}
    targets = {"truth": ["y"]}  # per-hit target aligned with 'hits'
    return inputs, targets


@pytest.fixture
def io_specs_with_skip():
    inputs = {"hits": ["px"]}
    targets = {"labels": ["y"], "event": ["z"]}
    return inputs, targets


# ----- Tests for basic properties -----


def test_len_returns_num_samples(io_specs_basic):
    inputs, targets = io_specs_basic
    ds = DummyDataset(".", num_samples=123, inputs=inputs, targets=targets)
    assert len(ds) == 123


def test_dtype_mapping_and_force_padding(io_specs_basic):
    inputs, targets = io_specs_basic
    # Force pad hits to 5, and use different input/target dtypes
    ds = DummyDataset(
        ".",
        num_samples=1,
        inputs=inputs,
        targets=targets,
        input_dtype="float16",
        target_dtype="float64",
        force_pad_sizes={"hits": 5},
    )

    sample = make_sample(n_hits=3)
    inpt, targ = ds.prep_sample(sample)

    # dtype checks
    assert inpt["hits_px"].dtype == torch.float16
    assert inpt["hits_feat"].dtype == torch.float16
    assert targ["labels_y"].dtype == torch.float64
    # valid mask is bool with batch dim
    assert inpt["hits_valid"].dtype == torch.bool
    assert inpt["hits_valid"].shape == (1, 5)  # forced padding length
    # targets should also contain input valid mask copy (for masking)
    assert "hits_valid" in targ
    assert torch.equal(targ["hits_valid"], inpt["hits_valid"])


# ----- Tests for dynamic padding in collate_fn -----


def test_collate_fn_dynamic_pad_scalar_and_vector(io_specs_basic):
    inputs, targets = io_specs_basic
    ds = DummyDataset(".", num_samples=2, inputs=inputs, targets=targets)

    # Two samples with different hit counts
    s1_in, s1_tg = ds.prep_sample(make_sample(n_hits=2))
    s2_in, s2_tg = ds.prep_sample(make_sample(n_hits=5))
    s1_tg["sample_id"] = torch.tensor(7, dtype=torch.int64)
    s2_tg["sample_id"] = torch.tensor(8, dtype=torch.int64)

    bat_in, bat_tg = ds.collate_fn([(s1_in, s1_tg), (s2_in, s2_tg)])

    # Shapes should be [batch, max_len, ...]
    assert bat_in["hits_valid"].shape == (2, 5)
    assert bat_in["hits_px"].shape == (2, 5)
    assert bat_in["hits_feat"].shape == (2, 5, 3)

    # sample_id stacking
    assert bat_tg["sample_id"].dtype == torch.int64
    assert torch.equal(bat_tg["sample_id"], torch.tensor([7, 8], dtype=torch.int64))


def test_collate_fn_respects_pad_values_for_inputs_and_targets(io_specs_truth_target):
    inputs, targets = io_specs_truth_target
    ds = DummyDataset(
        ".",
        num_samples=2,
        inputs=inputs,
        targets=targets,
        input_pad_value=9.0,
        target_pad_value=-1.0,
    )

    # sample 0: shorter (2 hits), sample 1: longer (5 hits)
    s1_in, s1_tg = ds.prep_sample(make_sample(n_hits=2, target_kind="truth"))
    s2_in, s2_tg = ds.prep_sample(make_sample(n_hits=5, target_kind="truth"))
    s1_tg["sample_id"] = torch.tensor(0, dtype=torch.int64)
    s2_tg["sample_id"] = torch.tensor(1, dtype=torch.int64)

    bat_in, bat_tg = ds.collate_fn([(s1_in, s1_tg), (s2_in, s2_tg)])

    # Input padding value should appear in the padded tail of sample 0
    assert torch.all(bat_in["hits_px"][0, 2:] == 9.0)
    # Vector field padded tail should also be input_pad_value
    assert torch.all(bat_in["hits_feat"][0, 2:, :] == 9.0)

    # Target padding value should appear in aligned per-hit target tail
    assert torch.all(bat_tg["truth_y"][0, 2:] == -1.0)


def test_collate_fn_skip_pad_items(io_specs_with_skip):
    inputs, targets = io_specs_with_skip
    ds = DummyDataset(
        ".",
        num_samples=2,
        inputs=inputs,
        targets=targets,
        skip_pad_items=["event"],
    )

    # Ensuring 'event' tensors are identical shapes across samples -> no padding needed
    s1_in, s1_tg = ds.prep_sample(make_sample(n_hits=2, include_event=True))
    s2_in, s2_tg = ds.prep_sample(make_sample(n_hits=5, include_event=True))
    s1_tg["sample_id"] = torch.tensor(10, dtype=torch.int64)
    s2_tg["sample_id"] = torch.tensor(11, dtype=torch.int64)

    bat_in, bat_tg = ds.collate_fn([(s1_in, s1_tg), (s2_in, s2_tg)])

    # 'event' was skipped for padding but still concatenated
    assert bat_in["event_valid"].shape == (2, 1)
    assert bat_tg["event_z"].shape == (2, 1)
    # Values preserved
    assert torch.all(bat_tg["event_z"] == 42.0)


# ----- LRSMDataModule wiring tests (no iteration) -----


def test_datamodule_setup_and_dataloaders(io_specs_basic):
    inputs, targets = io_specs_basic
    dm = LRSMDataModule(
        dataset_class=DummyDataset,  # ty: ignore [invalid-argument-type]
        batch_size=4,
        train_dir=".",
        val_dir=".",
        test_dir=".",
        num_workers=0,  # important: don't spawn workers in unit tests
        num_train=10,
        num_val=5,
        num_test=3,
        inputs=inputs,
        targets=targets,
    )

    dm.setup("fit")
    assert hasattr(dm, "train_dataset")
    assert hasattr(dm, "val_dataset")

    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()

    assert train_dl.batch_size == 4
    assert val_dl.batch_size == 4
    assert train_dl.dataset is dm.train_dataset
    assert val_dl.dataset is dm.val_dataset

    dm.setup("test")
    assert hasattr(dm, "test_dataset")
    test_dl = dm.test_dataloader()
    assert test_dl.batch_size == 4


def test_datamodule_setup_test_requires_dir(io_specs_basic):
    inputs, targets = io_specs_basic
    dm = LRSMDataModule(
        dataset_class=DummyDataset,  # ty: ignore [invalid-argument-type]
        batch_size=1,
        train_dir=".",
        val_dir=".",
        test_dir=None,  # should assert on setup("test")
        num_workers=0,
        num_train=1,
        num_val=1,
        num_test=1,
        inputs=inputs,
        targets=targets,
    )

    with pytest.raises(AssertionError):
        dm.setup("test")
