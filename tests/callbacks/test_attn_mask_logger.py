import numpy as np
import pytest
import torch

from hepattn.callbacks.attn_mask_logger import AttnMaskLogger


class DummyExperiment:
    def __init__(self):
        self.logged_assets: list[dict] = []

    def log_asset(self, *, file_data, file_name, step):
        # Load the saved npz to inspect its contents before it is unlinked
        with np.load(file_data) as data:
            coords = data["coords"]
            num_queries = int(data["num_queries"])
            num_hits = int(data["num_hits"])
        self.logged_assets.append({
            "file_data": file_data,
            "file_name": file_name,
            "step": step,
            "coords": coords,
            "num_queries": num_queries,
            "num_hits": num_hits,
        })


class DummyLogger:
    def __init__(self):
        self.experiment = DummyExperiment()


class DummyModule:
    def __init__(self):
        self.logger = DummyLogger()


@pytest.mark.parametrize("shape", [(5, 7), (10, 10)])
def test_log_mask_points_for_kde_basic(tmp_path, monkeypatch, shape):
    """Ensure that points where mask==1 are saved and logged as expected."""
    monkeypatch.chdir(tmp_path)

    num_queries, num_hits = shape
    mask = torch.ones(num_queries, num_hits, dtype=torch.bool)

    callback = AttnMaskLogger()
    pl_module = DummyModule()

    step = 123
    layer = 4
    prefix = "local_ma_mask"

    callback._log_mask_points_for_kde(pl_module, mask, step, layer, prefix)  # noqa: SLF001

    assets = pl_module.logger.experiment.logged_assets
    assert len(assets) == 1

    asset = assets[0]
    coords = asset["coords"]

    # All coordinates should be within the mask bounds
    assert coords.shape[1] == 2
    assert coords.shape[0] <= num_queries * num_hits
    assert coords[:, 0].min() >= 0
    assert coords[:, 0].max() < num_queries
    assert coords[:, 1].min() >= 0
    assert coords[:, 1].max() < num_hits

    # Shape metadata should match
    assert asset["num_queries"] == num_queries
    assert asset["num_hits"] == num_hits

    # For reasonably small masks we should not hit subsampling
    assert coords.shape[0] == num_queries * num_hits

    # For small sequence lengths, we expect uint16 coordinates
    assert coords.dtype == np.uint16


def test_log_mask_points_for_kde_no_hits(tmp_path, monkeypatch):
    """If there are no hits in the mask, nothing should be logged."""
    monkeypatch.chdir(tmp_path)

    mask = torch.zeros(8, 16, dtype=torch.bool)

    callback = AttnMaskLogger()
    pl_module = DummyModule()

    callback._log_mask_points_for_kde(pl_module, mask, step=0, layer=0, prefix="local_ma_mask")  # noqa: SLF001

    # Early return, so no assets are logged
    assert pl_module.logger.experiment.logged_assets == []


def test_log_mask_points_for_kde_uses_uint32_for_large_masks(tmp_path, monkeypatch):
    """Very large masks should use uint32 storage for indices."""
    monkeypatch.chdir(tmp_path)

    # Make at least one dimension exceed uint16 max so that uint32 is chosen
    num_queries = 70000
    num_hits = 2
    mask = torch.ones(num_queries, num_hits, dtype=torch.bool)

    callback = AttnMaskLogger()
    pl_module = DummyModule()

    callback._log_mask_points_for_kde(pl_module, mask, step=1, layer=1, prefix="local_ma_mask")  # noqa: SLF001

    assets = pl_module.logger.experiment.logged_assets
    assert len(assets) == 1
    coords = assets[0]["coords"]

    # Subsampling may have occurred, but dtype should reflect large dimensions
    assert coords.dtype == np.uint32
