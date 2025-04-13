from pathlib import Path

import awkward as ak
import h5py
import numpy as np


def write_awkward_to_hdf(array: ak.Array, file_path: str | Path, group_name: str):
    with h5py.File(file_path, "w") as file:
        group = file.create_group(group_name)
        packed = ak.to_packed(array)
        form, length, _container = ak.to_buffers(packed, container=group)
        group.attrs["__form"] = form.to_json()
        group.attrs["__length"] = length


def read_awkward_from_hdf(file_path: str | Path, group_name: str) -> ak.Array:
    with h5py.File(file_path, "r") as file:
        group = file[group_name]
        data = {k: np.asarray(v) for k, v in group.items()}
        form = ak.forms.from_json(group.attrs["__form"])
        length = group.attrs["__length"]
        return ak.from_buffers(form, length, data)
