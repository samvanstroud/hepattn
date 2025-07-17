from collections import defaultdict
from pathlib import Path

import torch
import yaml


class VarTransform:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.type = self.config.get("type", "std")
        self.fn = self.config.get("fn", None)

        if self.type == "std":
            self.shift = self.config.get("mean", 0.0)
            self.scale = self.config.get("std", 1.0)
        elif self.type == "min_max":
            self.shift = self.config.get("min", 0.0)
            self.scale = self.config.get("max", 1.0) - self.shift
        elif self.type == "min_max_sym":
            min_ = self.config.get("min", 0.0)
            max_ = self.config.get("max", 1.0)
            self.shift = (max_ + min_) / 2
            self.scale = (max_ - min_) / 2
        assert self.type in {"std", "min_max", "min_max_sym"}, f"Unknown scaling type {self.type}"

    def transform(self, x, shift=None, scale=None):
        if shift is None:
            shift = self.shift
        if scale is None:
            scale = self.scale
        if self.fn == "log":
            x = torch.log(x)
        elif self.fn == "log1p":
            x = torch.log1p(x)
        elif self.fn == "sqrt":
            x = torch.sqrt(x)

        return (x - shift) / scale

    def inverse_transform(self, x, shift=None, scale=None):
        if shift is None:
            shift = self.shift
        if scale is None:
            scale = self.scale
        x = x * scale + shift
        if self.fn == "log":
            return torch.exp(x)
        if self.fn == "log1p":
            return torch.expm1(x)
        if self.fn == "sqrt":
            return torch.pow(x, 2)

        return x


def get_empty_transform() -> VarTransform:
    """
    Get an empty VarTransform that does not apply any transformation.

    Returns
    -------
    VarTransform
        An instance of VarTransform with no transformation applied.
    """
    return VarTransform("", {"type": "std", "mean": 0.0, "std": 1.0})


class FeatureScaler:
    def __init__(self, scale_dict_path: str):
        """
        Initialize the FeatureScaler with a path to a YAML file containing scaling parameters.

        Parameters
        ----------
        scale_dict_path : str
            Path to the YAML file containing scaling parameters for features.
        """
        with Path.open(scale_dict_path) as f:
            self.scale_dict = yaml.safe_load(f)
        self.transforms = defaultdict(get_empty_transform)
        for name, config in self.scale_dict.items():
            self.transforms[name] = VarTransform(name, config)

    def __getitem__(self, name: str) -> VarTransform:
        """
        Get the VarTransform for a specific feature name.

        Parameters
        ----------
        name : str
            The name of the feature.

        Returns
        -------
        VarTransform
            The VarTransform object for the specified feature.
        """
        return self.transforms[name]

    def transform(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Apply the scaling transformations to the input features.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary of features to be transformed.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of transformed features.
        """
        for key, value in x.items():
            if key in self.transforms:
                x[key] = self.transforms[key].transform(value)
        return x

    def inverse_transform(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Apply the inverse scaling transformations to the input features.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Dictionary of features to be inverse transformed.
        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of inverse transformed features.
        """
        for key, value in x.items():
            if key in self.transforms:
                x[key] = self.transforms[key].inverse_transform(value)
        return x
