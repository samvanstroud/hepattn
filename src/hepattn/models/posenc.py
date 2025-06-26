import math

import torch
from torch import Tensor, nn


def get_omegas(alpha, dim, **kwargs):
    omega_1 = alpha * torch.logspace(0, 2 / (dim) - 1, (dim // 2), 100, **kwargs)
    omega_2 = omega_1
    if dim % 2 != 0:
        omega_2 = alpha * torch.logspace(0, 2 / (dim) - 1, (dim // 2) + 1, 100, **kwargs)
    return omega_1, omega_2


def pos_enc_symmetric(xs, dim, alpha=1000):
    """Symmetric positional encoding.

    Parameters
    ----------
    xs : torch.Tensor
        Input tensor.
    dim : int
        Dimension of the positional encoding.
    alpha : float, optional
        Scaling factor for the positional encoding, by default 100.

    Returns
    -------
    torch.Tensor
        Symmetric positional encoding.
    """
    xs = xs.unsqueeze(-1)
    kwargs = {"device": xs.device, "dtype": xs.dtype}
    omega_1, omega_2 = get_omegas(alpha, dim, **kwargs)
    p1 = (xs.sin() * omega_1).sin()
    p2 = (xs.cos() * omega_2).sin()
    return torch.cat((p1, p2), dim=-1)


def pos_enc(xs, dim, alpha=1000):
    """Positional encoding.

    Parameters
    ----------
    xs : torch.Tensor
        Input tensor.
    dim : int
        Dimension of the positional encoding.
    alpha : float, optional
        Scaling factor for the positional encoding, by default 100.

    Returns
    -------
    torch.Tensor
        Positional encoding.
    """
    xs = xs.unsqueeze(-1)
    kwargs = {"device": xs.device, "dtype": xs.dtype}
    omega_1, omega_2 = get_omegas(alpha, dim, **kwargs)
    p1 = (xs * omega_1).sin()
    p2 = (xs * omega_2).cos()
    return torch.cat((p1, p2), dim=-1)


class PositionEncoder(nn.Module):
    def __init__(self, input_name: str, fields: list[str], dim: int, sym_fields: list[str] | None = None, alpha=1000, per_input_dim: int | None = None, remainder_dim: int | None = None):
        """Positional encoder.

        Parameters
        ----------
        input_name : str
            The name of the input object that will be encoded.
        fields : list[str]
            List of fields belonging to the object to apply the positional encoding to.
        fields : list[str]
            List of fields that should use a rotationally symmetric positional encoding.
        dim : int
            Dimension to project the positional encoding into.
        alpha : float
            Scaling factor hyperparamater for the positional encoding.
        """
        super().__init__()

        self.input_name = input_name
        self.fields = fields
        self.sym_fields = sym_fields or []
        self.dim = dim
        self.alpha = alpha
        self.per_input_dim = per_input_dim if per_input_dim is not None else self.dim // len(self.fields)
        self.remainder_dim = remainder_dim if remainder_dim is not None else self.dim % len(self.fields)

    def forward(self, inputs: dict):
        """Apply positional encoding to the inputs.

        Parameters
        ----------
        inputs : dict
            Dictionary of inputs.

        Returns
        -------
        torch.Tensor
            Positional encoding of the input variables.
        """
        encodings = []

        for field in self.fields:
            pos_enc_fn = pos_enc_symmetric if field in self.sym_fields else pos_enc
            encodings.append(pos_enc_fn(inputs[f"{self.input_name}_{field}"], self.per_input_dim, self.alpha))
        if self.remainder_dim:
            # Handle remainder by appending zero tensors
            remaining = self.remainder_dim
            while remaining > 0:
                # Take either the full per_input_dim or the remaining amount
                current_size = min(self.per_input_dim, remaining)
                encodings.append(torch.zeros_like(encodings[0])[..., : current_size])
                remaining -= current_size
        encodings = torch.cat(encodings, dim=-1)
        return encodings


class FourierPositionEncoder(nn.Module):
    """
    An implementation of Gaussian Fourier positional encoding.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
    see https://arxiv.org/abs/2006.10739
    """

    def __init__(self, input_name: str, dim: int, fields: list[str], scale: float = 1) -> None:
        super().__init__()
        assert scale > 0
        assert dim % 2 == 0, "Dimension must be even"
        self.input_name = input_name
        self.fields = fields
        self.B = torch.nn.parameter.Buffer(scale * torch.randn((len(fields), dim // 2)))
        self.pi = torch.tensor(math.pi)

    def forward(self, xs: dict[str, Tensor]) -> Tensor:
        xs = torch.cat([xs[f"{self.input_name}_{field}"].unsqueeze(-1) for field in self.fields], dim=-1)
        xs = 2 * self.pi * xs
        xs @= self.B
        return torch.cat([torch.sin(xs), torch.cos(xs)], dim=-1)


# class SharedFourierPositionEncoder(FourierPositionEncoder):
#     """
#     A FourierPositionEncoder that shares the phi projection with a hit encoder.
#     This class is designed to be used for query encoders that need to align with hit encoders.
#     """
    
#     # Class-level registry to store hit encoders
    # _hit_encoders: dict[str, FourierPositionEncoder] = {}
    
    # @classmethod
    # def register_hit_encoder(cls, name: str, encoder: FourierPositionEncoder):
    #     """Register a hit encoder that can be shared with query encoders."""
    #     cls._hit_encoders[name] = encoder
    
    # @classmethod
    # def get_hit_encoder(cls, name: str) -> FourierPositionEncoder:
    #     """Get a registered hit encoder."""
    #     if name not in cls._hit_encoders:
    #         raise ValueError(f"No hit encoder registered with name '{name}'")
    #     return cls._hit_encoders[name]
    
    # def __init__(self, input_name: str, dim: int, fields: list[str], scale: float = 1, 
    #              hit_encoder_name: str = "default") -> None:
    #     # Get the hit encoder
    #     hit_encoder = self.get_hit_encoder(hit_encoder_name)
        
    #     # Create a B matrix with the shared phi projection
    #     query_B = torch.zeros((len(fields), dim // 2), 
    #                          device=hit_encoder.B.device, 
    #                          dtype=hit_encoder.B.dtype)
        
    #     # Map phi to the correct position in query fields
    #     if "phi" in fields and "phi" in hit_encoder.fields:
    #         query_phi_idx = fields.index("phi")
    #         hit_phi_idx = hit_encoder.fields.index("phi")
    #         query_B[query_phi_idx] = hit_encoder.B[hit_phi_idx]
        
    #     # Create field indices mapping
    #     field_indices = {field: i for i, field in enumerate(fields)}
        
    #     super().__init__(
    #         input_name=input_name,
    #         dim=dim,
    #         fields=fields,
    #         scale=scale,
    #         shared_B=query_B,
    #         field_indices=field_indices
    #     )