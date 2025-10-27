import torch


def get_omegas(alpha, dim, **kwargs):
    omega_1 = alpha * torch.logspace(0, 2 / (dim) - 1, (dim // 2), 10_000, **kwargs)
    omega_2 = omega_1
    if dim % 2 != 0:
        omega_2 = alpha * torch.logspace(0, 2 / (dim) - 1, (dim // 2) + 1, 10_000, **kwargs)
    return omega_1, omega_2


def pos_enc_symmetric(xs, dim, alpha=100):
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
    kwargs = dict(device=xs.device, dtype=xs.dtype)
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
    kwargs = dict(device=xs.device, dtype=xs.dtype)
    omega_1, omega_2 = get_omegas(alpha, dim, **kwargs)
    p1 = (xs * omega_1).sin()
    p2 = (xs * omega_2).cos()
    return torch.cat((p1, p2), dim=-1)


SYM_VARS = {"phi"}


class PositionalEncoder:
    def __init__(self, variables: list[str], dim: int, alpha=1000):
        """Positional encoder.

        Parameters
        ----------
        variables : list[str]
            List of variables to apply the positional encoding to.
        """
        self.variables = variables
        self.dim = dim
        self.alpha = alpha

        self.per_input_dim = self.dim // len(self.variables)
        self.last_dim = self.per_input_dim + self.dim % len(self.variables)
        if self.last_dim != self.per_input_dim:
            print(f"Last dimension of positional encoding is {self.last_dim} instead of {self.per_input_dim}.")

    def __call__(self, inputs: dict):
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
        for var in self.variables:
            this_dim = self.per_input_dim if var != self.variables[-1] else self.last_dim
            pos_enc_fn = pos_enc_symmetric if var in SYM_VARS else pos_enc
            encodings.append(pos_enc_fn(inputs[f"hit_{var}"], this_dim, self.alpha))
        encodings = torch.cat(encodings, dim=-1)
        return encodings
