import torch
from torch import Tensor, nn


def concat_tensors(tensors: list[Tensor]) -> Tensor:
    x = []

    for tensor in tensors:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        x.append(tensor)

    return torch.concatenate(x, dim=-1)


class InputNet(nn.Module):
    def __init__(self, input_name: str, net: nn.Module, fields: list[str], posenc: nn.Module | None = None):
        super().__init__()
        """ A wrapper which takes a list of input features, concatenates them, and passes them through a dense
        layer followed by an optional positional encoding module.

        Parameters
        ----------
        input_name : str
            The name of the feature / object that will be embedded, e.g. pix for pixel clusters.
        net : nn.Module
            Module used to perform the feature embedding.
        fields : list[str]
            A list of fields belonging to the feature that will be embedded. E.g. [x, y, z] together with a
            input name of "pix" would result in the fields "pix_x", "pix_y" and "pix_z" being concatenated
            together to make the feature vector.
        posenc : nn.Module
            An optional module used to perform the positional encoding.
        """

        self.input_name = input_name
        self.net = net
        self.fields = fields
        self.posenc = posenc

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """Embed the set of input features into an embedding.

        Parameters
        ----------
        inputs : dict
            Input data consisting of a dictionary the requested input features.

        Returns
        -------
        x : Tensor
            Tensor containing an embedding of the concatenated input features.
        """
        # Some input fields will be a vector, i.e. have shape (batch, keys, D) where D > 1
        # But must will be scalars, i.e. (batch, keys), so for these we reshape them to (batch, keys, 1)
        # After this we can then concatenate everything together

        x = self.net(concat_tensors([inputs[f"{self.input_name}_{field}"] for field in self.fields]))

        # Perform an optional positional encoding using the positonal encoding fields
        if self.posenc is not None:
            x += self.posenc(inputs)

        return x

class QueryPosEnc(nn.Module):
    def __init__(self, input_name: str, posenc: nn.Module):
        super().__init__()
        """ A wrapper which takes a list of input features, concatenates them, and passes them through a dense
        layer followed by an optional positional encoding module.

        Parameters
        ----------
        input_name : str
            The name of the feature / object that will be embedded, e.g. pix for pixel clusters.
        posenc : nn.Module
            An optional module used to perform the positional encoding.
        num_queries : int
            The number of queries to generate positional encodings for.
        """

        self.input_name = input_name
        self.posenc = posenc

    def forward(self, inputs: dict[str, Tensor], num_queries: int, batch_size: int, device: torch.device) -> Tensor:
        """Embed the set of input features into an embedding.

        Parameters
        ----------
        inputs : dict
            Input data consisting of a dictionary the requested input features.
        num_queries : int
            The number of queries to generate positional encodings for.
        batch_size : int
            The batch size.
        device : torch.device
            The device to create tensors on.

        Returns
        -------
        posenc : Tensor
            Tensor containing the positional encoding of the input features.
        """
        # Create inputs for positional encoding
        pos_inputs = {}
        for field in self.posenc.fields:
            if field == "phi":
                # for each query set value of phi to 2pi/n_queries * query_idx - should be shape [batch_size, num_queries]
                query_indices = torch.arange(num_queries, device=device)
                phi_values = 2 * torch.pi * (query_indices / num_queries - 0.5)
                pos_inputs[f"{self.input_name}_{field}"] = phi_values.unsqueeze(0).expand(batch_size, -1)
            else:
                raise ValueError(f"Field {field} not supported for query input net")

        return self.posenc(pos_inputs)