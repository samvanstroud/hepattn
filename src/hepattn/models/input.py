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

class QueryInputNet(nn.Module):
    def __init__(self, input_name: str, num_queries: int, dim: int, posenc: nn.Module | None = None):
        super().__init__()
        """ A wrapper which takes a list of input features, concatenates them, and passes them through a dense
        layer followed by an optional positional encoding module.

        Parameters
        ----------
        input_name : str
            The name of the feature / object that will be embedded, e.g. pix for pixel clusters.
        num_queries : int
            The number of queries to embed.
        dim : int
            The dimension of the query embedding.
        fields : list[str]
            A list of fields belonging to the feature that will be embedded. E.g. [x, y, z] together with a
            input name of "pix" would result in the fields "pix_x", "pix_y" and "pix_z" being concatenated
            together to make the feature vector.
        posenc : nn.Module
            An optional module used to perform the positional encoding.
        """

        self.input_name = input_name
        self.num_queries = num_queries
        self.query_initial = nn.Parameter(torch.randn(num_queries, dim))
        self.posenc = posenc

    def forward(self, inputs: dict[str, Tensor], batch_size: int, hit_input_net: InputNet = None) -> Tensor:
        """Embed the set of input features into an embedding.

        Parameters
        ----------
        inputs : dict
            Input data consisting of a dictionary the requested input features.
        batch_size : int
            The batch size of the input data.
        hit_input_net : InputNet
            The input net used to embed the hit features.

        Returns
        -------
        x : Tensor
            Tensor containing an embedding of the concatenated input features.
        """
        # Some input fields will be a vector, i.e. have shape (batch, keys, D) where D > 1
        # But must will be scalars, i.e. (batch, keys), so for these we reshape them to (batch, keys, 1)
        # After this we can then concatenate everything together

        x = self.query_initial.expand(batch_size, -1, -1)

        if self.posenc is not None:
            # Perform an optional positional encoding using the positonal encoding fields
            inputs = {}
            for field in self.posenc.fields:
                if field == "phi":
                    # for each query set value of phi to 2pi/n_queries * query_idx - should be shape [batch_size, num_queries]
                    inputs[f"{self.input_name}_{field}"] = 2 * torch.pi * (torch.arange(self.num_queries, device=x.device) / self.num_queries - 0.5)
                else:
                    raise ValueError(f"Field {field} not supported for query input net")
            # self.posenc.register_hit_encoder(hit_input_net.input_name, hit_input_net.posenc)
            x = x + self.posenc(inputs)

        return x