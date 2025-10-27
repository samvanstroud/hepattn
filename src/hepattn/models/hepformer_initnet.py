from torch import nn

from hepattn.models import Dense
from hepattn.utils.hepformer_positional_enc import PositionalEncoder
from hepattn.utils.tensor_utils import concat_tensors, get_module_dtype, get_torch_dtype


class InitNet(nn.Module):
    def __init__(
        self,
        name: str,
        net: Dense,
        fields,
        pos_enc: PositionalEncoder | None = None,
    ):
        """Initialiser network. Just a named dense network.

        Parameters
        ----------
        name : str
            Name of the input.
        net : Dense
            Dense network for performing the the initial embedding.
        pos_enc : PositionalEncoder, optional
            Positional encoder, by default None
        """
        super().__init__()

        self.name = name
        self.net = net
        self.pos_enc = pos_enc
        self.fields = fields

    def forward(self, inputs: dict):
        x = self.net(concat_tensors([inputs[f"{self.name}_{field}"] for field in self.fields]))
        if self.pos_enc:
            x += self.pos_enc(inputs)
        return x
