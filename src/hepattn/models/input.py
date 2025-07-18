from torch import Tensor, nn

from hepattn.utils.tensor_utils import concat_tensors, get_module_dtype, get_torch_dtype


class InputNet(nn.Module):
    def __init__(self, input_name: str, net: nn.Module, fields: list[str], posenc: nn.Module | None = None, input_dtype: str | None = None):
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
        input_dtype : str | None
            If specified, the input embedding and positional encoding will be performed in the given dtype,
            after which the embeddings will be cast back to the global model dtype.
        """

        self.input_name = input_name
        self.net = net
        self.fields = fields
        self.posenc = posenc

        # Record the global model dtype incase we want to have the input net at a different precision
        self.output_dtype = get_module_dtype(self)

        # If specified, change the embed and posenc networks to have a different dtype
        if input_dtype is not None:
            self.input_dtype = get_torch_dtype(input_dtype)
            self.net.to(dtype=self.input_dtype)
            self.posenc.to(dtype=self.input_dtype)
        else:
            self.input_dtype = self.output_dtype

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """Embed the set of input features into an embedding.

        Parameters
        ----------
        inputs : dict
            Input data consisting of a dictionary the requested input features.

        Returns:
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

        # If a specific dtype was specified, make sure we cast back to the
        # dtype the rest of the model is using
        if self.input_dtype != self.output_dtype:
            x = x.to(dtype=self.output_dtype)

        return x
