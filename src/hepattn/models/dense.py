from torch import Tensor, nn

from hepattn.models.activation import SwiGLU


class Dense(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int | None = None,
        hidden_layers: list[int] | None = None,
        hidden_dim_scale: int = 2,
        activation: nn.Module | None = None,
        final_activation: nn.Module | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        norm_input: bool = False,
    ) -> None:
        """A fully connected feed forward neural network, which can take in additional contextual information.

        Args:
            input_size: Input size.
            output_size: Output size. If not specified, this will be the same as the input size.
            hidden_layers: Number of nodes per layer. If not specified, the network will have a single hidden layer with size
                `input_size * hidden_dim_scale`.
            hidden_dim_scale: Scale factor for the hidden layer size.
            activation: Activation function for hidden layers.
            final_activation: Activation function for the output layer.
            dropout: Apply dropout with the supplied probability.
            bias: Whether to use bias in the linear layers.
            norm_input: Whether to apply layer normalization to the input.
        """
        super().__init__()

        if output_size is None:
            output_size = input_size
        if hidden_layers is None:
            hidden_layers = [input_size * hidden_dim_scale]
        if activation is None:
            activation = SwiGLU()

        self.input_size = input_size
        self.output_size = output_size
        gate = isinstance(activation, SwiGLU)

        layers = []
        if norm_input:
            layers.append(nn.LayerNorm(input_size))

        node_list = [input_size, *hidden_layers]
        for i in range(len(node_list) - 1):
            in_dim = node_list[i]
            proj_dim = node_list[i + 1]
            proj_dim = proj_dim * 2 if gate else proj_dim

            # inner projection and activation
            layers.extend((nn.Linear(in_dim, proj_dim, bias=bias), activation))

            # maybe dropout
            if dropout:
                layers.append(nn.Dropout(dropout))

        # final projection and activation
        layers.append(nn.Linear(node_list[-1], output_size, bias=bias))
        if final_activation:
            layers.append(final_activation)

        # build the net
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
