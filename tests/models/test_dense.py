import torch
from torch import nn

from hepattn.models import Dense  # Update the import to match your module's filename


def test_dense_default_parameters():
    """Test Dense network with default parameters."""
    model = Dense(input_size=10)
    x = torch.randn(5, 10)  # Batch of 5, input size of 10
    output = model(x)
    assert output.shape == (
        5,
        10,
    ), "Output shape should match batch size and output size"


def test_dense_custom_parameters():
    """Test Dense network with custom parameters."""
    model = Dense(
        input_size=10,
        output_size=5,
        hidden_layers=[20, 15],
        hidden_dim_scale=3,
        activation=nn.ReLU(),
        final_activation=nn.Sigmoid(),
        dropout=0.1,
        bias=False,
        norm_input=True,
    )
    x = torch.randn(4, 10, 10)  # Batch of 4, input size of 10
    output = model(x)
    assert output.shape == (4, 10, 5), "Output shape should match batch size and output size"


def test_dense_forward_propagation():
    """Test Dense forward propagation for expected behavior."""
    model = Dense(input_size=6, output_size=6, hidden_layers=[12, 8])
    x = torch.ones(2, 6)  # Batch of 2, input size of 6
    output = model(x)
    assert torch.all(output != 0), "Output should not be all zeros"


def test_nested_jagged_tensor():
    nt = torch.nested.nested_tensor([torch.arange(12).reshape(2, 6), torch.arange(18).reshape(3, 6)], dtype=torch.float, layout=torch.jagged)
    model = Dense(input_size=6, output_size=6, hidden_layers=[12, 8])
    output = model(nt)
    assert torch.all(output != 0), "Output should not be all zeros"


# test gpu if cuda is available
if torch.cuda.is_available():

    def test_compile_gpu():
        model = Dense(input_size=10)
        model.cuda()
        model = torch.compile(model)

        x = torch.randn(5, 10).cuda()
        output = model(x)
        assert output.shape == (5, 10), "Output shape should match batch size and output size"
        assert torch.all(output != 0), "Output should not be all zeros"

    def test_compile_gpu_nested_jagged_tensor():
        nt = torch.nested.nested_tensor(
            [torch.arange(12).reshape(2, 6), torch.arange(18).reshape(3, 6)], dtype=torch.float, layout=torch.jagged, device="cuda"
        )
        model = Dense(input_size=6, output_size=6, hidden_layers=[12, 8])
        model.cuda()
        model = torch.compile(model)

        output = model(nt)
        assert torch.all(output != 0), "Output should not be all zeros"
