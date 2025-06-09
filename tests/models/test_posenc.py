from pathlib import Path

import matplotlib.pyplot as plt
import torch

from hepattn.models.posenc import PositionEncoder, pos_enc, pos_enc_symmetric, FourierPositionEncoder



def test_pos_enc():
    xs = torch.linspace(-torch.pi, torch.pi, 1000)
    dim = 128
    out_dir = Path("tests/outputs/posenc")
    out_dir.mkdir(exist_ok=True, parents=True)
    for alpha in [10, 20, 50, 100]:
        pe = pos_enc(xs, dim, alpha)
        pe_sym = pos_enc_symmetric(xs, dim, alpha)

        # display the positional encoding itself
        plt.figure()
        plt.imshow(pe.detach().numpy(), aspect="auto")
        plt.colorbar()
        plt.savefig(out_dir / f"pe_{alpha}.png")
        plt.close()

        plt.figure()
        plt.imshow(pe_sym.detach().numpy(), aspect="auto")
        plt.colorbar()
        plt.savefig(out_dir / f"pe_sym_{alpha}.png")
        plt.close()

        sim = pe @ pe.T
        sim_sym = pe_sym @ pe_sym.T
        plt.figure()
        plt.imshow(sim.detach().numpy(), aspect="auto")
        plt.colorbar()
        plt.savefig(out_dir / f"sim_{alpha}.png")
        plt.close()

        plt.figure()
        plt.imshow(sim_sym.detach().numpy(), aspect="auto")
        plt.colorbar()
        plt.savefig(out_dir / f"sim_sym_{alpha}.png")
        plt.close()


def test_pos_enc_class():
    posenc = PositionEncoder(input_name="test_input", fields=["x", "y", "z"], dim=128, alpha=100)
    x = y = z = torch.randn(1, 100, 1)
    inputs = {"test_input_x": x, "test_input_y": y, "test_input_z": z}
    out = posenc(inputs)
    assert out.shape[-1] == 128


def test_pos_enc_random():
    variables = ["x", "y", "z"]
    pe = FourierPositionEncoder(dim=128, input_name='test', fields=variables)
    x = y = z = torch.randn(10, 100)
    xs = {"test_x": x, "test_y": y, "test_z": z}
    embedding = pe(xs)
    assert embedding.shape == (10, 100, 128)
