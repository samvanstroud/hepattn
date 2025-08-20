import torch

from hepattn.utils import model_utils


def test_unmerge_inputs():
    input_names = ["input1", "input2"]

    b, s, d = 2, 5, 8
    key_is_input1 = torch.full((b, s), False)
    key_is_input1[0, 0] = True
    key_is_input1[0, 1] = True
    key_is_input1[1, 1] = True
    key_is_input1[1, 2] = True
    key_is_input2 = torch.full((b, s), False)
    key_is_input2[0, 3] = True
    key_is_input2[1, 4] = True

    x = {
        "key_embed": torch.randn(b, s, d),
        "key_is_input1": key_is_input1,
        "key_is_input2": key_is_input2,
    }

    expected_output = {
        "key_embed": x["key_embed"],
        "key_is_input1": key_is_input1,
        "key_is_input2": key_is_input2,
        "input1_embed": x["key_embed"][[0, 0, 1, 1], [0, 1, 1, 2]].view(2, -1, d),
        "input2_embed": x["key_embed"][[0, 1], [3, 4]].view(2, -1, d),
    }

    output = model_utils.unmerge_inputs(x, input_names)
    assert all(torch.equal(output[key], expected_output[key]) for key in expected_output)
