import torch


def get_local_ca_mask(n_objects, n_inputs, window_size, stride=1, device=None, wrap=False):
    assert window_size >= 0, "Window size must be non-negative"
    assert window_size % 2 == 0, "Window size must be even"
    mask = torch.zeros((n_objects, n_inputs), dtype=torch.bool, device=device)
    for i in range(n_objects):
        start_raw = round(i * stride) - window_size // 2
        start = max(0, start_raw)
        end_raw = round(i * stride) + window_size // 2 + 1
        end = min(n_inputs, end_raw)
        mask[i, start:end] = 1

        # if wrap, left and right ends are connected
        if wrap:
            if start_raw < 0:
                start = n_inputs + start_raw
                end = n_inputs
                mask[i, start:end] = 1
            if end_raw > n_inputs:
                start = 0
                end = end_raw - n_inputs
                mask[i, start:end] = 1

    return mask


def auto_local_ca_mask(q, kv, window_size, wrap=True):
    n_objects = q.shape[1]
    n_inputs = kv.shape[1]
    device = q.device
    stride = n_inputs / n_objects
    mask = get_local_ca_mask(n_objects, n_inputs, window_size, stride, device, wrap)
    assert not (~mask.any(dim=0)).any(), "Some columns are all False, increase window size"
    return mask.unsqueeze(0)
