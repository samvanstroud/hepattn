import time

import scipy
import torch
from torch import nn

import lap1015


def solve_scipy(cost):
    _, col_idx = scipy.optimize.linear_sum_assignment(cost.T)
    return torch.as_tensor(col_idx)


def solve_1015_early(cost):
    return torch.as_tensor(lap1015.lap_early(cost.T))


def solve_1015_late(cost):
    return torch.as_tensor(lap1015.lap_late(cost.T))


SOLVERS = {
    "scipy": solve_scipy,
    # "1015_early": solve_1015_early,
    "1015_late": solve_1015_late,
}


class Matcher(nn.Module):
    def __init__(
        self,
        default_solver: str = "scipy",
        adaptive_solver: bool = True,
        adaptive_check_interval: int = 100,
    ):
        super().__init__()
        """ Used to match predictions to targets based on a given cost matrix.

        Parameters
        ----------
        default_solver: str
            The default solving algorithm to use.
        adaptive_solver: bool
            If true, then after every adaptive_check_interval calls of the solver,
            each solver algorithm is timed and used to determine the fastest solver, which
            is then set as the current solver.
        adaptive_check_interval: bool
            Interval for checking which solver is the fastest.
        """
        if default_solver not in SOLVERS:
            raise ValueError(f"Unknown solver: {default_solver}. Available solvers: {list(SOLVERS.keys())}")
        self.default_solver = default_solver
        self.solver = SOLVERS[default_solver]
        self.adaptive_solver = adaptive_solver
        self.adaptive_check_interval = adaptive_check_interval
        self.step = 0

    def compute_matching(self, costs, pad_mask=None):
        if pad_mask is None:
            pad_mask = torch.ones((costs.shape[0], costs.shape[1]), dtype=bool)
        pad_mask = pad_mask.detach().bool()
        batch_obj_lengths = torch.sum(pad_mask, dim=1).unsqueeze(-1)

        idxs = []
        default_idx = set(range(costs.shape[2]))
        # Do the matching sequentially for each example in the batch
        for k in range(len(costs)):
            # Get the cost matrix for the k-th element in batch and solve the assignment
            cost = costs[k][:, : batch_obj_lengths[k]]
            pred_idx = self.solver(cost)

            # scipy returns incomplete assignments, handle that here
            if self.default_solver == "scipy":
                full_col_idx = torch.empty(costs.shape[2], dtype=torch.long)
                full_col_idx[: batch_obj_lengths[k]] = pred_idx
                full_col_idx[batch_obj_lengths[k] :] = torch.tensor(list(default_idx - set(pred_idx.numpy())), dtype=torch.long)
                pred_idx = full_col_idx

            # These indicies can be used to permute the predictions so they now match the truth objects
            idxs.append(pred_idx)

        # Stack the indices into a tensor
        pred_idxs = torch.stack(idxs)

        return pred_idxs

    @torch.no_grad()
    def forward(self, costs, pad_mask=None):
        # Cost matrix dimensions are batch, pred, true
        # Have to detach and move the tensor from GPU to CPU then convert to numpy first
        device = costs.device
        costs = costs.detach().to(torch.float32)
        costs = torch.nan_to_num(costs, nan=1e6, posinf=1e6, neginf=1e6)
        costs = costs.cpu().numpy()
        pred_idxs = self.compute_matching(costs, pad_mask=pad_mask)

        self.step += 1

        # If we are at a check interval, use the current cost batch to see which
        # solver is the fastest, and set that to be the new solver
        if self.adaptive_solver and self.step % self.adaptive_check_interval == 0:
            self.adapt_solver(costs)

        # Convert back into a torch tensor and move it back onto the GPU
        return pred_idxs.long().to(device)

    def adapt_solver(self, costs):
        solver_times = {}

        # For each solver, compute the time to match the entire batch
        for solver_name, solver in SOLVERS.items():
            # Switch to the solver we are testing
            self.solver = solver
            start_time = time.time()
            self.compute_matching(costs)
            solver_times[solver_name] = time.time() - start_time

        # Get the solver that was the fastest
        fastest_solver = min(solver_times, key=solver_times.get)

        # Set the new solver to be the solver with the fastest time for the cost batch
        self.solver = SOLVERS[fastest_solver]
