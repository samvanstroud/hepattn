import time

import numpy as np
import scipy
import torch
from torch import nn

import lap1015


def solve_scipy(cost):
    _, col_idx = scipy.optimize.linear_sum_assignment(cost.T)
    return col_idx


def solve_1015_early(cost):
    return lap1015.lap_late(cost)


def solve_1015_late(cost):
    return lap1015.lap_late(cost)


solvers = {
    "scipy": solve_scipy,
    "1015_early": solve_1015_early,
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

        self.solver = solvers[default_solver]
        self.adaptive_solver = adaptive_solver
        self.adaptive_check_interval = adaptive_check_interval
        self.step = 0

    def compute_matching(self, costs):
        pred_idxs = np.zeros(shape=(costs.shape[0], costs.shape[1]), dtype=int)

        # Do the matching sequentially for each example in the batch
        for k in range(len(costs)):
            pred_idx = self.solver(costs[k])

            # These indicies can be used to permute the predictions so they now match the truth objects
            pred_idxs[k] = pred_idx

        return pred_idxs

    @torch.no_grad()
    def forward(self, costs):
        # Cost matrix dimensions are batch, pred, true
        # Have to detach and move the tensor from GPU to CPU then convert to numpy so we can use SciPy matcher
        device = costs.device
        costs = costs.detach().to(torch.float32).cpu().numpy()
        pred_idxs = self.compute_matching(costs)

        self.step += 1

        # If we are at a check interval, use the current cost batch to see which
        # solver is the fastest, and set that to be the new solver
        if self.adaptive_solver and self.step % self.adaptive_check_interval == 0:
            self.adapt_solver(costs)

        # Convert back into a torch tensor and move it back onto the GPU
        return torch.from_numpy(pred_idxs).long().to(device)

    def adapt_solver(self, costs):
        solver_times = {}

        # For each solver, compute the time to match the entire batch
        for solver_name, solver in solvers.items():
            # Switch to the solver we are testing
            self.solver = solver
            t_start = time.time()
            self.compute_matching(costs)
            solver_times[solver_name] = time.time() - t_start

        # Get the solver that was the fastest
        fastest_solver = min(solver_times, key=solver_times.get)

        # Set the new solver to be the solver with the fastest time for the cost batch
        self.solver = solvers[fastest_solver]
