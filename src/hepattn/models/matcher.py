import time
import warnings
from multiprocessing.pool import ThreadPool as Pool

import numpy as np
import scipy
import torch
from torch import nn

from hepattn.utils.import_utils import check_import_safe


def solve_scipy(cost):
    _, col_idx = scipy.optimize.linear_sum_assignment(cost)
    return col_idx


SOLVERS = {
    "scipy": solve_scipy,
}

# Some compiled extension can cause SIGKILL errors if compiled for the wrong arch
# So we have to check they won't kill everything when we import them
if check_import_safe("lap1015"):
    import lap1015

    def solve_1015_early(cost):
        return lap1015.lap_early(cost)

    def solve_1015_late(cost):
        return lap1015.lap_late(cost)

    SOLVERS["lap1015_late"] = solve_1015_late
    # SOLVERS["lap1015_early"] = lap1015_early
else:
    warnings.warn(
        """Failed to import lap1015 solver. This could be because it is not installed,
    or because it was built targeting a different architecture than supported on the current machine.
    Rebuilding the package on the current machine may fix this.""",
        ImportWarning,
        stacklevel=2,
    )


def match_individual(solver_fn, cost: np.ndarray, default_idx: torch.Tensor) -> torch.Tensor:
    pred_idx = torch.as_tensor(solver_fn(cost))
    if solver_fn == SOLVERS["scipy"]:
        pred_idx = torch.concatenate([pred_idx, default_idx[~torch.isin(default_idx, pred_idx)]])
    return pred_idx


def match_parallel(solver_fn, costs: np.ndarray, batch_obj_lengths: torch.Tensor, n_jobs: int = 8) -> torch.Tensor:
    default_idx = torch.arange(costs.shape[2])
    with Pool(processes=n_jobs) as pool:
        # Prepare the arguments for the parallel function
        args = ((solver_fn, costs[k][:, : batch_obj_lengths[k]].T, default_idx) for k in range(len(costs)))
        # Use the pool to map the function to the arguments
        pred_idxs = pool.starmap(match_individual, args)
    return torch.stack(pred_idxs, dim=0)


class Matcher(nn.Module):
    def __init__(
        self,
        default_solver: str = "scipy",
        adaptive_solver: bool = True,
        adaptive_check_interval: int = 1000,
        parallel_solver: bool = False,
        n_jobs: int = 8,
        verbose: bool = False,
    ):
        super().__init__()
        """ Used to match predictions to targets based on a given cost matrix.

        Parameters
        ----------
        default_solver : str
            The default solving algorithm to use.
        adaptive_solver : bool
            If true, then after every adaptive_check_interval calls of the solver,
            each solver algorithm is timed and used to determine the fastest solver, which
            is then set as the current solver.
        adaptive_check_interval : bool
            Interval for checking which solver is the fastest.
        parallel_solver : bool
            If true, then the solver will use a parallel implementation to speed up the matching.
        n_jobs: int
            Number of jobs to use for parallel matching. Only used if parallel_solver is True.
        verbose : bool
            If true, extra information on solver timing is printed.
        """
        if default_solver not in SOLVERS:
            raise ValueError(f"Unknown solver: {default_solver}. Available solvers: {list(SOLVERS.keys())}")
        self.solver = default_solver
        self.adaptive_solver = adaptive_solver
        self.adaptive_check_interval = adaptive_check_interval
        self.parallel_solver = parallel_solver
        self.n_jobs = n_jobs
        self.step = 0
        self.verbose = verbose

    def compute_matching(self, costs, object_valid_mask=None):
        if object_valid_mask is None:
            object_valid_mask = torch.ones((costs.shape[0], costs.shape[1]), dtype=bool)

        object_valid_mask = object_valid_mask.detach().bool()
        batch_obj_lengths = torch.sum(object_valid_mask, dim=1).unsqueeze(-1)

        idxs = []
        default_idx = torch.arange(costs.shape[2])

        if self.parallel_solver:
            # If we are using a parallel solver, we can use it to speed up the matching
            return match_parallel(SOLVERS[self.solver], costs, batch_obj_lengths, n_jobs=self.n_jobs)

        # Do the matching sequentially for each example in the batch
        for k in range(len(costs)):
            # remove invalid targets for efficiency
            cost = costs[k][:, : batch_obj_lengths[k]].T
            # Solve the matching problem using the current solver
            pred_idx = match_individual(SOLVERS[self.solver], cost, default_idx)
            # These indicies can be used to permute the predictions so they now match the truth objects
            idxs.append(pred_idx)

        return torch.stack(idxs)

    @torch.no_grad()
    def forward(self, costs, object_valid_mask=None):
        # Cost matrix dimensions are batch, pred, true
        # Solvers need numpy arrays on the cpu
        costs = costs.detach().to(torch.float32).cpu().numpy()

        # If we are at a check interval, use the current cost batch to see which
        # solver is the fastest, and set that to be the new solver
        if self.adaptive_solver and self.step % self.adaptive_check_interval == 0:
            self.adapt_solver(costs)

        pred_idxs = self.compute_matching(costs, object_valid_mask)
        self.step += 1

        assert torch.all(pred_idxs >= 0), "Matcher error!"
        return pred_idxs

    def adapt_solver(self, costs):
        solver_times = {}

        if self.verbose:
            print("\nAdaptive LAP Solver: Starting solver check...")

        # For each solver, compute the time to match the entire batch
        for solver in SOLVERS:
            # Switch to the solver we are testing
            self.solver = solver
            start_time = time.time()
            self.compute_matching(costs)
            solver_times[solver] = time.time() - start_time

            if self.verbose:
                print(f"Adaptive LAP Solver: Evaluated {solver}, took {solver_times[solver]:.2f}s")

        # Get the solver that was the fastest
        fastest_solver = min(solver_times, key=solver_times.get)

        if self.verbose:
            if fastest_solver != self.solver:
                print(f"Adaptive LAP Solver: Switching from {self.solver} solver to {fastest_solver} solver\n")
            else:
                print(f"Adaptive LAP Solver: Sticking with {self.solver} solver\n")

        # Set the new solver to be the solver with the fastest time for the cost batch
        self.solver = fastest_solver
