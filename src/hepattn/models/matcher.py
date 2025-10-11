import time
import warnings
from multiprocessing.pool import ThreadPool as Pool

import numpy as np
import scipy
import torch
from torch import Tensor, nn

from hepattn.utils.import_utils import check_import_safe

from py_lap_solver.solvers import Solvers

def solve_scipy(cost):
    _, col_idx = scipy.optimize.linear_sum_assignment(cost)
    return col_idx

def match_individual(solver_fn, cost: np.ndarray, default_idx: Tensor) -> np.ndarray:
    pred_idx = solver_fn(cost)
    if solver_fn == SOLVERS["scipy"]:
        pred_idx = np.concatenate([pred_idx, default_idx[~np.isin(default_idx, pred_idx)]])
    return pred_idx

SOLVERS = {
    "scipy": solve_scipy,
}
SOLVER_REGISTRY = Solvers.get_available_solvers()


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


# def match_individual(solver_fn, cost: np.ndarray, default_idx: Tensor) -> np.ndarray:
#     pred_idx = solver_fn(cost)
#     if solver_fn == SOLVERS["scipy"]:
#         pred_idx = np.concatenate([pred_idx, default_idx[~np.isin(default_idx, pred_idx)]])
#     return pred_idx


def match_parallel(solver_fn, costs: np.ndarray, batch_obj_lengths: torch.Tensor, n_jobs: int = 8) -> torch.Tensor:
    batch_size = len(costs)
    chunk_size = (batch_size + n_jobs - 1) // n_jobs
    default_idx = np.arange(costs.shape[2], dtype=np.int32)
    lengths_np = batch_obj_lengths.squeeze(-1).cpu().numpy().astype(np.int32)

    args = [(solver_fn, costs[i][:, : lengths_np[i]].T, default_idx) for i in range(batch_size)]
    with Pool(processes=n_jobs) as pool:
        results = pool.starmap(match_individual, args, chunksize=chunk_size)

    return torch.from_numpy(np.stack(results))

def fill_default_indices(assignments: torch.Tensor, P: int | None = None) -> torch.Tensor:
    """
    Vectorized replacement for '-1' entries in `assignments` using remaining default indices.

    Args:
        assignments: Long tensor of shape [B, M], values in [-1, 0..P-1].
        P: Total number of default indices (0..P-1). If None, inferred from data.

    Returns:
        Tensor of shape [B, M] with '-1' replaced by the per-row available indices in ascending order.
    """
    if assignments.dtype != torch.int64:
        assignments = assignments.to(torch.int64)

    B, M = assignments.shape
    device = assignments.device

    if P is None:
        # Infer P from data when possible; fallback to M
        has_nonneg = (assignments >= 0).any()
        if has_nonneg:
            P = int(assignments[assignments >= 0].max().item()) + 1
        else:
            P = M  # typical case: M == P

    unmatched_mask = assignments.eq(-1)
    if not torch.any(unmatched_mask):
        return assignments

    # Build per-row "used" mask over [0..P-1]
    valid_mask = assignments.ge(0)
    safe_vals = assignments.clamp_min(0)
    used_counts = torch.zeros((B, P), dtype=torch.int32, device=device)
    used_counts.scatter_add_(1, safe_vals, valid_mask.to(used_counts.dtype))
    used_mask = used_counts.gt(0)                            # [B, P]

    # List available indices first per row
    arangeP = torch.arange(P, device=device).unsqueeze(0).expand(B, -1)  # [B, P]
    key = (~used_mask).to(arangeP.dtype) * P + arangeP                   # available first, stable sort
    perm = torch.argsort(key, dim=1, stable=True)                        # [B, P]; first block = available ascending

    # Rank unmatched slots within each row (0,1,2,...) so we map to the first k available indices
    unmatched_rank = unmatched_mask.cumsum(dim=1) - 1                    # [-1, 0, 1, ...] where unmatched
    unmatched_rank = torch.clamp_min(unmatched_rank, 0)
    avail_count = (~used_mask).sum(dim=1, keepdim=True)                  # [B, 1]

    # Make gather indices safe for *all* positions (even matched ones); they’ll be masked out later.
    safe_rank = torch.minimum(unmatched_rank, torch.clamp_min(avail_count - 1, 0))

    # Choose kth available index for each position
    fill_values = perm.gather(1, safe_rank)

    # Replace only where unmatched
    return torch.where(unmatched_mask, fill_values, assignments)


class NewMatcher(nn.Module):
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
        if default_solver not in SOLVER_REGISTRY:
            raise ValueError(f"Unknown solver: {default_solver}. Available solvers: {list(SOLVER_REGISTRY.keys())}")
        self.solver = default_solver
        self.adaptive_solver = adaptive_solver
        self.adaptive_check_interval = adaptive_check_interval
        self.parallel_solver = parallel_solver
        self.n_jobs = n_jobs
        self.step = 0
        self.verbose = verbose
        self.sname = default_solver
        self.solver = SOLVER_REGISTRY[default_solver]
        # self.old_matcher = Matcher(
        #     default_solver='scipy',
        #     adaptive_solver=adaptive_solver,
        #     adaptive_check_interval=adaptive_check_interval,
        #     parallel_solver=parallel_solver,
        #     n_jobs=n_jobs,
        # )
        print("Using solver:", self.sname)

    def compute_matching(self, costs, object_valid_mask=None):
        if object_valid_mask is None:
            object_valid_mask = torch.ones((costs.shape[0], costs.shape[2]), dtype=torch.bool)

        object_valid_mask = object_valid_mask.detach().bool()
        batch_obj_lengths = torch.sum(object_valid_mask, dim=1).unsqueeze(-1)
        num_valid = batch_obj_lengths.detach().cpu().numpy()
        # costs is (batch, pred, true)
        # Transpose to (batch, true, pred) so rows=truth, cols=pred
        costs_transposed = np.transpose(costs, (0, 2, 1))

        # Solve: for each truth (row), assign a prediction (column)
        # num_valid limits to first num_valid predictions (columns)
        assignments = self.solver.batch_solve(costs_transposed, num_valid=num_valid[:, 0])
        assignments = torch.from_numpy(assignments).to(torch.int64)
        # # Fill in any -1 entries with remaining PREDICTION indices
        # assignments = fill_default_indices(assignments, P=costs.shape[1])
        default_ind = torch.arange(costs.shape[1])
        for b in range(len(costs)):
            if (assignments[b] < 0).any():
                unmatched = default_ind[~torch.isin(default_ind, assignments[b])]
                assignments[b, assignments[b] < 0] = unmatched[:torch.sum(assignments[b] < 0)]

        return assignments


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
        num_valid = torch.sum(object_valid_mask, dim=1)

        self.step += 1

        assert torch.all(pred_idxs >= 0), "Matcher error!"
        return pred_idxs

    def adapt_solver(self, costs):
        solver_times = {}

        if self.verbose:
            print("\nAdaptive LAP Solver: Starting solver check...")

        # For each solver, compute the time to match the entire batch
        for sname, solver in SOLVER_REGISTRY.items():
            # Switch to the solver we are testing
            self.solver = solver
            start_time = time.time()
            self.compute_matching(costs)
            solver_times[sname] = time.time() - start_time

            if self.verbose:
                print(f"Adaptive LAP Solver: Evaluated {sname}, took {solver_times[sname]:.2f}s")

        # Get the solver that was the fastest
        fastest_solver = min(solver_times, key=solver_times.get)

        if self.verbose:
            if fastest_solver != self.solver:
                print(f"Adaptive LAP Solver: Switching from {self.solver} solver to {fastest_solver} solver\n")
            else:
                print(f"Adaptive LAP Solver: Sticking with {self.solver} solver\n")

        # Set the new solver to be the solver with the fastest time for the cost batch
        self.solver = fastest_solver

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
            object_valid_mask = torch.ones((costs.shape[0], costs.shape[1]), dtype=torch.bool)

        object_valid_mask = object_valid_mask.detach().bool()
        batch_obj_lengths = torch.sum(object_valid_mask, dim=1).unsqueeze(-1)

        if self.parallel_solver:
            # If we are using a parallel solver, we can use it to speed up the matching
            return match_parallel(SOLVERS[self.solver], costs, batch_obj_lengths, n_jobs=self.n_jobs)

        # Do the matching sequentially for each example in the batch
        idxs = []
        default_idx = torch.arange(costs.shape[2])

        for k in range(len(costs)):
            # remove invalid targets for efficiency
            cost = costs[k][:, : batch_obj_lengths[k]].T
            # Solve the matching problem using the current solver
            pred_idx = match_individual(SOLVERS[self.solver], cost, default_idx)
            # These indicies can be used to permute the predictions so they now match the truth objects
            idxs.append(pred_idx)

        return torch.from_numpy(np.stack(idxs))

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