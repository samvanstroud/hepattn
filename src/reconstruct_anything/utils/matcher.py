import atexit
import contextlib
import time
import warnings
from multiprocessing import get_context, shared_memory
from multiprocessing.pool import ThreadPool
from threading import Lock
from typing import Literal

import numpy as np
import scipy
import torch
from torch import nn

_POOL_LOCK = Lock()
_THREAD_POOLS: dict[int, ThreadPool] = {}
_PROCESS_POOLS = {}


def _get_thread_pool(n_jobs: int) -> ThreadPool:
    """Return (or lazily create) a shared thread pool of the given size."""
    with _POOL_LOCK:
        pool = _THREAD_POOLS.get(n_jobs)
        if pool is None:
            pool = ThreadPool(processes=n_jobs)
            _THREAD_POOLS[n_jobs] = pool
        return pool


def _get_process_pool(n_jobs: int):
    """Get persistent multiprocessing pool using spawn method."""
    with _POOL_LOCK:
        pool = _PROCESS_POOLS.get(n_jobs)
        if pool is None:
            ctx = get_context("spawn")
            pool = ctx.Pool(processes=n_jobs)
            _PROCESS_POOLS[n_jobs] = pool
        return pool


@atexit.register
def _close_pools() -> None:
    """Clean up thread and process pools at exit."""
    for pool in list(_THREAD_POOLS.values()):
        try:
            pool.close()
            pool.join()
        except Exception:  # noqa: BLE001, S110
            pass

    for pool in list(_PROCESS_POOLS.values()):
        try:
            pool.close()
            pool.join(timeout=1.0)
        except Exception:  # noqa: BLE001,
            try:
                pool.terminate()
                pool.join(timeout=1.0)
            except Exception:  # noqa: BLE001, S110
                pass


def solve_scipy(cost):
    """Solve the linear assignment problem using SciPy and return column indices."""
    _, col_idx = scipy.optimize.linear_sum_assignment(cost)
    return col_idx


SOLVERS = {
    "scipy": solve_scipy,
}

try:
    from py_lap_solver.solvers import Solvers as _PyLapSolvers

    if _PyLapSolvers.Lap1015Sequential is not None:
        _lap1015_solver = _PyLapSolvers.Lap1015Sequential

        def solve_lap1015(cost):
            return _lap1015_solver.solve_single(cost)

        SOLVERS["lap1015"] = solve_lap1015
except ImportError:
    warnings.warn(
        "py-lap-solver not installed; only scipy LAP solver available.",
        ImportWarning,
        stacklevel=2,
    )


def match_individual(solver_fn, cost: np.ndarray, default_idx: np.ndarray) -> np.ndarray:
    """Solve one assignment problem and pad unmatched predictions with default indices."""
    pred_idx = np.asarray(solver_fn(cost), dtype=np.int32)

    remaining = np.ones(default_idx.shape[0], dtype=np.bool_)
    remaining[pred_idx] = False
    return np.concatenate([pred_idx, default_idx[remaining]])


def match_parallel(solver_fn, costs_t: np.ndarray, lengths_np: np.ndarray, pred_dim: int, n_jobs: int = 8) -> torch.Tensor:
    """Thread-based parallel matching across batch."""
    batch_size = len(costs_t)
    n_jobs = min(n_jobs, batch_size)
    chunk_size = (batch_size + n_jobs - 1) // n_jobs
    default_idx = np.arange(pred_dim, dtype=np.int32)

    if n_jobs <= 1 or batch_size <= 1:
        results = [match_individual(solver_fn, costs_t[i][: lengths_np[i]], default_idx) for i in range(batch_size)]
        return torch.from_numpy(np.stack(results, axis=0))

    def _run(i: int) -> np.ndarray:
        return match_individual(solver_fn, costs_t[i][: lengths_np[i]], default_idx)

    pool = _get_thread_pool(n_jobs)
    results = pool.map(_run, range(batch_size), chunksize=chunk_size)
    return torch.from_numpy(np.stack(results, axis=0))


def _mp_match_task(args: tuple[str, str, tuple[int, int, int], str, int, int, int]) -> np.ndarray:
    """Worker function for multiprocess matching; reads costs from shared memory."""
    solver_name, shm_name, shape, dtype_str, i, length, pred_dim = args
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        costs_t = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        default_idx = np.arange(pred_dim, dtype=np.int32)
        cost = costs_t[i][:length]
        return match_individual(SOLVERS[solver_name], cost, default_idx)
    finally:
        shm.close()


def match_multiprocess(
    solver_name: str,
    costs_t: np.ndarray,
    lengths_np: np.ndarray,
    pred_dim: int,
    n_jobs: int = 8,
) -> torch.Tensor:
    """Multiprocess matching using shared memory to bypass GIL.

    Raises:
        ValueError: If solver_name is not in the available SOLVERS.
    """
    if solver_name not in SOLVERS:
        raise ValueError(f"Unknown solver: {solver_name}. Available solvers: {list(SOLVERS.keys())}")

    batch_size = len(costs_t)
    n_jobs = min(n_jobs, batch_size)
    chunk_size = (batch_size + n_jobs - 1) // n_jobs

    shm = shared_memory.SharedMemory(create=True, size=costs_t.nbytes)
    try:
        shm_arr = np.ndarray(costs_t.shape, dtype=costs_t.dtype, buffer=shm.buf)
        shm_arr[...] = costs_t

        tasks = [(solver_name, shm.name, costs_t.shape, costs_t.dtype.str, i, int(lengths_np[i]), pred_dim) for i in range(batch_size)]

        pool = _get_process_pool(n_jobs)
        results = pool.map(_mp_match_task, tasks, chunksize=chunk_size)
        return torch.from_numpy(np.stack(results, axis=0))
    finally:
        try:
            shm.close()
        finally:
            with contextlib.suppress(FileNotFoundError):
                shm.unlink()


class Matcher(nn.Module):
    """Match predictions to targets using the linear assignment problem.

    Optionally benchmarks available solvers at runtime and switches to the fastest one.
    """

    def __init__(
        self,
        default_solver: str = "scipy",
        adaptive_solver: bool = True,
        adaptive_check_interval: int = 1000,
        parallel_solver: bool = False,
        parallel_backend: Literal["thread", "process"] = "thread",
        n_jobs: int = 8,
        verbose: bool = False,
    ):
        """Initialize the Matcher.

        Args:
            default_solver: Name of the LAP solver to use initially. Must be a key of ``SOLVERS``.
            adaptive_solver: If True, periodically benchmarks all solvers and switches to the fastest.
            adaptive_check_interval: How many forward calls to wait between solver benchmarks.
            parallel_solver: If True, solves each batch element in parallel.
            parallel_backend: Parallelism backend when ``parallel_solver=True``. One of
                ``'thread'`` or ``'process'``.
            n_jobs: Number of parallel workers. Only used when ``parallel_solver=True``.
            verbose: If True, prints solver timing information during adaptive checks.

        Raises:
            ValueError: If ``default_solver`` or ``parallel_backend`` is invalid.
        """
        super().__init__()
        if default_solver not in SOLVERS:
            raise ValueError(f"Unknown solver: {default_solver}. Available solvers: {list(SOLVERS.keys())}")
        if parallel_backend not in {"thread", "process"}:
            raise ValueError(f"parallel_backend must be 'thread' or 'process', got: {parallel_backend}")
        self.solver = default_solver
        self.adaptive_solver = adaptive_solver
        self.adaptive_check_interval = adaptive_check_interval
        self.parallel_solver = parallel_solver
        self.parallel_backend = parallel_backend
        self.n_jobs = n_jobs
        self.step = 0
        self.verbose = verbose

    def compute_matching(self, costs, object_valid_mask=None, query_valid_mask=None):
        """Run the LAP solver on a batch of cost matrices.

        Args:
            costs: NumPy cost array of shape (B, num_pred, num_target).
            object_valid_mask: Optional boolean tensor (B, num_target). True for valid targets.
            query_valid_mask: Optional boolean tensor (B, num_pred). True for valid queries.
                Invalid queries have their costs set to a large value to avoid assignment.

        Returns:
            Long tensor of shape (B, num_pred) with the assignment column indices.
        """
        if object_valid_mask is None:
            object_valid_mask = torch.ones((costs.shape[0], costs.shape[1]), dtype=torch.bool)

        object_valid_mask = object_valid_mask.detach().bool()
        batch_obj_lengths = torch.sum(object_valid_mask, dim=1).unsqueeze(-1)
        lengths_np = batch_obj_lengths.squeeze(-1).cpu().numpy().astype(np.int32, copy=False)

        pred_dim = costs.shape[1]

        # If we have invalid/padded queries, set their costs to a high value
        # so they won't be matched to valid targets
        if query_valid_mask is not None:
            query_valid_mask = query_valid_mask.detach().bool()
            # Set costs for invalid queries to max float32 value
            # costs shape: [batch, num_pred, num_target]
            invalid_query_mask = ~query_valid_mask.unsqueeze(-1)  # [batch, num_pred, 1]
            costs = np.where(invalid_query_mask.cpu().numpy(), np.finfo(np.float32).max / 10, costs)

        if self.parallel_solver:
            # Transpose costs: [batch, pred, true] -> [batch, true, pred]
            costs_t = np.ascontiguousarray(costs.swapaxes(1, 2))
            if self.parallel_backend == "thread":
                return match_parallel(SOLVERS[self.solver], costs_t, lengths_np, pred_dim, n_jobs=self.n_jobs)
            return match_multiprocess(self.solver, costs_t, lengths_np, pred_dim, n_jobs=self.n_jobs)

        # Sequential matching
        costs_t = costs.swapaxes(1, 2)
        default_idx = np.arange(pred_dim, dtype=np.int32)
        idxs = []

        for k in range(len(costs)):
            cost = costs_t[k][: lengths_np[k]]
            pred_idx = match_individual(SOLVERS[self.solver], cost, default_idx)
            idxs.append(pred_idx)

        return torch.from_numpy(np.stack(idxs))

    @torch.no_grad()
    def forward(self, costs, object_valid_mask=None, query_valid_mask=None):
        """Compute optimal assignments for a batch of cost matrices.

        Args:
            costs: Cost tensor of shape (B, num_pred, num_target).
            object_valid_mask: Optional boolean tensor (B, num_target).
            query_valid_mask: Optional boolean tensor (B, num_pred).

        Returns:
            Long tensor of shape (B, num_pred) with assignment indices.
        """
        # Convert costs to numpy on CPU for solver compatibility
        costs = costs.detach().to(torch.float32).cpu().numpy()

        if self.adaptive_solver and self.step % self.adaptive_check_interval == 0:
            self.adapt_solver(costs)

        pred_idxs = self.compute_matching(costs, object_valid_mask, query_valid_mask)
        self.step += 1

        assert torch.all(pred_idxs >= 0), "Matcher error!"
        return pred_idxs

    def adapt_solver(self, costs):
        """Benchmark all available solvers and switch to the fastest.

        Args:
            costs: NumPy cost array used as a representative benchmark input.
        """
        solver_times = {}

        if self.verbose:
            print("\nAdaptive LAP Solver: Starting solver check...")

        for solver in SOLVERS:
            self.solver = solver
            start_time = time.time()
            self.compute_matching(costs)
            solver_times[solver] = time.time() - start_time

            if self.verbose:
                print(f"Adaptive LAP Solver: Evaluated {solver}, took {solver_times[solver]:.2f}s")

        fastest_solver = min(solver_times, key=solver_times.get)

        if self.verbose:
            if fastest_solver != self.solver:
                print(f"Adaptive LAP Solver: Switching from {self.solver} solver to {fastest_solver} solver\n")
            else:
                print(f"Adaptive LAP Solver: Sticking with {self.solver} solver\n")

        self.solver = fastest_solver
