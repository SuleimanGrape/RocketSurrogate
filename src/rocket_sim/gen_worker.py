"""Process-isolated simulation pool with hard-kill timeouts and worker recycling.

Why this exists
---------------
RocketPy/scipy solves cannot be interrupted from outside the thread running
them, so the previous thread-based timeout (simulator.run_simulation) left
unkillable daemon threads running the integrator in the background. Inside a
never-recycled multiprocessing.Pool worker those zombies — and their solver
state + numpy arrays — accumulated until the machine OOM'd overnight.

Structural fix
--------------
Each simulation runs SYNCHRONOUSLY inside a persistent child process. The
parent watches every in-flight task; on overrun it ``kill()``s the child, so
the OS reclaims *everything* the solve allocated regardless of leaks. A killed
or recycled child is replaced with a fresh one. Workers are also recycled
after ``maxtasksperchild`` completed tasks to bound any slow residual growth.

Public API
----------
``run_batch(batch, ...)`` yields ``(index, result_or_None)`` in completion
order, where ``result`` is the dict from the pipeline (``{"input", "output"}``)
or ``None`` for reject/timeout/crash. ``index`` is the position in ``batch``.
"""

import os
import sys
import time
import queue
import multiprocessing as mp
from typing import Optional

# Ensure this package dir is importable in spawned children (spawn propagates
# the parent's sys.path, but be defensive if imported oddly).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config as cfg  # noqa: E402

_SENTINEL = None  # shutdown signal placed on a worker's input queue


def _simulate_one(param: dict) -> Optional[dict]:
    """Full pipeline for one rocket, run synchronously (no internal timeout).

    Mirrors generator._prevalidate_and_simulate but uses the synchronous
    solve; the parent process enforces the wall-clock limit by killing us.
    """
    import validator
    import rocket_builder
    import simulator
    import outputs

    ok, _ = validator.prevalidate(param)
    if not ok:
        return None
    try:
        rocket = rocket_builder.build_rocket(param)
    except Exception:
        return None

    flight = simulator.simulate_flight(rocket, param)
    if flight is None:
        return None
    if not validator.is_valid(param, flight):
        return None
    try:
        return {
            "input": outputs.extract_input(param),
            "output": outputs.extract_output(param, flight),
        }
    except Exception:
        return None


def _worker_loop(in_q, out_q):
    """Child: pull (idx, param) tasks, run synchronously, push (idx, result).

    Exits on the sentinel. The parent controls recycling (after N tasks it
    sends the sentinel) and hard-kills us on timeout — so this loop stays dumb.
    """
    # RocketPy pulls in matplotlib; force the non-interactive backend so no GUI
    # canvas state is ever created in a worker.
    import matplotlib
    matplotlib.use("Agg")
    import warnings
    warnings.filterwarnings("ignore")

    while True:
        try:
            item = in_q.get()
        except (EOFError, OSError):
            return
        if item is _SENTINEL:
            return
        idx, param = item
        try:
            res = _simulate_one(param)
        except Exception:
            res = None
        out_q.put((idx, res))


class _Slot:
    """One worker process plus its dedicated input queue and bookkeeping."""

    __slots__ = ("ctx", "out_q", "in_q", "proc", "cur_idx", "started", "done_count")

    def __init__(self, ctx, out_q):
        self.ctx = ctx
        self.out_q = out_q
        self.in_q = None
        self.proc = None
        self.cur_idx = None      # index of in-flight task, or None if idle
        self.started = 0.0       # monotonic dispatch time of current task
        self.done_count = 0      # tasks completed since this process spawned
        self._spawn()

    def _spawn(self):
        self.in_q = self.ctx.Queue()
        self.proc = self.ctx.Process(
            target=_worker_loop, args=(self.in_q, self.out_q), daemon=True)
        self.proc.start()
        self.cur_idx = None
        self.done_count = 0

    @property
    def busy(self) -> bool:
        return self.cur_idx is not None

    def dispatch(self, idx, param):
        self.cur_idx = idx
        self.started = time.monotonic()
        self.in_q.put((idx, param))

    def mark_done(self):
        self.cur_idx = None
        self.done_count += 1

    def recycle(self):
        """Replace the process with a fresh one (graceful)."""
        try:
            self.in_q.put(_SENTINEL)
            self.proc.join(timeout=2.0)
        except Exception:
            pass
        if self.proc.is_alive():
            self.kill()
        else:
            self._cleanup_queue()
            self._spawn()

    def kill(self):
        """Hard-kill the process and replace it (used on timeout/hang)."""
        try:
            self.proc.kill()
        except Exception:
            pass
        try:
            self.proc.join(timeout=2.0)
        except Exception:
            pass
        self._cleanup_queue()
        self._spawn()

    def _cleanup_queue(self):
        try:
            self.in_q.close()
            self.in_q.cancel_join_thread()
        except Exception:
            pass

    def shutdown(self):
        try:
            if self.proc.is_alive():
                self.in_q.put(_SENTINEL)
                self.proc.join(timeout=2.0)
            if self.proc.is_alive():
                self.proc.kill()
        except Exception:
            pass


def _timeout_for(param, force_timeout, per_sim_timeout) -> float:
    if force_timeout is not None:
        return force_timeout
    if per_sim_timeout is not None:
        return per_sim_timeout
    return cfg.SIM_TIMEOUT_BY_CLASS.get(param["motor_class"], cfg.SIM_TIMEOUT_S)


def run_batch(
    batch,
    workers: int = 4,
    maxtasksperchild: int = 10,
    per_sim_timeout: Optional[float] = None,
    force_timeout: Optional[float] = None,
    poll_interval: float = 0.2,
):
    """Run ``batch`` through process-isolated workers; yield (idx, result).

    - Each task runs in a child process; on wall-clock overrun the child is
      killed (OS reclaims all memory) and the task yields ``(idx, None)``.
    - Workers recycle after ``maxtasksperchild`` tasks to bound slow growth.
    - Results are yielded in completion order as they arrive.
    """
    n = len(batch)
    if n == 0:
        return

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    workers = max(1, workers)
    slots = [_Slot(ctx, out_q) for _ in range(min(workers, n))]

    next_task = 0           # next index in batch to dispatch
    completed = 0
    timeouts = {}           # idx -> deadline timestamp for in-flight tasks
    resolved = set()        # idx already yielded (guards kill/late-result races)

    try:
        while completed < n:
            # 1) Dispatch queued work to idle workers.
            for s in slots:
                if not s.busy and next_task < n:
                    idx = next_task
                    next_task += 1
                    s.dispatch(idx, batch[idx])
                    timeouts[idx] = s.started + _timeout_for(
                        batch[idx], force_timeout, per_sim_timeout)

            # 2) Drain any finished results (non-blocking).
            drained = False
            while True:
                try:
                    idx, res = out_q.get_nowait()
                except queue.Empty:
                    break
                drained = True
                # find the slot that owned this idx and free/recycle it
                for s in slots:
                    if s.cur_idx == idx:
                        s.mark_done()
                        if maxtasksperchild and s.done_count >= maxtasksperchild:
                            s.recycle()
                        break
                # A late result for a task we already killed/abandoned: drop it.
                if idx in resolved:
                    continue
                resolved.add(idx)
                timeouts.pop(idx, None)
                completed += 1
                yield idx, res

            # 3) Enforce timeouts: kill any worker whose task overran.
            now = time.monotonic()
            for s in slots:
                if s.busy and now >= timeouts.get(s.cur_idx, float("inf")):
                    idx = s.cur_idx
                    timeouts.pop(idx, None)
                    s.kill()  # respawns a fresh, empty process
                    if idx not in resolved:
                        resolved.add(idx)
                        completed += 1
                        yield idx, None

            # 4) Detect a worker that died unexpectedly (crash/segfault) while
            #    busy — its result will never arrive, so reclaim the task.
            for s in slots:
                if s.busy and not s.proc.is_alive() and now >= s.started + 1.0:
                    idx = s.cur_idx
                    timeouts.pop(idx, None)
                    s._cleanup_queue()
                    s._spawn()
                    if idx not in resolved:
                        resolved.add(idx)
                        completed += 1
                        yield idx, None

            if not drained:
                time.sleep(poll_interval)
    finally:
        for s in slots:
            s.shutdown()
        try:
            out_q.close()
            out_q.cancel_join_thread()
        except Exception:
            pass
