import time
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum, auto
from threading import Lock
from typing import Dict

import torch
import torch.distributed as dist

from areal.utils import logging
from areal.utils.datapack import flat2d

logger = logging.getLogger("stats_tracker")


class ReduceType(Enum):
    AVG_MIN_MAX = auto()
    AVG = auto()
    SUM = auto()
    MIN = auto()
    MAX = auto()
    SCALAR = auto()


MOE_AUX_LOSSES = {}


class DistributedStatsTracker:
    def __init__(self, name: str = ""):
        self.lock = Lock()
        self.scope_stack = []
        if name:
            self.scope_stack.append(name.strip("/"))
        self.denominators = {}  # key -> denominator key
        self.reduce_types = {}  # key -> ReduceType

        self.stats = defaultdict(list)

    def scope(self, name):
        """Context manager for hierarchical scoping"""
        with self.lock:
            return self.Scope(self, name)

    class Scope:
        def __init__(self, tracker, name):
            self.tracker = tracker
            self.name = name.strip("/")

        def __enter__(self):
            self.tracker.scope_stack.append(self.name)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.tracker.scope_stack.pop()

    def _get_full_key(self, key):
        """Combine scope stack with current key"""
        if not self.scope_stack:
            return key
        return "/".join(self.scope_stack + [key])

    @contextmanager
    def record_timing(self, key):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            with self.lock:
                # NOTE: timing records are fixed under the "timeperf" scope
                full_key = f"timeperf/{key}"
                self._set_reduce_type(full_key, ReduceType.SCALAR)
                self.stats[full_key].append(time.perf_counter() - start_time)

    def denominator(self, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
                    raise ValueError(
                        f"`{key}` must be a pytorch bool tensor: {value.dtype}"
                    )
                if value.numel() == 0:
                    raise ValueError(f"`{key}` must be non-empty")
                full_key = self._get_full_key(key)
                self._set_reduce_type(full_key, ReduceType.SUM)
                self.stats[full_key].append(value.detach().clone())

    def scalar(self, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                full_key = self._get_full_key(key)
                self._set_reduce_type(full_key, ReduceType.SCALAR)
                self.stats[full_key].append(float(value))

    def stat(
        self,
        denominator: str,
        reduce_type: ReduceType | None = None,
        **kwargs,
    ):
        """Record multiple values from a dictionary"""
        with self.lock:
            for key, value in kwargs.items():
                if not isinstance(value, torch.Tensor) or value.dtype != torch.float:
                    raise ValueError(
                        f"`{key}` should be a pytorch float tensor: {value.dtype}"
                    )
                if value.numel() == 0:
                    raise ValueError(f"`{key}` should be non-empty")
                if reduce_type == ReduceType.SCALAR:
                    raise ValueError("Cannot use the scalar reduce type for a tensor")
                full_key = self._get_full_key(key)

                denorm = self._get_full_key(denominator)
                if denorm not in self.stats or not self.stats[denorm]:
                    raise ValueError(f"Denominator `{denorm}` does not exist")
                for x, y in zip(self.stats[denorm], self.stats[full_key] + [value]):
                    assert x.shape == y.shape, (x.shape, y.shape)
                self.denominators[full_key] = denorm

                if reduce_type is None:
                    reduce_type = ReduceType.AVG_MIN_MAX
                self._set_reduce_type(full_key, reduce_type)
                self.stats[full_key].append(value.detach().clone())

    def _set_reduce_type(self, key, reduce_type):
        if not isinstance(reduce_type, ReduceType):
            raise ValueError("reduce_type must be a ReduceType enum")
        self.reduce_types[key] = reduce_type

    def export(self, key=None, reduce_group=None, reset=True) -> Dict[str, float]:
        """Get aggregated statistics"""
        with self.lock:
            if key is not None:
                full_key = self._get_full_key(key)
                result = self._aggregate(full_key, reduce_group)
                if reset:
                    if full_key in self.denominators:
                        self.denominators.pop(full_key)
                    if full_key in self.reduce_types:
                        self.denominators.pop(full_key)
                    self.stats.pop(full_key)
                return result

            # synchronize keys across ranks
            keys = list(self.stats.keys())
            if reduce_group is not None:
                all_keys = [None for _ in range(dist.get_world_size(reduce_group))]
                dist.all_gather_object(all_keys, keys, group=reduce_group)
                # Should ensure that the orders are the same
                keys = sorted(list(set(flat2d(all_keys))))
            results = {}
            for key in keys:
                results.update(self._aggregate(key, reduce_group))
            if reset:
                self.denominators = {}
                self.reduce_types = {}
                self.stats = defaultdict(list)
            results = {
                k: v.cpu().item() if torch.is_tensor(v) else v
                for k, v in results.items()
            }
            return results

    def _aggregate(self, key, reduce_group):
        reduce_type = self.reduce_types.get(key, ReduceType.SCALAR)

        result = {}
        if reduce_type == ReduceType.AVG_MIN_MAX:
            result["/".join([key, "avg"])] = self._avg_of(key, reduce_group)
            result["/".join([key, "min"])] = self._min_of(key, reduce_group)
            result["/".join([key, "max"])] = self._max_of(key, reduce_group)
        elif reduce_type == ReduceType.AVG:
            result[key] = self._avg_of(key, reduce_group)
        elif reduce_type == ReduceType.SUM:
            result[key] = self._sum_of(key, reduce_group)
        elif reduce_type == ReduceType.MIN:
            result[key] = self._min_of(key, reduce_group)
        elif reduce_type == ReduceType.MAX:
            result[key] = self._max_of(key, reduce_group)
        elif reduce_type == ReduceType.SCALAR:
            if torch.cuda.is_initialized():
                device = "cuda"
            else:
                device = "cpu"
            value = torch.tensor(
                sum(self.stats[key]), dtype=torch.float32, device=device
            )
            cnt = torch.tensor(len(self.stats[key]), dtype=torch.float32, device=device)
            if reduce_group is not None:
                dist.all_reduce(value, group=reduce_group)
                dist.all_reduce(cnt, group=reduce_group)
            result[key] = float(value / cnt)
        else:
            raise ValueError(f"Unknown reduce type: {reduce_type}")

        keys_to_pop = [k for k, v in result.items() if v is None]
        for k in keys_to_pop:
            result.pop(k)
        return result

    def _sum_of(self, key, reduce_group):
        values = self.stats[key]
        if key not in self.denominators:
            x = sum([x.sum() for x in values])
            if reduce_group is not None:
                dist.all_reduce(x, group=reduce_group)
        else:
            denominator = self.denominators[key]
            if denominator not in self.stats:
                raise ValueError(
                    f"Denominator `{denominator}` not set for key `{key}`."
                )
            xs = []
            for v, d in zip(values, self.stats[denominator]):
                xs.append(torch.where(d, v, 0.0).sum())
            x = sum(xs)
            if reduce_group is not None:
                dist.all_reduce(x, group=reduce_group)
        return float(x)

    def _avg_of(self, key, reduce_group):
        values = self.stats[key]
        denominator = self.denominators[key]
        if denominator not in self.stats:
            raise ValueError(f"Denominator `{denominator}` not set for key `{key}`.")
        xs = []
        ds = []
        for v, d in zip(values, self.stats[denominator]):
            xs.append(torch.where(d, v, 0.0).sum())
            ds.append(d.sum())
        x = sum(xs)
        d = sum(ds)
        if reduce_group is not None:
            dist.all_reduce(x, group=reduce_group)
            dist.all_reduce(d, group=reduce_group)
        if d == 0:
            return None
        return x / d

    def _min_of(self, key, reduce_group):
        values = self.stats[key]
        denominator = self.denominators[key]
        if denominator not in self.stats:
            raise ValueError(f"Denominator `{denominator}` not set for key `{key}`.")
        xs = []
        for v, d in zip(values, self.stats[denominator]):
            xs.append(torch.where(d, v, float("inf")).min())
        x = min(xs)
        if reduce_group is not None:
            dist.all_reduce(x, group=reduce_group, op=dist.ReduceOp.MIN)
        if torch.isinf(x):
            return None
        return float(x)

    def _max_of(self, key, reduce_group):
        values = self.stats[key]
        denominator = self.denominators[key]
        if denominator not in self.stats:
            raise ValueError(f"Denominator `{denominator}` not set for key `{key}`.")
        xs = []
        for v, d in zip(values, self.stats[denominator]):
            xs.append(torch.where(d, v, -float("inf")).max())
        x = max(xs)
        if reduce_group is not None:
            dist.all_reduce(x, group=reduce_group, op=dist.ReduceOp.MAX)
        if torch.isinf(x):
            return None
        return float(x)


DEFAULT_TRACKER = DistributedStatsTracker()
stat = DEFAULT_TRACKER.stat
denominator = DEFAULT_TRACKER.denominator
export = DEFAULT_TRACKER.export
scope = DEFAULT_TRACKER.scope
scalar = DEFAULT_TRACKER.scalar
record_timing = DEFAULT_TRACKER.record_timing

TRACKERS = {"": DEFAULT_TRACKER}
LOCK = Lock()


def get(name: str = ""):
    global TRACKERS, LOCK
    with LOCK:
        if name not in TRACKERS:
            TRACKERS[name] = DistributedStatsTracker(name)
        return TRACKERS[name]


def export_all(reduce_group=None, reset=True) -> Dict[str, float]:
    stat = {}
    duplicate_keys = set()
    for tracker in TRACKERS.values():
        x = tracker.export(reduce_group=reduce_group, reset=reset)
        for k in x.keys():
            if k in stat:
                duplicate_keys.add(k)
        stat.update(x)
    if duplicate_keys:
        logger.warning(f"Duplicate stat keys detected: {list(duplicate_keys)}")
    return stat
