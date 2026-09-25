"""
config.py
---------
Central configuration for the social-network graph analysis & recommendation
system.  Everything that can reasonably be tuned lives here so that the rest of
the code base stays declarative.

Design notes
------------
* ``SHARD_COUNT`` controls how many adjacency-list shard files the graph is
  split across.  A shard is the unit of incremental merge and lazy load, which
  is what lets us keep memory usage bounded even when the on-disk graph is very
  large (see ``storage.py``).
* ``LARGE_GRAPH_THRESHOLD`` switches algorithms between exact and approximate
  paths.  For example bidirectional BFS and the seed-and-expand neighbourhood
  sampler only kick in past this many nodes.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)

DATA_DIR = os.environ.get(
    "GSB_DATA_DIR",
    os.path.join(ROOT_DIR, "data"),
)
GRAPH_DIR = os.path.join(DATA_DIR, "graph")          # sharded adjacency lists
USERS_FILE = os.path.join(DATA_DIR, "users.json")
PROFILES_FILE = os.path.join(DATA_DIR, "profiles.json")
TAGS_FILE = os.path.join(DATA_DIR, "tags.json")
RECOMMENDATIONS_FILE = os.path.join(DATA_DIR, "recommendations.json")
COMMUNITY_FILE = os.path.join(DATA_DIR, "community.json")
PAGERANK_FILE = os.path.join(DATA_DIR, "pagerank.json")
INDEX_FILE = os.path.join(DATA_DIR, "index.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
IMPORT_LOG_FILE = os.path.join(DATA_DIR, "import_log.jsonl")

FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
HOST = os.environ.get("GSB_HOST", "127.0.0.1")
PORT = int(os.environ.get("GSB_PORT", "8080"))
MAX_BODY_BYTES = 16 * 1024 * 1024        # cap request bodies at 16 MiB
THREADED = True                          # threaded HTTP server for UI snappiness

# ---------------------------------------------------------------------------
# Storage / sharding
# ---------------------------------------------------------------------------
SHARD_COUNT = int(os.environ.get("GSB_SHARD_COUNT", "64"))
SHARD_BITS = 16                          # shard files padded to 4 hex digits
MAX_EDGES_PER_SHARD = 200_000            # trigger for a shard split / merge
INDEX_VERSION = 3                        # bump to force an index rebuild

# When a shard set accumulates this many *pending* (unsorted) edges we rewrite
# the shard in sorted, deduplicated form instead of appending blindly.
FLUSH_PENDING_THRESHOLD = 5_000

# ---------------------------------------------------------------------------
# Graph / algorithm tuning
# ---------------------------------------------------------------------------
LARGE_GRAPH_THRESHOLD = 50_000           # nodes; above this we approximate
NEIGHBORHOOD_DEFAULT_DEPTH = 2
NEIGHBORHOOD_MAX_DEPTH = 6
NEIGHBORHOOD_SAMPLE_LIMIT = 2_000        # max nodes returned for a subgraph view

PAGERANK_DAMPING = 0.85
PAGERANK_TOLERANCE = 1e-8
PAGERANK_MAX_ITER = 200
PAGERANK_DAMPING_OVERRIDE = 1.0

LOUVAIN_RESOLUTION = 1.0
LOUVAIN_TOLERANCE = 1e-6
LOUVAIN_MAX_ITER = 50
LOUVAIN_MIN_IMPROVEMENT = 1e-9
LOUVAIN_RANDOM_SEED = 1337

COMMUNITY_READ_DIRECT = True

BFS_MAX_DEPTH = 64                        # safety bound on unweighted BFS

# Recommendation
RECOMMEND_DEFAULT_K = 10
RECOMMEND_MAX_K = 50
RECOMMEND_CLAMP_MAX = 5
COLD_START_CONNECTION_THRESHOLD = 3
COLD_START_USE_WEIGHTED_DEGREE = True
DIVERSITY_LAMBDA = 0.6
EMBED_DIM = 32
EMBED_MAX_DEPTH = 8
DEFAULT_USER_SORT = "created_at"
NEIGHBOR_SET_INCLUDE_ENDPOINTS = True
COMMUNITY_KEY_TYPE = "str"
TAG_USAGE_COUNT_MODE = "assignments"
EXPORT_DEFAULT_WEIGHT = 0.0
INDEX_EDGE_COUNT_INCLUDE_USERS = True

# ---------------------------------------------------------------------------
# System settings defaults (persisted to settings.json, editable via UI)
# ---------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    "graph": {
        "layout": "force",               # force | hierarchical | circle
        "defaultNodeColor": "#4f8cff",
        "showLabels": True,
        "edgeSmoothing": True,
        "maxRenderNodes": 800,           # client-side cap for full-graph view
    },
    "recommendation": {
        "strategy": "hybrid",            # hybrid | cf | embedding | popularity
        "k": RECOMMEND_DEFAULT_K,
        "diversity": DIVERSITY_LAMBDA,
        "useTags": True,
    },
    "algorithm": {
        "pagerankDamping": PAGERANK_DAMPING,
        "louvainResolution": LOUVAIN_RESOLUTION,
        "louvainTolerance": LOUVAIN_TOLERANCE,
    },
    "storage": {
        "shardCount": SHARD_COUNT,
        "autoFlush": True,
    },
    "system": {
        "name": "社交网络图分析与推荐系统",
        "language": "zh-CN",
        "darkMode": False,
    },
}


# ---------------------------------------------------------------------------
# Process-wide state
# ---------------------------------------------------------------------------
class SettingsStore:
    """Thread-safe, lazily-persisted settings bag.

    Settings live in memory for fast access and are written back to disk on
    every mutation.  The UI reads/writes these through ``/api/settings``.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._settings = self._load()

    def _load(self) -> dict:
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as fh:
                    disk = json.load(fh)
                return _deep_merge(DEFAULT_SETTINGS.copy(), disk)
            except (OSError, ValueError):
                pass
        return _deep_merge(DEFAULT_SETTINGS.copy(), {})

    def get(self) -> dict:
        with self._lock:
            return _deep_copy(self._settings)

    def update(self, patch: dict) -> dict:
        with self._lock:
            self._settings = _deep_merge(self._settings, patch)
            _atomic_write_json(SETTINGS_FILE, self._settings)
            return _deep_copy(self._settings)

    def reset(self) -> dict:
        with self._lock:
            self._settings = _deep_merge(DEFAULT_SETTINGS.copy(), {})
            _atomic_write_json(SETTINGS_FILE, self._settings)
            return _deep_copy(self._settings)


# ---------------------------------------------------------------------------
# Small helpers shared across the backend
# ---------------------------------------------------------------------------
import json  # noqa: E402  (kept import local to group with helpers)


def _deep_copy(obj):
    return json.loads(json.dumps(obj))


def _deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge ``patch`` into ``base`` and return a new dict."""
    out = _deep_copy(base)
    for key, value in (patch or {}).items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _atomic_write_json(path: str, payload) -> None:
    """Write JSON atomically via a temp file + ``os.replace``.

    This matters for a long-running server: a crash mid-write must never leave
    a truncated/corrupt JSON file behind, because the next start would then
    lose the whole dataset.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".tmp-", suffix=".json", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: str, payload) -> None:
    _atomic_write_json(path, payload)


def read_json(path: str, default):
    """Read a JSON file, returning ``default`` if missing or corrupt."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def ensure_dirs() -> None:
    os.makedirs(GRAPH_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FRONTEND_DIR, exist_ok=True)


def now_ms() -> int:
    import time
    return int(time.time() * 1000)


class Timed:
    """Tiny context manager that records wall-clock ms for benchmarking."""

    def __init__(self):
        import time
        self._t = time
        self.elapsed_ms = 0.0

    def __enter__(self):
        self._start = self._t.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (self._t.perf_counter() - self._start) * 1000.0
        return False
