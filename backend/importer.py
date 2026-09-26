"""
importer.py
-----------
Edge-list **pre-check** (dry run) for the bulk relationship import.

Why this module exists
~~~~~~~~~~~~~~~~~~~~~~
The import flow is split in two phases:

1. pre-check -- :func:`classify_edges` categorises every parsed row against a
   read-only snapshot of the current graph (registered users, existing edges);
2. commit     -- the *very same function* runs against the snapshot taken at
   write time, so the reported categories and the rows actually persisted can
   never drift apart ("预检分类准确、和最终写入一致").

Categories (one row may match several)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
================  ===========================================================
invalid           line could not be parsed (not an edge at all)
self_loop         u == v, never written
missing_user      an endpoint has no user record in ``users.json``
duplicate         edge already exists in the graph, or is repeated inside
                  this very batch (only the first occurrence can take effect)
isolated          a warning: an endpoint keeps degree 0 after the write
                  (its only referencing row is rejected, e.g. a self loop on
                  a brand-new user or an edge to a non-existent user)
================  ===========================================================

The pre-check result is persisted as a reusable **ticket** (one JSON file per
ticket).  ``commit`` only accepts a ticket whose graph-state fingerprint still
matches the snapshot it was built from; any import/user change in between
invalidates the ticket instead of silently writing stale data.

Everything here is pure stdlib -- hashlib for the fingerprint, secrets for the
ticket id, no third-party dependency.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import threading
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    from . import config
except ImportError:  # pragma: no cover
    import config


# A row is one parsed, structurally valid edge.
Row = Dict[str, Any]

# Maximum detail rows kept *inside the ticket response* per category.  The full
# accepted row list is always stored in the ticket for commit regardless.
PREVIEW_DETAIL_LIMIT = 100

# Reason codes (also used by the frontend to render Chinese labels).
REASON_SELF_LOOP = "self_loop"
REASON_MISSING_USER = "missing_user"
REASON_DUP_EXISTING = "duplicate_existing"
REASON_DUP_BATCH = "duplicate_batch"
REASON_ISOLATED = "isolated"

BLOCKING_REASONS = (REASON_SELF_LOOP, REASON_MISSING_USER,
                    REASON_DUP_EXISTING, REASON_DUP_BATCH)

_SEP = re.compile(r"[\s,;\t]+")


# ---------------------------------------------------------------------------
# Parsing -- the server owns parsing so pre-check and commit see identical rows
# ---------------------------------------------------------------------------
def parse_edge_text(text: str) -> Tuple[List[Row], List[dict]]:
    """Parse edge-list text into ``(rows, invalid)``.

    Grammar (mirrors the import page): one edge per non-empty line,
    ``u v [weight]`` separated by spaces / commas / semicolons / tabs; lines
    starting with ``#`` are comments.  A malformed line is collected in
    ``invalid`` instead of aborting the whole batch.
    """
    rows: List[Row] = []
    invalid: List[dict] = []
    for lineno, raw in enumerate((text or "").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in _SEP.split(line) if p]
        if len(parts) < 2:
            invalid.append({"line": lineno, "raw": line, "reason": "字段不足，需要至少两个端点"})
            continue
        try:
            u = int(parts[0])
        except ValueError:
            u = None
        try:
            v = int(parts[1])
        except ValueError:
            v = None
        weight = 1.0
        if len(parts) >= 3:
            try:
                weight = float(parts[2])
            except ValueError:
                weight = 1.0  # be lenient about the optional weight, like the UI
        if u is None or v is None:
            invalid.append({"line": lineno, "raw": line, "reason": "端点必须是整数 ID"})
            continue
        if u < 0 or v < 0:
            invalid.append({"line": lineno, "raw": line, "reason": "用户 ID 不能为负数"})
            continue
        rows.append({"line": lineno, "raw": line, "u": u, "v": v, "w": weight})
    return rows, invalid


def rows_from_edges(edges: Iterable) -> Tuple[List[Row], List[dict]]:
    """Adapt a JSON ``edges: [[u, v, w], ...]`` payload to parsed rows."""
    rows: List[Row] = []
    invalid: List[dict] = []
    for lineno, edge in enumerate(edges or [], 1):
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            invalid.append({"line": lineno, "raw": str(edge), "reason": "边必须是 [u, v, 权重?] 数组"})
            continue
        try:
            u, v = int(edge[0]), int(edge[1])
        except (TypeError, ValueError):
            invalid.append({"line": lineno, "raw": str(edge), "reason": "端点必须是整数 ID"})
            continue
        weight = 1.0
        if len(edge) > 2:
            try:
                weight = float(edge[2])
            except (TypeError, ValueError):
                weight = 1.0
        if u < 0 or v < 0:
            invalid.append({"line": lineno, "raw": str(edge), "reason": "用户 ID 不能为负数"})
            continue
        rows.append({"line": lineno, "raw": f"{u} {v} {weight}", "u": u, "v": v, "w": weight})
    return rows, invalid


# ---------------------------------------------------------------------------
# Snapshot fingerprint -- detects "graph changed between pre-check and commit"
# ---------------------------------------------------------------------------
def snapshot_fingerprint(users: Set[int], edges: Set[Tuple[int, int]]) -> str:
    """Order-independent SHA-256 over registered users and existing edges."""
    digest = hashlib.sha256()
    for u, v in sorted(edges):
        digest.update(f"{u},{v};".encode("utf-8"))
    for uid in sorted(users):
        digest.update(f"@{uid};".encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Classification -- the single source of truth shared by pre-check & commit
# ---------------------------------------------------------------------------
def _norm(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def classify_edges(
    rows: List[Row],
    invalid: List[dict],
    snapshot: dict,
    detail_limit: int = PREVIEW_DETAIL_LIMIT,
) -> Tuple[dict, List[Row]]:
    """Classify parsed rows against a graph snapshot.

    ``snapshot`` must contain ``users`` (set of registered ids), ``edges`` (set
    of normalised ``(min, max)`` tuples) and ``nodes`` (set of ids currently in
    the graph).

    Returns ``(report, accepted_rows)`` where ``report`` carries the summary
    counters and capped detail previews, and ``accepted_rows`` is the exact
    list of edges that may be written (in input order).
    """
    users: Set[int] = snapshot["users"]
    existing: Set[Tuple[int, int]] = snapshot["edges"]
    nodes: Set[int] = snapshot["nodes"]

    # Annotate every parsed row in input order; the annotation is what the
    # second pass reads for detail rendering.
    annotated: List[dict] = []
    accepted: List[Row] = []
    seen_in_batch: Set[Tuple[int, int]] = set()
    incident: Counter = Counter()          # accepted-edge incidence per node

    n_self = n_missing = n_dup = n_dup_existing = n_dup_batch = 0

    for row in rows:
        u, v = row["u"], row["v"]
        reasons: List[str] = []
        if u == v:
            reasons.append(REASON_SELF_LOOP)
            n_self += 1
        missing = [x for x in (u, v) if x not in users]
        if missing:
            reasons.append(REASON_MISSING_USER)
            n_missing += 1
        key = _norm(u, v)
        if u != v:
            if key in existing:
                reasons.append(REASON_DUP_EXISTING)
                n_dup += 1
                n_dup_existing += 1
            elif key in seen_in_batch:
                reasons.append(REASON_DUP_BATCH)
                n_dup += 1
                n_dup_batch += 1
            seen_in_batch.add(key)
        if not reasons:
            accepted.append(row)
            incident[u] += 1
            incident[v] += 1
        annotated.append({"row": row, "reasons": reasons, "missing": missing})

    # A node is "left isolated" when it is not in the graph today and no
    # accepted edge in the batch touches it.  Accepted rows always add
    # incidence, so this flag only appears on rejected rows (self loops,
    # missing-user edges, ...) -- exactly the rows that *would* have stranded
    # the endpoint.
    isolated_nodes: Set[int] = set()
    details: Dict[str, List[dict]] = {
        "accepted": [], "self_loops": [], "duplicates": [],
        "missing_users": [], "isolated": [],
    }
    n_isolated_rows = 0

    for ann in annotated:
        row, reasons = ann["row"], ann["reasons"]
        u, v = row["u"], row["v"]
        iso = sorted({x for x in (u, v) if x not in nodes and incident[x] == 0})
        # Display-only: "isolated" never blocks a row (acceptance was decided
        # above), it just cross-labels the row in every detail bucket.
        display_reasons = list(reasons)
        if iso:
            display_reasons.append(REASON_ISOLATED)
        item = {
            "line": row["line"], "u": u, "v": v, "w": row["w"],
            "reasons": display_reasons,
        }
        if not reasons:
            details["accepted"].append(item)
        if REASON_SELF_LOOP in reasons:
            details["self_loops"].append(item)
        if REASON_DUP_EXISTING in reasons or REASON_DUP_BATCH in reasons:
            details["duplicates"].append(item)
        if REASON_MISSING_USER in reasons:
            details["missing_users"].append({**item, "missing": ann["missing"]})
        if iso:
            n_isolated_rows += 1
            isolated_nodes.update(iso)
            details["isolated"].append({**item, "endpoints": iso})

    invalid_items = [dict(item) for item in invalid]

    total_rows = len(rows) + len(invalid)
    summary = {
        "total_rows": total_rows,
        "parsed": len(rows),
        "accepted": len(accepted),
        "invalid": len(invalid),
        "self_loops": n_self,
        "missing_users": n_missing,
        "duplicates": n_dup,
        "duplicate_existing": n_dup_existing,
        "duplicate_batch": n_dup_batch,
        "isolated_rows": n_isolated_rows,
        "isolated_nodes": len(isolated_nodes),
    }

    capped: Dict[str, dict] = {}
    for name, items in details.items():
        capped[name] = {
            "total": len(items),
            "has_more": len(items) > detail_limit,
            "items": items[:detail_limit],
        }
    capped["invalid"] = {
        "total": len(invalid_items),
        "has_more": len(invalid_items) > detail_limit,
        "items": invalid_items[:detail_limit],
    }

    report = {"summary": summary, "details": capped}
    return report, accepted


# ---------------------------------------------------------------------------
# Reusable pre-check tickets (persisted to data/import_preview/)
# ---------------------------------------------------------------------------
class PreviewError(Exception):
    """Base class for ticket lifecycle errors."""


class PreviewMissing(PreviewError):
    """Ticket does not exist (or has expired)."""


class PreviewStale(PreviewError):
    """Graph state changed since the pre-check; the ticket is unusable."""


class PreviewStore:
    """File-backed store of pre-check tickets with TTL.

    One ticket per file keeps pre-check results reusable across UI navigation
    and even server restarts; the fingerprint embedded in each ticket is what
    makes a restarted (or concurrently mutated) server reject stale commits.
    """

    def __init__(self, directory: Optional[str] = None, ttl_ms: int = 30 * 60 * 1000) -> None:
        self.dir = directory or config.IMPORT_PREVIEW_DIR
        self.ttl_ms = ttl_ms
        self._lock = threading.RLock()
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, ticket_id: str) -> str:
        return os.path.join(self.dir, f"{ticket_id}.json")

    def create(self, source: str, rows: List[Row], invalid: List[dict],
               report: dict, fingerprint: str) -> dict:
        with self._lock:
            self._sweep_locked()
            ticket_id = "pc_" + secrets.token_hex(8)
            now = config.now_ms()
            ticket = {
                "id": ticket_id,
                "source": source,
                "created_at": now,
                "expires_at": now + self.ttl_ms,
                "fingerprint": fingerprint,
                "rows": rows,
                "invalid": invalid,
                "report": report,
            }
            config.atomic_write_json(self._path(ticket_id), ticket)
            return ticket

    def get(self, ticket_id: str) -> dict:
        with self._lock:
            if not re.fullmatch(r"pc_[0-9a-f]+", ticket_id or ""):
                raise PreviewMissing("预检单不存在")
            path = self._path(ticket_id)
            data = config.read_json(path, None)
            if data is None:
                raise PreviewMissing("预检单不存在或已失效，请重新预检")
            if data.get("expires_at", 0) < config.now_ms():
                self.delete(ticket_id)
                raise PreviewMissing("预检单已过期，请重新预检")
            return data

    def delete(self, ticket_id: str) -> None:
        with self._lock:
            try:
                os.remove(self._path(ticket_id))
            except OSError:
                pass

    def _sweep_locked(self) -> None:
        """Best-effort removal of expired ticket files."""
        now = config.now_ms()
        try:
            names = os.listdir(self.dir)
        except OSError:
            return
        for name in names:
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.dir, name)
            data = config.read_json(path, None)
            if data is None or data.get("expires_at", 0) < now:
                try:
                    os.remove(path)
                except OSError:
                    pass
