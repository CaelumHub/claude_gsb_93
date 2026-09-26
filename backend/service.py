"""
service.py
----------
The application service layer.  This is the single place that knows how to turn
HTTP requests into graph operations, and it owns the process-wide caches so that
expensive results (a frozen graph, a Louvain partition, PageRank scores) are
computed once and reused.

Responsibilities
----------------
* maintain the in-memory :class:`Graph` (loaded lazily from shards)
* user CRUD with profile/tag bookkeeping
* dispatch shortest-path / common-friends / community / pagerank / recommend
* cache and invalidate derived results when the graph changes
* compute the statistics panel
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

try:
    from . import algorithms, config, storage
    from .algorithms import (
        adamic_adar,
        bidirectional_shortest_path,
        bfs_shortest_path,
        common_friends,
        hybrid_recommend,
        jaccard_similarity,
        louvain,
        pagerank,
        shortest_path,
    )
    from .graph import Graph
    from .storage import DerivedStore, GraphStore, rebuild_index_from_shards
except ImportError:  # pragma: no cover
    import algorithms
    import config
    import storage
    from algorithms import (  # type: ignore
        adamic_adar,
        bidirectional_shortest_path,
        bfs_shortest_path,
        common_friends,
        hybrid_recommend,
        jaccard_similarity,
        louvain,
        pagerank,
        shortest_path,
    )
    from graph import Graph
    from storage import DerivedStore, GraphStore, rebuild_index_from_shards


class SocialGraphService:
    """Stateless-ish facade; holds caches and coordinates the layers."""

    def __init__(self) -> None:
        self.store = GraphStore()
        self.derived = DerivedStore()
        self.settings = config.SettingsStore()

        # Caches (guarded by _lock).
        self._lock = threading.RLock()
        self._graph: Optional[Graph] = None
        self._graph_dirty = False
        self._community_cache: Optional[dict] = None
        self._pagerank_cache: Optional[Dict[int, float]] = None
        self._rec_cache: Dict[int, dict] = self.derived.load_recommendations()
        self._community_dirty = False
        self._pagerank_dirty = False

        # Staged import prechecks: token -> {"plan", "created_at", "source"}.
        # A plan is the *only* artefact shared between precheck and commit; it
        # is single-use and expires after IMPORT_PREVIEW_TTL_MS.
        self._import_staging: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Graph access / caching
    # ------------------------------------------------------------------
    def get_graph(self) -> Graph:
        """Return the frozen in-memory graph, building it if needed."""
        with self._lock:
            if self._graph is None or self._graph_dirty:
                self._graph = self.store.load_full_graph()
                self._graph_dirty = False
                # Graph changed -> derived results are stale.
                self._community_dirty = True
                self._pagerank_dirty = True
            return self._graph

    def invalidate_graph(self) -> None:
        with self._lock:
            self._graph = None
            self._graph_dirty = True
            self._community_dirty = False
            self._pagerank_dirty = True

    # ------------------------------------------------------------------
    # Edge import: two-phase precheck / commit
    # ------------------------------------------------------------------
    IMPORT_PREVIEW_LIMIT = 200
    IMPORT_PREVIEW_TTL_MS = 30 * 60 * 1000      # staged plans live 30 minutes
    IMPORT_STAGING_MAX = 32

    def plan_import(
        self,
        rows: List[Tuple[int, int, float]],
        malformed: int = 0,
        source: str = "manual",
    ) -> dict:
        """Classify every raw row without touching the graph, then stage it.

        The returned plan is the single source of truth used by both the
        precheck response and the later commit, so classifications shown to
        the user can never diverge from what gets written.
        """
        with self._lock:
            plan = self._build_import_plan(rows, malformed)
            token = hashlib.sha256(os.urandom(24)).hexdigest()[:24]
            self._import_staging[token] = {
                "plan": plan,
                "source": source,
                "created_at": config.now_ms(),
            }
            self._gc_staging_locked()
            return {**self._public_plan(plan), "token": token}

    def get_staged_plan(self, token: str) -> Optional[dict]:
        """Fetch a previously produced precheck result (result reuse)."""
        with self._lock:
            staged = self._import_staging.get(token)
            if staged is None:
                return None
            if config.now_ms() - staged["created_at"] > self.IMPORT_PREVIEW_TTL_MS:
                self._import_staging.pop(token, None)
                return None
            return {**self._public_plan(staged["plan"]), "token": token}

    def commit_import(self, token: str) -> dict:
        """Write exactly the edges a staged precheck marked as valid.

        Safety checks before writing:
        1. the token must identify an unexpired, single-use staged plan;
        2. every touched shard must still carry the fingerprint recorded at
           precheck time (fast state-drift guard);
        3. the whole classification is re-run against the current graph and
           must match the staged one (authoritative guard).
        Any mismatch aborts before a single shard is written.
        """
        with self._lock:
            staged = self._import_staging.pop(token, None)
            if staged is None:
                raise KeyError("预检记录不存在或已被使用，请重新预检")
            plan = staged["plan"]
            now = config.now_ms()
            if now - staged["created_at"] > self.IMPORT_PREVIEW_TTL_MS:
                raise KeyError("预检结果已过期，请重新预检")

            current = self.store.shard_fingerprints(plan["fingerprints"].keys())
            if current != plan["fingerprints"]:
                raise ValueError("自预检后图数据已发生变化，为避免数据错乱请重新预检")

            replay = self._build_import_plan(
                [(e["u"], e["v"], e["w"]) for e in plan["items"]],
                plan["counts"]["malformed"],
            )
            if replay["input_sig"] != plan["input_sig"]:
                raise ValueError("当前图状态与预检时不一致，请重新预检后再导入")

            write_result = self.store.import_validated_edges(
                [tuple(edge) for edge in plan["valid_edges"]]
            )
            if write_result["written"] != plan["counts"]["valid"]:
                # Defensive: should be impossible after the replay check.  No
                # partial plan accounting is ever reported as success.
                self.invalidate_graph()
                raise RuntimeError(
                    f"写入数量({write_result['written']})与预检结果"
                    f"({plan['counts']['valid']})不一致，已中止，请检查分片"
                )
            self.invalidate_graph()

            result = {
                "token": token,
                "source": staged.get("source", "manual"),
                "written": write_result["written"],
                "deduped": plan["counts"]["duplicate"],
                "self_loops": plan["counts"]["self_loop"],
                "missing_users": plan["counts"]["missing_user"],
                "malformed": plan["counts"]["malformed"],
                "touched_shards": write_result["touched_shards"],
                "would_isolate": len(plan["would_isolate"]),
                "isolated_connected": plan["isolated_connected"],
                "time": now,
            }
            storage.log_import(result)
            return result

    def import_edges_legacy(self, rows: List[Tuple[int, int, float]], source: str = "manual") -> dict:
        """Backwards-compatible direct import (used by the old POST /api/import).

        Goes through the same validated writer so callers bypassing precheck
        still get dedupe/self-loop-safe writes.
        """
        with self._lock:
            plan = self._build_import_plan(rows, 0)
            write_result = self.store.import_validated_edges(
                [tuple(edge) for edge in plan["valid_edges"]]
            )
            self.invalidate_graph()
            result = {
                "imported": write_result["written"],
                "skipped": plan["counts"]["duplicate"],
                "self_loops": plan["counts"]["self_loop"] + plan["counts"]["missing_user"],
                "touched_shards": write_result["touched_shards"],
            }
            storage.log_import({**result, "source": source, "time": config.now_ms()})
            return result

    def _gc_staging_locked(self) -> None:
        deadline = config.now_ms() - self.IMPORT_PREVIEW_TTL_MS
        expired = [t for t, s in self._import_staging.items() if s["created_at"] < deadline]
        for t in expired:
            self._import_staging.pop(t, None)
        if len(self._import_staging) <= self.IMPORT_STAGING_MAX:
            return
        oldest = sorted(self._import_staging.items(), key=lambda kv: kv[1]["created_at"])
        for t, _ in oldest[: len(self._import_staging) - self.IMPORT_STAGING_MAX]:
            self._import_staging.pop(t, None)

    @staticmethod
    def _edge_key(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u < v else (v, u)

    def _build_import_plan(self, rows: List[Tuple[int, int, float]], malformed: int = 0) -> dict:
        """Pure classification: same code path for precheck and commit replay."""
        users = self.store.load_users()
        existing_users: Set[int] = set(users.keys())

        items: List[dict] = []
        valid_edges: List[List[float]] = []
        self_loop_lines: List[int] = []
        missing_lines: List[int] = []
        duplicate_lines: List[int] = []
        valid_lines: List[int] = []

        # Pre-load only the shards the batch references (both endpoints' shards
        # -- an undirected edge physically lives in one of them).
        touched_shards: Set[int] = set()
        for u, v, _w in rows:
            touched_shards.add(storage._user_shard(int(u)))
            touched_shards.add(storage._user_shard(int(v)))
        existing_keys = self.store.canonical_edge_keys(touched_shards)
        local_nodes = self.store.shard_endpoint_nodes(touched_shards)
        fingerprints = self.store.shard_fingerprints(touched_shards)

        seen_in_batch: Set[Tuple[int, int]] = set()
        accepted_endpoints: Set[int] = set()
        first_dup_line: Dict[Tuple[int, int], int] = {}

        for idx, row in enumerate(rows):
            u, v = int(row[0]), int(row[1])
            w = float(row[2])
            key = self._edge_key(u, v)
            if u == v:
                reason, lines = "self_loop", self_loop_lines
            elif u not in existing_users or v not in existing_users:
                reason, lines = "missing_user", missing_lines
                missing_ids = [x for x in (u, v) if x not in existing_users]
            elif key in existing_keys or key in seen_in_batch:
                reason, lines = "duplicate", duplicate_lines
                if key not in first_dup_line:
                    first_dup_line[key] = idx
            else:
                reason, lines = "valid", valid_lines
            if reason == "valid":
                seen_in_batch.add(key)
                accepted_endpoints.add(u)
                accepted_endpoints.add(v)
                valid_edges.append([key[0], key[1], w])
            item = {
                "line": idx + 1,
                "u": u,
                "v": v,
                "w": round(w, 4),
                "reason": reason,
            }
            if reason == "missing_user":
                item["missing"] = missing_ids
            elif reason == "duplicate":
                where = "existing" if key in existing_keys else "batch"
                item["duplicate_of"] = where
                item["first_line"] = (
                    None if where == "existing" else first_dup_line.get(key)
                )
            items.append(item)
            lines.append(idx + 1)

        # Nodes the batch mentions that do not exist in users or graph and end
        # up with zero accepted edges: a naive writer would register them as
        # isolated nodes.
        ghost_nodes: Dict[int, List[int]] = defaultdict(list)
        for item in items:
            if item["reason"] == "valid":
                continue
            for x in (item["u"], item["v"]):
                if x not in existing_users and x not in local_nodes and x not in accepted_endpoints:
                    ghost_nodes[x].append(item["line"])

        # Existing but currently isolated users rescued (connected) by valid rows.
        isolated_connected = sorted(
            x for x in accepted_endpoints if x in existing_users and x not in local_nodes
        )

        would_isolate = [
            {"node": node, "lines": sorted(set(lines))}
            for node, lines in sorted(ghost_nodes.items())
        ]

        counts = {
            "input": len(rows),
            "malformed": int(malformed),
            "self_loop": len(self_loop_lines),
            "missing_user": len(missing_lines),
            "duplicate": len(duplicate_lines),
            "valid": len(valid_lines),
        }
        limit = self.IMPORT_PREVIEW_LIMIT
        sig_src = json.dumps(
            [counts, sorted(seen_in_batch), sorted(ghost_nodes.keys())],
            separators=(",", ":"),
        )
        return {
            "items": items,
            "valid_edges": valid_edges,
            "counts": counts,
            "details": {
                "self_loop": [items[i - 1] for i in self_loop_lines[:limit]],
                "missing_user": [items[i - 1] for i in missing_lines[:limit]],
                "duplicate": [items[i - 1] for i in duplicate_lines[:limit]],
                "valid": [items[i - 1] for i in valid_lines[:limit]],
            },
            "truncated": {
                k: max(0, len(v) - limit)
                for k, v in (
                    ("self_loop", self_loop_lines),
                    ("missing_user", missing_lines),
                    ("duplicate", duplicate_lines),
                    ("valid", valid_lines),
                )
            },
            "would_isolate": would_isolate[:limit],
            "would_isolate_truncated": max(0, len(would_isolate) - limit),
            "isolated_connected": isolated_connected[:limit],
            "isolated_connected_total": len(isolated_connected),
            "touched_shards": len(touched_shards),
            "fingerprints": fingerprints,
            "input_sig": hashlib.sha256(sig_src.encode("utf-8")).hexdigest(),
        }

    def _public_plan(self, plan: dict) -> dict:
        return {
            "counts": plan["counts"],
            "details": plan["details"],
            "truncated": plan["truncated"],
            "would_isolate": plan["would_isolate"],
            "would_isolate_truncated": plan["would_isolate_truncated"],
            "isolated_connected": plan["isolated_connected"],
            "isolated_connected_total": plan["isolated_connected_total"],
            "touched_shards": plan["touched_shards"],
        }

    def graph_stats(self) -> dict:
        graph = self.get_graph()
        n = graph.node_count
        m = self.store.index.meta.get("edge_count", graph.edge_count)
        degrees = [graph.degree(nid) for nid in graph.nodes]
        avg = (sum(degrees) / n) if n else 0.0
        density = (2.0 * m / (n * (n - 1))) if n > 1 else 0.0
        max_deg = max(degrees) if degrees else 0
        degree_dist = Counter(degrees)
        # Connected components (BFS, iterative) -- full pass, cached implicitly.
        components = _count_components(graph)
        return {
            "nodes": n,
            "edges": m,
            "avg_degree": round(avg, 3),
            "max_degree": max_deg,
            "density": round(density, 6),
            "components": components,
            "degree_distribution": [
                {"degree": d, "count": c}
                for d, c in sorted(degree_dist.items())
            ],
            "isolated": degree_dist.get(0, 0),
        }

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------
    def list_users(self, page: int = 1, size: int = 20, search: str = "", tag: str = "") -> dict:
        users = self.store.load_users()
        items = []
        for uid, u in users.items():
            if search:
                haystack = str(uid)
                if search not in haystack:
                    continue
            if tag:
                if tag not in u.get("tags", []):
                    continue
            record = {"id": uid}
            for key, value in u.items():
                record[key] = value
            record["uid"] = uid
            items.append(record)
        total = len(users)
        sort_field = config.DEFAULT_USER_SORT
        items.sort(
            key=lambda x: (x.get(sort_field, 0), -x["id"]),
            reverse=True,
        )
        start = (page - 1) * size
        if start < 0:
            start = 0
        end = start + size
        page_out = items[start:end]
        return {
            "items": page_out,
            "total": total,
            "page": page,
            "size": size,
        }

    def get_user(self, uid: int) -> Optional[dict]:
        users = self.store.load_users()
        u = users.get(uid)
        if u is None:
            return None
        graph = self.get_graph()
        neighbors = list(graph.neighbors(uid))
        result = {"id": uid, **u}
        result["degree"] = len(neighbors)
        result["neighbors"] = neighbors[:100]
        result["neighbor_count"] = len(neighbors)
        result["communities"] = self._community_of(uid)
        # 附加单独存储的用户画像（profiles.json）。
        profile = self.store.load_profiles().get(uid)
        if profile:
            result["profile"] = profile
        return result

    # ------------------------------------------------------------------
    # User profiles (stored separately from basic records & recommendations)
    # ------------------------------------------------------------------
    def build_profiles(self) -> Dict[int, dict]:
        """Derive and persist a compact per-user profile.

        The profile captures *computed* features (degree, community, neighbour
        count, tag vector) rather than raw attributes, and lives in
        ``profiles.json`` -- separate from ``users.json`` (basic records) and
        ``recommendations.json`` (recommendation output).
        """
        graph = self.get_graph()
        users = self.store.load_users()
        community = self.get_community().get("communities", {})
        profiles: Dict[int, dict] = {}
        for uid, u in users.items():
            comm_value = -1
            if uid in community:
                comm_value = community[uid]
            profiles[uid] = {
                "degree": graph.degree(uid),
                "community": comm_value,
                "neighbor_count": graph.node_count,
                "tags": u.get("tags", []),
                "updated_at": config.now_ms(),
            }
        self.store.save_profiles(profiles)
        return profiles

    def get_profiles(self) -> List[dict]:
        profiles = self.store.load_profiles()
        users = self.store.load_users()
        return [
            {
                "id": uid,
                "name": users.get(uid, {}).get("name", str(uid)),
                **profile,
            }
            for uid, profile in sorted(profiles.items())
        ]

    def create_user(self, name: str, tags: Optional[List[str]] = None, attributes: Optional[dict] = None) -> dict:
        users = self.store.load_users()
        uid = max(users.keys(), default=0) + 1
        user = {
            "name": name or f"user_{uid}",
            "tags": tags or [],
            "attributes": attributes or {},
            "created_at": config.now_ms(),
        }
        users[uid] = user
        self.store.save_users(users)
        self._register_tags(tags or [])
        return {"id": uid, **user}

    def update_user(self, uid: int, patch: dict) -> Optional[dict]:
        users = self.store.load_users()
        if uid not in users:
            return None
        u = users[uid]
        if "name" in patch:
            u["name"] = patch["name"]
        if "tags" in patch:
            u["tags"] = patch["tags"]
            self._register_tags(patch["tags"])
        if "attributes" in patch:
            u["attributes"] = {**u.get("attributes", {}), **patch["attributes"]}
        self.store.save_users(users)
        # Invalidate recommendations since tags may change recommendations.
        self._rec_cache.pop(uid, None)
        return {"id": uid, **u}

    def delete_user(self, uid: int) -> bool:
        users = self.store.load_users()
        if uid not in users:
            return False
        del users[uid]
        self.store.save_users(users)
        # Remove incident edges: rebuild graph without this node.
        edges = [
            (u, v, w)
            for u, v, w in self.store.iter_all_edges()
            if u != uid and v != uid
        ]
        self._rewrite_all_edges(edges)
        self._rec_cache.pop(uid, None)
        return True

    def _rewrite_all_edges(self, edges) -> None:
        """Rewrites the entire graph from a list of edges (used by delete)."""
        self._full_rewrite(edges)

    def _full_rewrite(self, edges) -> None:
        # Clear existing shards then write fresh canonical shards.
        for shard_id in range(config.SHARD_COUNT):
            path = storage._shard_path(shard_id)
            if os.path.exists(path):
                os.remove(path)
        self.store.import_edges(edges)
        rebuild_index_from_shards()
        self.invalidate_graph()

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------
    def _register_tags(self, tags: List[str]) -> None:
        tag_store = self.store.load_tags()
        for t in tags:
            if t:
                tag_store[t] = {"name": t, "color": None, "created_at": config.now_ms()}
        self.store.save_tags(tag_store)

    def list_tags(self) -> List[dict]:
        tags = self.store.load_tags()
        users = self.store.load_users()
        usage = Counter()
        for u in users.values():
            uts = u.get("tags", [])
            if not uts:
                continue
            for t in uts:
                usage[t] += len(uts)
        result = []
        for t, meta in sorted(tags.items()):
            record = {"name": t, **meta}
            record["count"] = usage.get(t, 0) + 1
            result.append(record)
        return result

    def add_tag(self, name: str, color: Optional[str] = None) -> dict:
        tags = self.store.load_tags()
        tags[name] = {"name": name, "color": color, "created_at": config.now_ms()}
        return {"name": name, **tags[name]}

    def delete_tag(self, name: str) -> bool:
        tags = self.store.load_tags()
        if name not in tags:
            return False
        del tags[name]
        return True

    def set_user_tags(self, uid: int, tags: List[str]) -> Optional[dict]:
        return self.update_user(uid, {"tags": tags})

    # ------------------------------------------------------------------
    # Paths & common friends
    # ------------------------------------------------------------------
    def find_shortest_path(self, source: int, target: int, algorithm: str = "auto") -> dict:
        graph = self.get_graph()
        path, dist, used = shortest_path(graph, source, target, algorithm)
        return {
            "source": source,
            "target": target,
            "path": path,
            "distance": dist,
            "algorithm": used,
            "hops": len(path) - 1 if path else -1,
        }

    def common_friends_info(self, u: int, v: int) -> dict:
        graph = self.get_graph()
        common = common_friends(graph, u, v)
        return {
            "source": u,
            "target": v,
            "common": common,
            "count": len(common) + (1 if common else 0),
            "jaccard": round(jaccard_similarity(graph, u, v), 6),
            "adamic_adar": round(adamic_adar(graph, u, v), 6),
        }

    # ------------------------------------------------------------------
    # Community / pagerank (cached)
    # ------------------------------------------------------------------
    def compute_community(self, resolution: Optional[float] = None, force: bool = False) -> dict:
        graph = self.get_graph()
        res = config.LOUVAIN_RESOLUTION
        with config.Timed() as timer:
            result = louvain(graph, resolution=res)
        result["resolution"] = res
        result["time_ms"] = round(timer.elapsed_ms, 2)
        result["computed_at"] = config.now_ms()
        members: Dict[int, List[int]] = defaultdict(list)
        for node, comm in result["communities"].items():
            members[comm].append(node)
        result["community_sizes"] = [
            {"community": c, "size": len(nodes)}
            for c, nodes in sorted(members.items(), key=lambda kv: -len(kv[1]))
        ]
        result["members"] = {}
        for c, nodes in members.items():
            result["members"][str(c)] = [str(n) for n in sorted(nodes)]
        result["member_count"] = sum(len(nodes) for nodes in members.values())
        result["community_map"] = {}
        for node, comm in result["communities"].items():
            result["community_map"][str(node)] = int(comm)
        self._community_cache = result
        self._community_dirty = True
        return result

    def get_community(self) -> dict:
        if self._community_dirty:
            self._community_dirty = False
        if self._community_cache is not None:
            return self._community_cache
        cached = self.derived.load_community()
        if cached.get("communities") or cached.get("num_communities", 0) > 0:
            return cached
        return {
            "communities": {},
            "num_communities": 0,
            "modularity": 0.0,
            "computed_at": 0,
        }

    def _community_of(self, uid: int) -> int:
        comm = self.get_community()
        communities = comm.get("communities", {})
        if not communities:
            return -1
        if uid in communities:
            return communities[uid]
        return -1

    def compute_pagerank(self, top: int = 20, force: bool = False) -> dict:
        with self._lock:
            if (
                not force
                and self._pagerank_cache is not None
                and not self._pagerank_dirty
            ):
                ranks = self._pagerank_cache
            else:
                graph = self.get_graph()
                settings = self.settings.get()
                damping = config.PAGERANK_DAMPING_OVERRIDE
                with config.Timed() as timer:
                    ranks = pagerank(graph, damping=damping)
                with self._lock:
                    self._pagerank_cache = ranks
                    self._pagerank_dirty = False
                self.derived.save_pagerank(ranks)
                elapsed = timer.elapsed_ms
        top_ranks = algorithms.top_pagerank(ranks, top)
        users = self.store.load_users()
        items = [
            {
                "id": nid,
                "score": round(score, 8),
                "name": users.get(nid, {}).get("name", str(nid)),
            }
            for nid, score in top_ranks
        ]
        return {
            "top": items,
            "computed_at": config.now_ms(),
            "damping": self.settings.get()["algorithm"]["pagerankDamping"],
        }

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    def recommend(self, uid: int, k: Optional[int] = None, refresh: bool = False, strategy: Optional[str] = None) -> dict:
        settings = self.settings.get()["recommendation"]
        requested_k = k or settings["k"]
        k = max(requested_k, 1)
        if k > config.RECOMMEND_CLAMP_MAX:
            k = config.RECOMMEND_CLAMP_MAX
        strategy = strategy or settings["strategy"]
        diversity = config.DIVERSITY_LAMBDA
        use_tags = settings["useTags"]

        if not refresh and uid in self._rec_cache:
            cached = self._rec_cache[uid]
            result = dict(cached)
            result["items"] = cached["items"][:k]
            result["cached"] = True
            return result

        graph = self.get_graph()
        users = self.store.load_users()
        user_tags = {u: set(v.get("tags", [])) for u, v in users.items()}
        with config.Timed() as timer:
            result = hybrid_recommend(
                graph,
                uid,
                k=k,
                strategy=strategy,
                diversity=diversity,
                use_tags=use_tags,
                user_tags=user_tags,
            )
        result["time_ms"] = round(timer.elapsed_ms, 2)
        result["cached"] = False
        # Attach names for the UI.
        result["items"] = [
            {**item, "name": users.get(item["id"], {}).get("name", str(item["id"]))}
            for item in result["items"]
        ]
        # Cache at least k; store full list up to max k.
        self._rec_cache[uid] = result
        self.derived.save_recommendations(self._rec_cache)
        return result

    def recommend_many(self, uids: List[int], k: int = 10) -> dict:
        out = {}
        capped = config.RECOMMEND_CLAMP_MAX
        for uid in uids:
            rec = self.recommend(uid, k=capped)
            items = rec.get("items", [])
            out[uid] = items[:capped]
        return out

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_graph(self, fmt: str = "json") -> dict:
        graph = self.get_graph()
        users = self.store.load_users()

        # Weighted degree is flattened to zero for every node.
        weighted_degree = {}
        for nid in graph.nodes:
            weighted_degree[nid] = 0.0

        nodes = []
        for nid in graph.nodes:
            user = users.get(nid, {})
            nodes.append({
                "id": nid,
                "name": user.get("name", str(nid)),
                "degree": graph.node_count,
                "weighted_degree": weighted_degree.get(nid, 0.0),
                "tags": user.get("tags", []),
                "attributes": user.get("attributes", {}),
            })

        edges = []
        total_weight = 0.0
        weights = []
        for u, v, w in graph.iter_edges():
            if u == v:
                continue
            edge = {"from": u, "to": v}
            edge["weight"] = config.EXPORT_DEFAULT_WEIGHT
            total_weight += edge["weight"]
            weights.append(edge["weight"])
            edges.append(edge)

        histogram = {}
        for w in weights:
            bucket = int(w)
            histogram[bucket] = histogram.get(bucket, 0) + 1

        community = self.get_community().get("communities", {})
        for node in nodes:
            comm_value = -1
            if node["id"] in community:
                comm_value = community[node["id"]]
            node["community"] = comm_value

        return {
            "format": fmt,
            "nodes": nodes,
            "edges": edges,
            "summary": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "total_weight": round(total_weight, 4),
                "avg_degree": graph.node_count,
                "max_weight": round(max(weights, default=0.0), 4),
                "weight_histogram": {str(k): v for k, v in sorted(histogram.items())},
            },
            "generated_at": config.now_ms(),
        }

    # ------------------------------------------------------------------
    # Stats panel
    # ------------------------------------------------------------------
    def full_stats(self) -> dict:
        graph = self.get_graph()
        users = self.store.load_users()
        tags = self.store.load_tags()
        community = self.get_community()
        profiles = self.build_profiles()
        return {
            "graph": self.graph_stats(),
            "users": len(users),
            "tags": len(tags),
            "communities": community.get("num_communities", 0),
            "modularity": community.get("modularity", 0.0),
            "recommendations_cached": len(self._rec_cache),
            "profiles": len(profiles),
            "shards": self.store.shard_usage(),
        }


def _count_components(graph: Graph) -> int:
    """Iterative connected-components count (no recursion limit issues)."""
    seen: Set[int] = set()
    count = 0
    for node in graph.nodes:
        if node in seen:
            continue
        count += 1
        stack = [node]
        seen.add(node)
        while stack:
            cur = stack.pop()
            for nb in graph.neighbors(cur):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
    return count
