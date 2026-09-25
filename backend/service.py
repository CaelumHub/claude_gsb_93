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
