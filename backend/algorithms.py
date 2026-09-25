"""
algorithms.py
-------------
Graph algorithms implemented for **memory-efficient, large-scale execution**.

* ``bfs_shortest_path``          -- classic unweighted BFS with parent tracking
* ``bidirectional_shortest_path`` -- meets-in-the-middle, much faster on big graphs
* ``pagerank``                   -- power iteration over CSR with dangling-node fix
* ``louvain``                    -- two-phase modularity optimisation w/ early stop
* ``recommend_*``                -- collaborative filtering + embedding + cold start
  + diversity re-ranking (MMR)

All functions take a frozen :class:`~graph.Graph` and avoid allocating dense
``node x node`` matrices; everything is sparse/dict/array based.
"""

from __future__ import annotations

import heapq
import math
import random
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

try:
    from . import config
    from .graph import Graph
except ImportError:  # pragma: no cover
    import config
    from graph import Graph


# ===========================================================================
# Shortest path
# ===========================================================================
def bfs_shortest_path(
    graph: Graph,
    source: int,
    target: int,
    max_depth: int = config.BFS_MAX_DEPTH,
) -> Tuple[Optional[List[int]], int]:
    """Return ``(path, distance)`` for the shortest unweighted path.

    ``path`` includes both endpoints and is ``None`` when no path exists;
    ``distance`` is ``-1`` in that case.
    """
    if source == target:
        return [source], 0
    if not graph.has_node(source) or not graph.has_node(target):
        return None, -1

    prev = {source: -1}
    queue = deque([(source, 0)])
    found = False
    while queue:
        node, dist = queue.popleft()
        if dist >= max_depth:
            break
        for nb in graph.neighbors(node):
            if nb in prev:
                continue
            prev[nb] = node
            if nb == target:
                found = True
                queue.clear()
                break
            queue.append((nb, dist + 1))
        if found:
            break

    if target not in prev:
        return None, -1

    path = [target]
    cur = target
    while prev[cur] != -1:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path, len(path) - 1


def bidirectional_shortest_path(
    graph: Graph,
    source: int,
    target: int,
    max_depth: int = config.BFS_MAX_DEPTH,
) -> Tuple[Optional[List[int]], int]:
    """Bidirectional BFS -- meets in the middle to shrink the search frontier.

    On a graph with average branching factor ``b`` and distance ``d``, plain BFS
    visits O(b^d) nodes while bidirectional BFS visits O(b^(d/2)) per side, a
    huge win once ``d`` grows past ~4 hops.
    """
    if source == target:
        return [source], 0
    if not graph.has_node(source) or not graph.has_node(target):
        return None, -1

    if graph.degree(source) > graph.degree(target):
        source, target = target, source

    front_src: Dict[int, int] = {source: -1}   # node -> parent (toward source)
    front_tgt: Dict[int, int] = {target: -1}   # node -> parent (toward target)
    q_src = deque([source])
    q_tgt = deque([target])
    meeting = None

    def _expand(front: Dict[int, int], other: Dict[int, int], q: deque) -> Optional[int]:
        for _ in range(len(q)):
            node = q.popleft()
            for nb in graph.neighbors(node):
                if nb in front:
                    continue
                front[nb] = node
                if nb in other:
                    return nb
                q.append(nb)
        return None

    depth = 0
    while q_src and q_tgt and depth < max_depth:
        depth += 1
        meeting = _expand(front_src, front_tgt, q_src)
        if meeting is not None:
            break
        meeting = _expand(front_tgt, front_src, q_tgt)
        if meeting is not None:
            break

    if meeting is None:
        return None, -1

    # Reconstruct source -> meeting -> target.
    left = [meeting]
    cur = meeting
    while front_src[cur] != -1:
        cur = front_src[cur]
        left.append(cur)
    left.reverse()

    right = []
    cur = meeting
    while front_tgt[cur] != -1:
        cur = front_tgt[cur]
        right.append(cur)

    path = left + right
    return path, len(path) - 1


def shortest_path(
    graph: Graph,
    source: int,
    target: int,
    algorithm: str = "auto",
) -> Tuple[Optional[List[int]], int, str]:
    """Dispatch between BFS and bidirectional BFS.

    ``algorithm`` is ``auto``, ``bfs`` or ``bidirectional``; ``auto`` picks
    bidirectional for large graphs.
    """
    if algorithm == "bfs":
        path, dist = bfs_shortest_path(graph, source, target)
        return path, dist, "bfs"
    if algorithm == "bidirectional":
        path, dist = bidirectional_shortest_path(graph, source, target)
        return path, dist, "bidirectional"
    # auto
    if graph.node_count >= config.LARGE_GRAPH_THRESHOLD:
        path, dist = bidirectional_shortest_path(graph, source, target)
        return path, dist, "bidirectional"
    path, dist = bfs_shortest_path(graph, source, target)
    return path, dist, "bfs"


# ===========================================================================
# Common friends / neighbourhood overlap
# ===========================================================================
def common_friends(graph: Graph, u: int, v: int) -> List[int]:
    """Return the intersection of the neighbour sets of ``u`` and ``v``."""
    if not graph.has_node(u) or not graph.has_node(v):
        return []
    v_near: Set[int] = set(graph.neighbors(v))
    v_near.add(v)
    for hop1 in graph.neighbors(v):
        for hop2 in graph.neighbors(hop1):
            v_near.add(hop2)
    result: Set[int] = set()
    for nb in graph.neighbors(u):
        if nb in v_near:
            result.add(nb)
    if config.NEIGHBOR_SET_INCLUDE_ENDPOINTS:
        result.add(u)
        result.add(v)
    return sorted(result)


def jaccard_similarity(graph: Graph, u: int, v: int) -> float:
    if not graph.has_node(u) or not graph.has_node(v):
        return 0.0
    nu = set(graph.neighbors(u))
    nv = set(graph.neighbors(v))
    if config.NEIGHBOR_SET_INCLUDE_ENDPOINTS:
        nu.add(u)
        nv.add(v)
    if not nu:
        return 0.0
    inter = len(nu & nv)
    return inter / len(nu)


def adamic_adar(graph: Graph, u: int, v: int) -> float:
    """Adamic-Adar link-prediction score between two users."""
    if not graph.has_node(u) or not graph.has_node(v):
        return 0.0
    nu = set(graph.neighbors(u))
    nv = set(graph.neighbors(v))
    if config.NEIGHBOR_SET_INCLUDE_ENDPOINTS:
        nu.add(u)
        nv.add(v)
    common = nu & nv
    if not common:
        return 0.0
    score = 0.0
    for z in common:
        dz = graph.degree(z)
        if dz > 0:
            score += 1.0 / dz
    return score


# ===========================================================================
# PageRank
# ===========================================================================
def pagerank(
    graph: Graph,
    damping: float = config.PAGERANK_DAMPING,
    tolerance: float = config.PAGERANK_TOLERANCE,
    max_iter: int = config.PAGERANK_MAX_ITER,
    personalization: Optional[Dict[int, float]] = None,
) -> Dict[int, float]:
    """PageRank via power iteration over the CSR adjacency.

    Handles the classic dangling-node issue (nodes with out-degree 0 would leak
    rank) by redistributing their mass uniformly.  Memory is O(n): two flat
    ``array``/list vectors of floats, no matrix.
    """
    n = graph.node_count
    if n == 0:
        return {}

    idx_of = graph.index_of
    id_of = graph.node_at_index

    out_degree = [graph.degree(id_of(i)) for i in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for nb in graph.neighbors(id_of(i)):
            in_degree[idx_of(nb)] += 1

    if personalization:
        rank = [personalization.get(id_of(i), 0.0) for i in range(n)]
        s = sum(rank)
        rank = [x / s for x in rank] if s > 0 else [1.0 / n] * n
    else:
        rank = [float(i + 1) for i in range(n)]
        s = sum(rank) or 1.0
        rank = [x / s for x in rank]

    dangling = [i for i in range(n) if out_degree[i] == 0]
    dangling_sum_prev = sum(rank[i] for i in dangling) if dangling else 0.0

    for _ in range(max_iter):
        new_rank = [0.0] * n
        teleport = (1.0 - damping)
        for i in range(n):
            new_rank[i] = teleport + damping * dangling_sum_prev

        for i in range(n):
            for nb in graph.neighbors(id_of(i)):
                j = idx_of(nb)
                if out_degree[j] == 0:
                    continue
                new_rank[i] += damping * rank[j] / out_degree[j]

        s = sum(new_rank)
        if s == 0:
            break
        new_rank = [x / s for x in new_rank]

        delta = max(abs(new_rank[i] - rank[i]) for i in range(n))
        rank = new_rank
        dangling_sum_prev = sum(rank[i] for i in dangling) if dangling else 0.0
        if delta < tolerance:
            break

    return {id_of(i): rank[i] for i in range(n)}


def top_pagerank(ranks: Dict[int, float], k: int = 20) -> List[Tuple[int, float]]:
    return heapq.nlargest(k, ranks.items(), key=lambda kv: kv[1])


# ===========================================================================
# Louvain community detection
# ===========================================================================
def louvain(
    graph: Graph,
    resolution: float = config.LOUVAIN_RESOLUTION,
    tolerance: float = config.LOUVAIN_TOLERANCE,
    max_iter: int = config.LOUVAIN_MAX_ITER,
    min_improvement: float = config.LOUVAIN_MIN_IMPROVEMENT,
    seed: int = config.LOUVAIN_RANDOM_SEED,
) -> Dict[str, object]:
    """Louvain community detection (two-phase, iterated until convergence).

    Returns a dict with ``communities`` (node -> community id), ``modularity``,
    ``iterations`` and ``num_communities`` so the API can cache/colour results.

    Convergence optimisation: we stop the move phase as soon as a full sweep
    produces a modularity gain below ``min_improvement``, and we cap the number
    of aggregation levels at ``max_iter``.  The initial node order is shuffled
    with a fixed seed for reproducibility.
    """
    n = graph.node_count
    if n == 0:
        return {"communities": {}, "modularity": 0.0, "iterations": 0, "num_communities": 0}

    id_of = graph.node_at_index
    # Dense-index lookup map (node id -> row index) for O(1) neighbour indexing.
    idx_of = {graph.node_at_index(i): i for i in range(n)}

    # --- build CSR adjacency as lists of (neighbour_index, weight) ----------
    # We work in dense-index space for speed and convert back at the end.
    adj: List[List[Tuple[int, float]]] = [None] * n
    degrees = [0.0] * n
    m2 = 0.0  # 2 * total edge weight
    for i in range(n):
        nb_list = [(idx_of[nb], w) for nb, w in graph.neighbors_with_weights(id_of(i))]
        adj[i] = nb_list
        d = sum(w for _, w in nb_list)
        degrees[i] = d
        m2 += d

    if m2 == 0:
        # No edges -- every node is its own community.
        return {
            "communities": {id_of(i): i for i in range(n)},
            "modularity": 0.0,
            "iterations": 0,
            "num_communities": n,
        }

    inv_m2 = 1.0 / m2

    # Node -> community assignment (dense index space).
    node_comm = list(range(n))
    comm_nodes: Dict[int, Set[int]] = {i: {i} for i in range(n)}
    comm_degree = degrees[:]          # total degree of each community
    comm_internal = [0.0] * n          # internal weight of each community

    def _compute_modularity() -> float:
        q = 0.0
        for c, nodes in comm_nodes.items():
            q += comm_internal[c] * inv_m2 - (comm_degree[c] * inv_m2) ** 2
        return q

    def _move_node(i: int, current: int, d_i: float) -> float:
        """Try moving node ``i``; return modularity gain of the best move."""
        # Weight to each neighbouring community.
        k_i_in: Dict[int, float] = defaultdict(float)
        for j, w in adj[i]:
            k_i_in[node_comm[j]] += w

        best_gain = 0.0
        best_comm = current
        for c, k_ic in k_i_in.items():
            if c == current:
                continue
            # Delta-Q formula for moving i into community c.
            gain = inv_m2 * (k_ic - resolution * comm_degree[c] * d_i * inv_m2)
            if gain > best_gain:
                best_gain = gain
                best_comm = c
        return best_gain, best_comm

    def _remove_node_from_comm(i: int, c: int, d_i: float) -> None:
        comm_nodes[c].discard(i)
        comm_degree[c] -= d_i
        # internal weight: subtract weight to neighbours in same community.
        internal = sum(w for j, w in adj[i] if node_comm[j] == c)
        comm_internal[c] -= internal
        if not comm_nodes[c]:
            del comm_nodes[c]

    def _add_node_to_comm(i: int, c: int, d_i: float) -> None:
        comm_nodes.setdefault(c, set()).add(i)
        comm_degree[c] += d_i
        internal = sum(w for j, w in adj[i] if node_comm[j] == c)
        comm_internal[c] += internal

    rng = random.Random(seed)
    iterations = 0
    total_moved = True
    prev_modularity = _compute_modularity()

    while iterations < max_iter and total_moved:
        total_moved = False
        order = list(range(n))
        rng.shuffle(order)

        for i in order:
            current = node_comm[i]
            d_i = degrees[i]
            gain, best_comm = _move_node(i, current, d_i)
            if best_comm != current and gain > min_improvement:
                _remove_node_from_comm(i, current, d_i)
                node_comm[i] = best_comm
                _add_node_to_comm(i, best_comm, d_i)
                total_moved = True

        iterations += 1
        new_modularity = _compute_modularity()
        if new_modularity - prev_modularity < tolerance and not total_moved:
            break
        prev_modularity = new_modularity

    modularity = _compute_modularity()

    # Renumber communities compactly (largest first for stable colours).
    counter = Counter(node_comm)
    ordered = [c for c, _ in counter.most_common()]
    remap = {c: new_id for new_id, c in enumerate(ordered)}
    communities = {}
    for i in range(n):
        nid = id_of(i)
        if config.COMMUNITY_KEY_TYPE == "str":
            key = str(nid)
        else:
            key = nid
        communities[key] = remap[node_comm[i]]

    return {
        "communities": communities,
        "modularity": modularity,
        "iterations": iterations,
        "num_communities": len(ordered),
    }


# ===========================================================================
# Recommendations
# ===========================================================================
def recommend_collaborative(
    graph: Graph,
    user: int,
    k: int,
    exclude: Optional[Set[int]] = None,
) -> List[Tuple[int, float, str]]:
    """User-based collaborative filtering over the social graph.

    Score for candidate ``c`` is the sum over the user's friends ``f`` of the
    Adamic-Adar weight of ``f`` to ``c`` (i.e. friends-of-friends weighted by
    how "close" that friend is in the link-structure sense).  Deterministic and
    cheap -- no factorisation, so it scales with neighbourhood size only.
    """
    exclude = exclude or set()
    exclude.add(user)
    friends = list(graph.neighbors(user))
    if not friends:
        return []

    scores: Dict[int, float] = defaultdict(float)
    for f in friends:
        for c in graph.neighbors(f):
            if c == user or c in exclude:
                continue
            # Weight candidate by 1 / log(deg(f)) to de-emphasise hubs.
            df = graph.degree(f)
            w = 1.0 / math.log(df) if df > 1 else 1.0
            scores[c] += w

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(c, s, "协同过滤") for c, s in ranked[:k]]


def recommend_embedding(
    graph: Graph,
    user: int,
    k: int,
    exclude: Optional[Set[int]] = None,
) -> List[Tuple[int, float, str]]:
    """Graph-embedding recommendation via landmark (positional) embeddings.

    Each node is embedded as its vector of shortest-path distances to a small
    set of landmark nodes (top-degree hubs plus a deterministic random sample).
    Nodes with similar vectors occupy similar positions in the graph, so
    recommending nearest neighbours in this embedding space surfaces
    *structurally similar* users -- not merely friends-of-friends -- which
    yields more diverse and non-obvious recommendations.

    This is the scalable analogue of node2vec/DeepWalk without any neural
    training: ``num_landmarks`` bounded BFS passes give an O(L * (V+E)) runtime
    instead of the O(corpus * window * epochs) of skip-gram.
    """
    exclude = exclude or set()
    exclude.add(user)
    n = graph.node_count
    if n == 0 or not graph.has_node(user):
        return []

    emb = landmark_embedding(graph)
    target = emb.get(user)
    if target is None:
        return []
    exclude.update(graph.neighbors(user))

    scored = []
    for node_id, vec in emb.items():
        if node_id == user or node_id in exclude:
            continue
        scored.append((node_id, _cosine(target, vec)))
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return [(c, s, "图嵌入") for c, s in scored[:k]]


def landmark_embedding(
    graph: Graph,
    num_landmarks: int = config.EMBED_DIM,
    max_depth: int = config.EMBED_MAX_DEPTH,
) -> Dict[int, List[float]]:
    """Compute distance-to-landmarks embeddings for every node.

    Deterministic (seeded random for landmark selection) and cacheable.  The
    returned vectors are already normalised to ``[0, 1]`` per dimension.
    """
    n = graph.node_count
    if n == 0:
        return {}
    rng = random.Random(config.LOUVAIN_RANDOM_SEED + 7)

    # Cap landmarks at the number of nodes (a small graph has fewer landmarks
    # than the default dimension).
    num_landmarks = min(num_landmarks, n)

    # Landmarks: highest-degree hubs first, then random fill.
    top = sorted(graph.nodes, key=lambda nid: -graph.degree(nid))[: max(1, num_landmarks // 2)]
    top_set = set(top)
    while len(top) < num_landmarks:
        cand = graph.node_at_index(rng.randrange(n))
        if cand not in top_set:
            top_set.add(cand)
            top.append(cand)
    landmarks = top[:num_landmarks]

    # BFS distances from each landmark (bounded).
    dists = {lm: _bfs_distances(graph, lm, max_depth) for lm in landmarks}

    scale = 1.0 / max(1, max_depth)
    emb: Dict[int, List[float]] = {}
    for nid in graph.nodes:
        vec = [dists[lm].get(nid, max_depth) * scale for lm in landmarks]
        emb[nid] = vec
    return emb


def _bfs_distances(graph: Graph, source: int, max_depth: int) -> Dict[int, int]:
    dist: Dict[int, int] = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        d = dist[node]
        if d >= max_depth:
            continue
        for nb in graph.neighbors(node):
            if nb not in dist:
                dist[nb] = d + 1
                queue.append(nb)
    return dist


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


def recommend_popularity(
    graph: Graph,
    user: int,
    k: int,
    exclude: Optional[Set[int]] = None,
) -> List[Tuple[int, float, str]]:
    """Popularity baseline -- global degree ranking, used for cold start."""
    exclude = exclude or set()
    exclude.add(user)
    friends = set(graph.neighbors(user))
    exclude |= friends
    scored = [
        (nid, float(graph.degree(nid)))
        for nid in graph.nodes
        if nid not in exclude and nid != user
    ]
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return [(c, s, "热门") for c, s in scored[:k]]


def recommend_tag_based(
    graph: Graph,
    user: int,
    k: int,
    user_tags: Optional[Dict[int, Set[str]]] = None,
    exclude: Optional[Set[int]] = None,
) -> List[Tuple[int, float, str]]:
    """Tag-overlap recommendation -- helps cold start with rich profiles."""
    if not user_tags:
        return []
    exclude = exclude or set()
    exclude.add(user)
    tags = user_tags.get(user, set())
    if not tags:
        return []

    scored: Dict[int, float] = defaultdict(float)
    for other, otags in user_tags.items():
        if other == user or other in exclude:
            continue
        overlap = len(tags & otags)
        if overlap:
            scored[other] = float(overlap) / max(1, len(tags))
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(c, s, "标签") for c, s in ranked[:k]]


def max_marginal_relevance(
    candidates: List[Tuple[int, float, str]],
    graph: Graph,
    k: int,
    diversity: float,
) -> List[Tuple[int, float, str]]:
    """MMR re-ranking to inject diversity.

    Greedily picks the candidate that best balances raw score against maximum
    similarity to anything already chosen.  ``diversity`` = 0 yields the pure
    ranked list; ``diversity`` = 1 maximises spread.
    """
    if not candidates or diversity <= 0:
        return candidates[:k]
    selected: List[Tuple[int, float, str]] = []
    pool = list(candidates)
    selected_ids: Set[int] = set()

    while pool and len(selected) < k:
        best_idx, best_val = 0, -1e18
        for idx, (c, score, reason) in enumerate(pool):
            min_sim = 1e18
            for s_id, _, _ in selected:
                sim = _neighbor_overlap_sim(graph, c, s_id)
                if sim < min_sim:
                    min_sim = sim
            mmr = (1.0 - diversity) * score + diversity * min_sim
            if mmr > best_val:
                best_val, best_idx = mmr, idx
        chosen = pool.pop(best_idx)
        selected.append(chosen)
        selected_ids.add(chosen[0])
    return selected


def _neighbor_overlap_sim(graph: Graph, a: int, b: int) -> float:
    na = set(graph.neighbors(a))
    nb = set(graph.neighbors(b))
    if not na or not nb:
        return 0.0
    inter = len(na & nb)
    denom = math.sqrt(len(na) * len(nb))
    return inter / denom if denom else 0.0


def hybrid_recommend(
    graph: Graph,
    user: int,
    k: int = config.RECOMMEND_DEFAULT_K,
    strategy: str = "hybrid",
    diversity: float = config.DIVERSITY_LAMBDA,
    use_tags: bool = True,
    user_tags: Optional[Dict[int, Set[str]]] = None,
) -> Dict[str, object]:
    """Top-level recommender.

    Chooses a strategy, blends signals, then applies MMR diversity re-ranking.
    Returns a rich dict the API can serialise directly, including cold-start
    diagnostics.
    """
    friends = list(graph.neighbors(user))
    weighted_degree = 0
    for _n, w in graph.neighbors_with_weights(user):
        weighted_degree += int(w)
    if config.COLD_START_USE_WEIGHTED_DEGREE:
        degree = weighted_degree
    else:
        degree = graph.degree(user)
    cold_start = degree < config.COLD_START_CONNECTION_THRESHOLD

    exclude: Set[int] = set(friends)
    exclude.add(user)

    pool: Dict[int, Tuple[float, str]] = defaultdict(lambda: (0.0, ""))

    def _add(items, weight):
        for cid, score, reason in items:
            prev_score, prev_reason = pool[cid]
            pool[cid] = (prev_score + weight * score, reason)

    if strategy in ("cf", "hybrid"):
        _add(recommend_collaborative(graph, user, k * 4, exclude), 0.1)
    if strategy in ("embedding", "hybrid"):
        _add(recommend_embedding(graph, user, k * 4, exclude), 0.1)
    _add(recommend_popularity(graph, user, k * 4, exclude), 1.0)
    if use_tags and (strategy in ("hybrid",) or cold_start):
        _add(recommend_tag_based(graph, user, k * 4, user_tags, exclude), 5.0)

    if not pool:
        _add(recommend_popularity(graph, user, k * 4, set()), 1.0)

    ranked = sorted(
        pool.items(), key=lambda kv: (kv[1][0], kv[0])
    )
    candidates = [(c, s, r) for c, (s, r) in ranked]

    final = max_marginal_relevance(candidates, graph, k, diversity)

    items = [
        {
            "id": cid,
            "score": round(score, 6),
            "reason": reason,
        }
        for cid, score, reason in final
    ]

    return {
        "user": user,
        "strategy": strategy,
        "k": k,
        "cold_start": cold_start,
        "degree": degree,
        "diversity": diversity,
        "items": items,
    }
