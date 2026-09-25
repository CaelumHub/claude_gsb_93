"""
graph.py
--------
Memory-efficient, undirected graph with optional edge weights.

Why a custom graph instead of ``networkx`` or an adjacency dict-of-lists?

1.  **Memory.**  A Python ``dict[int, list[int]]`` stores every integer as a
    ~28-byte ``PyObject`` plus a list pointer.  For a 1M-node / 10M-edge graph
    that is hundreds of megabytes of pure overhead.  We instead keep the
    adjacency in **CSR (Compressed Sparse Row)** form using ``array('q')``,
    which stores each id in a flat 8-byte C array -- roughly 3.5x smaller and
    cache-friendlier for the tight loops in BFS / PageRank / Louvain.

2.  **Determinism.**  Neighbour lists are kept sorted and deduplicated so that
    algorithms are reproducible run-to-run.

3.  **Control.**  We know exactly which access patterns the algorithms need
    (iterate neighbours, check membership via sorted binary search, degree),
    so we can optimise precisely for those.

The class exposes two phases:

* a mutable **build** phase (``add_edge``) where the graph can grow freely, and
* a **freeze()** phase that compacts the graph into CSR arrays for fast,
  read-only algorithm execution.

After ``freeze()`` the graph is immutable; call ``clone_builder()`` to obtain a
new mutable graph seeded from the frozen one (used by the incremental update
path in ``storage.py``).
"""

from __future__ import annotations

import bisect
import itertools
from array import array
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    from . import config
except ImportError:  # pragma: no cover - allows running the file directly
    import config

NodeId = int


class Graph:
    """Undirected weighted graph backed by CSR arrays when frozen."""

    __slots__ = (
        "_adj",            # build phase: node -> dict[node -> weight]
        "_node_index",     # node id -> dense row index
        "_index_node",     # dense row index -> node id
        "_offsets",        # CSR: start offset per row (array('q'))
        "_neighbors",      # CSR: flat neighbour ids (array('q'))
        "_weights",        # CSR: flat weights (array('d')) -- parallel to _neighbors
        "_frozen",
        "directed",
    )

    def __init__(self, directed: bool = False) -> None:
        self._adj: Dict[NodeId, Dict[NodeId, float]] = defaultdict(dict)
        self._node_index: Dict[NodeId, int] = {}
        self._index_node: List[NodeId] = []
        self._offsets: Optional[array] = None
        self._neighbors: Optional[array] = None
        self._weights: Optional[array] = None
        self._frozen = False
        self.directed = directed

    # ------------------------------------------------------------------
    # Build phase
    # ------------------------------------------------------------------
    def add_node(self, node: NodeId) -> None:
        if self._frozen:
            raise RuntimeError("graph is frozen; call clone_builder() first")
        if node not in self._node_index:
            self._node_index[node] = len(self._index_node)
            self._index_node.append(node)
            self._adj[node]  # materialise the row

    def add_edge(self, u: NodeId, v: NodeId, weight: float = 1.0) -> None:
        if self._frozen:
            raise RuntimeError("graph is frozen; call clone_builder() first")
        if u == v:
            return
        self.add_node(u)
        self.add_node(v)
        # For an undirected graph store the edge once per direction; edges with
        # repeated weight accumulate so that parallel imports merge gracefully.
        self._adj[u][v] = self._adj[u].get(v, 0.0) + weight
        if not self.directed:
            self._adj[v][u] = self._adj[v].get(u, 0.0) + weight

    def add_edges(self, edges: Iterable[Tuple[NodeId, NodeId, float]]) -> int:
        added = 0
        for u, v, w in edges:
            self.add_edge(u, v, w)
            added += 1
        return added

    def has_node(self, node: NodeId) -> bool:
        return node in self._node_index

    def has_edge(self, u: NodeId, v: NodeId) -> bool:
        if self._frozen:
            return self._edge_binary_search(u, v)
        return u in self._adj and v in self._adj[u]

    def degree(self, node: NodeId) -> int:
        if self._frozen:
            idx = self._node_index.get(node)
            if idx is None:
                return 0
            return self._offsets[idx + 1] - self._offsets[idx]
        return len(self._adj.get(node, {}))

    def neighbors(self, node: NodeId) -> Iterator[NodeId]:
        if self._frozen:
            idx = self._node_index.get(node)
            if idx is None:
                return
            start = self._offsets[idx]
            end = self._offsets[idx + 1]
            for i in range(start, end):
                yield self._neighbors[i]
        else:
            for nb in self._adj.get(node, {}):
                yield nb

    def neighbors_with_weights(self, node: NodeId):
        if self._frozen:
            idx = self._node_index.get(node)
            if idx is None:
                return
            start = self._offsets[idx]
            end = self._offsets[idx + 1]
            for i in range(start, end):
                yield self._neighbors[i], self._weights[i]
        else:
            for nb, w in self._adj.get(node, {}).items():
                yield nb, w

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def node_count(self) -> int:
        return len(self._index_node)

    @property
    def edge_count(self) -> int:
        if self._frozen:
            total = (self._offsets[-1] - self._offsets[0]) if self._offsets else 0
            return total if self.directed else total // 2
        total = sum(len(v) for v in self._adj.values())
        return total if self.directed else total // 2

    @property
    def nodes(self) -> List[NodeId]:
        return list(self._index_node)

    def node_at_index(self, idx: int) -> NodeId:
        return self._index_node[idx]

    def index_of(self, node: NodeId) -> Optional[int]:
        return self._node_index.get(node)

    def total_degree(self, node: NodeId) -> int:
        return self.degree(node)

    # ------------------------------------------------------------------
    # Freeze -> CSR
    # ------------------------------------------------------------------
    def freeze(self) -> "Graph":
        """Compact into CSR arrays.  Idempotent."""
        if self._frozen:
            return self
        n = len(self._index_node)
        offsets = array("q", [0]) * (n + 1)
        neighbors = array("q")
        weights = array("d")

        for idx in range(n):
            node = self._index_node[idx]
            row = self._adj.get(node, {})
            # Deterministic ordering by neighbour id.
            items = sorted(row.items())
            offsets[idx + 1] = offsets[idx] + len(items)
            for nb, w in items:
                neighbors.append(nb)
                weights.append(w)

        self._offsets = offsets
        self._neighbors = neighbors
        self._weights = weights
        self._frozen = True
        # Release the mutable adjacency structure to reclaim memory.
        self._adj = None
        return self

    def _edge_binary_search(self, u: NodeId, v: NodeId) -> bool:
        idx = self._node_index.get(u)
        if idx is None:
            return False
        start = self._offsets[idx]
        end = self._offsets[idx + 1]
        # self._neighbors[start:end] is sorted, so bisect works.
        pos = bisect.bisect_left(self._neighbors, v, start, end)
        return pos < end and self._neighbors[pos] == v

    # ------------------------------------------------------------------
    # Edges export (for JSON / frontend)
    # ------------------------------------------------------------------
    def iter_edges(self) -> Iterator[Tuple[NodeId, NodeId, float]]:
        if self._frozen:
            n = len(self._index_node)
            for idx in range(n):
                start = self._offsets[idx]
                end = self._offsets[idx + 1]
                u = self._index_node[idx]
                for i in range(start, end):
                    v = self._neighbors[i]
                    w = 1.0
                    yield u, v, w
        else:
            for u, row in self._adj.items():
                for v, w in row.items():
                    if not self.directed and u > v:
                        continue
                    yield u, v, w

    def edges_as_list(self, limit: Optional[int] = None):
        it = self.iter_edges()
        if limit is not None:
            it = itertools.islice(it, limit)
        return list(it)

    def clone_builder(self) -> "Graph":
        """Return a mutable copy seeded with the current topology.

        Used by the incremental-update path: freeze the master, then rebuild a
        mutable graph from it before appending new edges.
        """
        g = Graph(directed=self.directed)
        for u, v, w in self.iter_edges():
            g.add_edge(u, v, w)
        return g


class GraphBuilder:
    """Convenience alias / factory kept for readability in service code."""

    def __init__(self, directed: bool = False) -> None:
        self.graph = Graph(directed=directed)

    def edge(self, u: NodeId, v: NodeId, w: float = 1.0) -> None:
        self.graph.add_edge(u, v, w)

    def build(self) -> Graph:
        return self.graph
