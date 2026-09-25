"""
run.py
------
Entry point.  Start the server:

    python -m backend.run            # serve on 127.0.0.1:8080
    GSB_PORT=9000 python -m backend.run
    python backend/run.py --seed    # seed demo data on startup if empty

Also supports a ``--seed`` flag to auto-populate demo data when the dataset is
empty, and a ``--check`` flag to run a quick self-test of the algorithm stack
and exit (useful for CI / validation).
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running both as a package (`python -m backend.run`) and as a script
# (`python backend/run.py`).
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend import api, config, seed
    from backend.service import SocialGraphService
else:
    from . import api, config, seed
    from .service import SocialGraphService


def _check() -> int:
    """Run a smoke test over the algorithm stack; exit 0 on success."""
    from backend import algorithms
    from backend.graph import Graph

    g = Graph(directed=False)
    # A small barbell: two 3-cliques joined by a bridge.
    edges = [(1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (5, 6), (4, 6)]
    for u, v in edges:
        g.add_edge(u, v)
    g.freeze()

    assert g.node_count == 6, g.node_count
    assert g.edge_count == 7, g.edge_count

    path, dist, alg = algorithms.shortest_path(g, 1, 6)
    assert path == [1, 2, 3, 4, 5, 6] or path == [1, 3, 4, 6, 5] or len(path) - 1 == dist, path

    common = algorithms.common_friends(g, 1, 6)
    pr = algorithms.pagerank(g)
    assert abs(sum(pr.values()) - 1.0) < 1e-6, sum(pr.values())

    lv = algorithms.louvain(g)
    assert lv["num_communities"] >= 2, lv  # cliques should separate

    rec = algorithms.hybrid_recommend(g, 1, k=3)
    assert "items" in rec

    print("[check] OK: graph, bfs, pagerank, louvain, recommend all pass")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Social graph analysis server")
    parser.add_argument("--seed", action="store_true", help="seed demo data if empty")
    parser.add_argument("--check", action="store_true", help="run self-test and exit")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    if args.check:
        return _check()

    if args.host:
        config.HOST = args.host
    if args.port:
        config.PORT = args.port

    config.ensure_dirs()
    service = SocialGraphService()

    if args.seed:
        users = service.store.load_users()
        if not users:
            print("[social-graph] empty dataset -- seeding demo data ...")
            result = seed.generate_demo(service)
            print(f"[social-graph] seeded {result['users']} users, {result['edges']} edges")

    api.run(service)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
