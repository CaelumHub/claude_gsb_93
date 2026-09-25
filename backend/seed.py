"""
seed.py
-------
Demo data generator.  Builds a realistic-looking social graph with clustered
communities (so Louvain has something to find), a power-law-ish degree
distribution (so PageRank has meaningful hubs), and tagged users (so the
recommender has cold-start signals).

The generator is deterministic for reproducibility: the same parameters always
produce the same graph.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List

try:
    from . import config
except ImportError:  # pragma: no cover
    import config

# Chinese name fragments for realistic labels.
SURNAMES = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华"
GIVEN = ["伟", "芳", "娜", "敏", "静", "丽", "强", "磊", "军", "洋",
         "勇", "艳", "杰", "娟", "涛", "明", "超", "秀英", "霞", "平",
         "刚", "桂英", "文", "辉", "鑫", "海", "晨", "雪", "梅", "睿"]
TOPICS = ["科技", "音乐", "体育", "美食", "旅行", "摄影", "编程", "文学",
          "电影", "游戏", "健身", "时尚", "教育", "金融", "艺术", "汽车"]
CITIES = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安"]


def _name(rng: random.Random) -> str:
    return rng.choice(SURNAMES) + rng.choice(GIVEN)


def generate_demo(
    service,
    n_users: int = 120,
    communities: int = 5,
    avg_degree: int = 8,
    seed: int = 42,
    **kwargs,
) -> dict:
    """Generate a demo graph and populate the stores.

    The graph is built via a community-cluster model: each user belongs to one
    of ``communities`` clusters; edges are drawn mostly within a cluster with
    occasional cross-cluster "bridge" edges.  This yields clear community
    structure plus a realistic long tail.
    """
    rng = random.Random(seed)
    users: Dict[int, dict] = {}
    edges: List[tuple] = []

    # -- create users ------------------------------------------------------
    cluster_size = n_users // communities
    for uid in range(1, n_users + 1):
        cid = min((uid - 1) // cluster_size, communities - 1)
        tags = [TOPICS[cid % len(TOPICS)]]
        if rng.random() < 0.4:
            tags.append(rng.choice(TOPICS))
        users[uid] = {
            "name": _name(rng),
            "tags": tags,
            "attributes": {
                "city": rng.choice(CITIES),
                "age": rng.randint(18, 60),
                "cluster": cid,
            },
            "created_at": config.now_ms() - rng.randint(0, 30 * 86400_000),
        }

    # -- edges: intra-cluster dense, inter-cluster sparse --------------------
    seen = set()
    def _add(u, v):
        if u == v:
            return
        key = (u, v) if u < v else (v, u)
        if key in seen:
            return
        seen.add(key)
        edges.append((u, v, round(rng.uniform(0.5, 1.0), 2)))

    for uid in range(1, n_users + 1):
        cid = min((uid - 1) // cluster_size, communities - 1)
        lo = cid * cluster_size + 1
        hi = min(n_users, lo + cluster_size - 1)
        # intra-cluster edges
        for _ in range(rng.randint(avg_degree // 2, avg_degree)):
            other = rng.randint(lo, hi)
            _add(uid, other)
        # a few bridge edges
        if rng.random() < 0.08:
            other = rng.randint(1, n_users)
            _add(uid, other)

    # Guarantee no fully isolated nodes (avoid trivial components).
    for uid in range(1, n_users + 1):
        if all(u != uid and v != uid for u, v, _ in edges):
            other = rng.randint(1, n_users)
            _add(uid, other)

    # -- persist ------------------------------------------------------------
    service.store.save_users(users)
    for t in TOPICS:
        service.add_tag(t)
    result = service.store.import_edges(edges)
    service.invalidate_graph()
    service._rec_cache.clear()

    # Pre-compute community + pagerank + profiles so the UI is instantly populated.
    comm = service.compute_community(force=True)
    pr = service.compute_pagerank(top=20, force=True)
    service.build_profiles()

    return {
        "users": n_users,
        "edges": len(edges),
        "communities": communities,
        "imported": result["imported"],
        "modularity": comm.get("modularity", 0.0),
        "num_communities": comm.get("num_communities", 0),
        "time_ms": comm.get("time_ms"),
    }
