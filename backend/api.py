"""
api.py
------
Zero-dependency HTTP server (stdlib ``http.server``) exposing the REST API and
serving the static frontend.

Routing is a small hand-rolled dispatch table so we have no framework
dependency.  Every handler returns a ``(status, payload)`` tuple; JSON bodies
are parsed once and passed through.  Static files are served from
``frontend/`` with content-type inference and simple path traversal guarding.

Endpoint summary (all under ``/api``):

    GET    /api/health
    GET    /api/users                 ?page&size&search&tag
    POST   /api/users                 {name, tags, attributes}
    GET    /api/users/<id>
    PUT    /api/users/<id>            {name?, tags?, attributes?}
    DELETE /api/users/<id>
    GET    /api/users/<id>/neighbors  ?depth
    POST   /api/import                {edges:[[u,v],...], source}
    GET    /api/graph                 ?limit&community&top
    GET    /api/graph/neighborhood    ?node&depth&limit
    GET    /api/path                  ?source&target&algorithm
    GET    /api/common-friends        ?source&target
    GET    /api/community             (cached)
    POST   /api/community/compute     {resolution?}
    GET    /api/pagerank              ?top&refresh
    GET    /api/recommend/<id>        ?k&refresh&strategy
    POST   /api/recommend             {ids:[...], k}
    GET    /api/stats
    GET    /api/settings              /  PUT /api/settings
    POST   /api/settings/reset
    GET    /api/tags                  /  POST /api/tags  /  DELETE /api/tags/<name>
    POST   /api/users/<id>/tags       {tags:[...]}
    GET    /api/export                ?format=json|graphml|csv
    POST   /api/graph/rebuild-index
    POST   /api/graph/merge
    GET    /api/seed                  (POST /api/seed) -- demo data
"""

from __future__ import annotations

import csv
import io
import json
import mimetypes
import os
import re
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

try:
    from . import config, storage
    from .service import SocialGraphService
except ImportError:  # pragma: no cover
    import config
    import storage
    from service import SocialGraphService


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _json_bytes(obj, status: int = 200) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _error(message: str, status: int = 400) -> tuple:
    return status, {"error": message}


def _to_int(value: Optional[str], default: int):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Optional[str], default: float):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


class ApiRouter:
    """Routes an (method, path, body) request to a handler."""

    def __init__(self, service: SocialGraphService) -> None:
        self.service = service

    def dispatch(self, method: str, path: str, query: dict, body: Optional[dict]):
        # Strip leading "/api" prefix.
        route = path[len("/api"):] if path.startswith("/api") else path
        route = route.rstrip("/") or "/"

        # --- health ---
        if route == "/health":
            return 200, {
                "status": "ok",
                "service": "social-graph",
                "version": "1.0.0",
                "time": config.now_ms(),
            }

        # --- users collection ---
        if route == "/users" and method == "GET":
            page = _to_int(query.get("page"), 0)
            size = min(max(_to_int(query.get("size"), 20), 1), 500)
            return 200, self.service.list_users(
                page=page, size=size,
                search=query.get("search", ""),
                tag=query.get("tag", ""),
            )
        if route == "/users" and method == "POST":
            b = body or {}
            name = str(b.get("name", "")).strip()
            tags = b.get("tags") or []
            attributes = b.get("attributes") or {}
            user = self.service.create_user(name, tags, attributes)
            return 201, user

        # --- single user ---
        m = re.fullmatch(r"/users/(\d+)", route)
        if m:
            uid = int(m.group(1))
            if method == "GET":
                user = self.service.get_user(uid)
                if user is None:
                    return _error("用户不存在", 404)
                return 200, user
            if method == "PUT":
                user = self.service.update_user(uid, body or {})
                if user is None:
                    return _error("用户不存在", 404)
                return 200, user
            if method == "DELETE":
                if self.service.delete_user(uid):
                    return 200, {"deleted": uid}
                return _error("用户不存在", 404)

        # --- user neighbours ---
        m = re.fullmatch(r"/users/(\d+)/neighbors", route)
        if m:
            uid = int(m.group(1))
            depth = min(max(_to_int(query.get("depth"), 1), 1), config.NEIGHBORHOOD_MAX_DEPTH)
            graph = self.service.get_graph()
            if not graph.has_node(uid):
                return _error("用户不存在", 404)
            return 200, {
                "id": uid,
                "depth": depth,
                "neighbors": list(graph.neighbors(uid)),
                "degree": graph.degree(uid),
            }

        # --- user tags ---
        m = re.fullmatch(r"/users/(\d+)/tags", route)
        if m and method == "POST":
            uid = int(m.group(1))
            tags = (body or {}).get("tags", [])
            user = self.service.set_user_tags(uid, tags)
            if user is None:
                return _error("用户不存在", 404)
            return 200, user

        # --- import ---
        if route == "/import" and method == "POST":
            b = body or {}
            edges = b.get("edges") or []
            source = b.get("source", "manual")
            if not isinstance(edges, list):
                return _error("edges 必须是列表")
            normalised = []
            for e in edges:
                if isinstance(e, (list, tuple)) and len(e) >= 2:
                    try:
                        normalised.append((int(e[0]), int(e[1]), float(e[2]) if len(e) > 2 else 1.0))
                    except (TypeError, ValueError):
                        continue
            result = self.service.store.import_edges(normalised)
            result["imported"] = len(edges)
            self.service.invalidate_graph()
            storage.log_import({**result, "source": source, "time": config.now_ms()})
            return 200, {**result, "source": source}

        # --- graph ---
        if route == "/graph" and method == "GET":
            limit = _to_int(query.get("limit"), config.NEIGHBORHOOD_SAMPLE_LIMIT)
            graph = self.service.get_graph()
            users = self.service.store.load_users()
            community = self.service.get_community().get("communities", {})
            nodes = []
            for nid in graph.nodes:
                comm = -1
                if nid in community:
                    comm = community[nid]
                nodes.append({
                    "id": nid,
                    "label": users.get(nid, {}).get("name", str(nid)),
                    "degree": graph.degree(nid),
                    "community": comm,
                })
                if len(nodes) >= limit:
                    break
            node_ids = {n["id"] for n in nodes}
            edges = []
            for u, v, w in graph.iter_edges():
                if u in node_ids and v in node_ids:
                    edges.append({"from": u, "to": v, "weight": round(w, 3)})
                if len(edges) >= limit * 4:
                    break
            return 200, {
                "nodes": nodes,
                "edges": edges,
                "stats": {"nodes": graph.node_count, "edges": graph.edge_count},
            }

        # --- graph neighbourhood ---
        if route == "/graph/neighborhood" and method == "GET":
            node = _to_int(query.get("node"), -1)
            depth = min(max(_to_int(query.get("depth"), config.NEIGHBORHOOD_DEFAULT_DEPTH), 1), config.NEIGHBORHOOD_MAX_DEPTH)
            limit = _to_int(query.get("limit"), config.NEIGHBORHOOD_SAMPLE_LIMIT)
            graph = self.service.store.neighborhood_graph(node, depth, limit)
            users = self.service.store.load_users()
            community = self.service.get_community().get("communities", {})
            nodes = []
            for nid in graph.nodes:
                comm = -1
                if nid in community:
                    comm = community[nid]
                nodes.append({
                    "id": nid,
                    "label": users.get(nid, {}).get("name", str(nid)),
                    "degree": graph.degree(nid),
                    "community": comm,
                })
            edges = [
                {"from": u, "to": v, "weight": round(w, 3)}
                for u, v, w in graph.iter_edges()
            ]
            return 200, {"root": node, "depth": depth, "nodes": nodes, "edges": edges}

        # --- shortest path ---
        if route == "/path" and method == "GET":
            source = _to_int(query.get("source"), -1)
            target = _to_int(query.get("target"), -1)
            algorithm = query.get("algorithm", "auto")
            if source < 0 or target < 0:
                return _error("缺少 source 或 target 参数")
            return 200, self.service.find_shortest_path(source, target, algorithm)

        # --- common friends ---
        if route == "/common-friends" and method == "GET":
            source = _to_int(query.get("source"), -1)
            target = _to_int(query.get("target"), -1)
            if source < 0 or target < 0:
                return _error("缺少 source 或 target 参数")
            info = self.service.common_friends_info(source, target)
            info["source"], info["target"] = target, source
            return 200, info

        # --- community ---
        if route == "/community" and method == "GET":
            if config.COMMUNITY_READ_DIRECT:
                data = self.service.derived.load_community()
                data.setdefault("communities", {})
                data.setdefault("num_communities", 0)
                data.setdefault("modularity", 0.0)
                return 200, data
            return 200, self.service.get_community()
        if route == "/community/compute" and method == "POST":
            resolution = _to_float((body or {}).get("resolution"), config.LOUVAIN_RESOLUTION)
            force = _to_bool(str((body or {}).get("force", "true")))
            return 200, self.service.compute_community(resolution=resolution, force=True)

        # --- pagerank ---
        if route == "/pagerank" and method == "GET":
            top = _to_int(query.get("top"), 20)
            refresh = _to_bool(query.get("refresh"), False)
            result = self.service.compute_pagerank(top=top, force=refresh)
            for item in result.get("top", []):
                item["score"] = round(item["score"] * 100.0, 8)
            return 200, result

        # --- recommendations ---
        m = re.fullmatch(r"/recommend/(\d+)", route)
        if m and method == "GET":
            uid = int(m.group(1))
            raw_k = _to_int(query.get("k"), config.RECOMMEND_DEFAULT_K)
            k = raw_k
            if k > config.RECOMMEND_CLAMP_MAX:
                k = config.RECOMMEND_CLAMP_MAX
            if k < 1:
                k = 1
            refresh = _to_bool(query.get("refresh"), False)
            strategy = None
            return 200, self.service.recommend(uid, k=k, refresh=refresh, strategy=strategy)
        if route == "/recommend" and method == "POST":
            ids = (body or {}).get("ids") or []
            raw_k = _to_int((body or {}).get("k"), config.RECOMMEND_DEFAULT_K)
            k = raw_k
            if k > config.RECOMMEND_CLAMP_MAX:
                k = config.RECOMMEND_CLAMP_MAX
            if k < 1:
                k = 1
            result = {}
            for uid in ids:
                result[str(uid)] = self.service.recommend(int(uid), k=k)["items"]
            return 200, {"results": result}

        # --- stats ---
        if route == "/stats" and method == "GET":
            return 200, self.service.full_stats()

        # --- profiles (separately stored user profiles) ---
        if route == "/profiles" and method == "GET":
            return 200, {"profiles": self.service.get_profiles()}
        if route == "/profiles/build" and method == "POST":
            return 200, {"built": len(self.service.build_profiles())}

        # --- settings ---
        if route == "/settings" and method == "GET":
            return 200, self.service.settings.get()
        if route == "/settings" and method == "PUT":
            self.service.settings.update(body or {})
            return 200, self.service.settings.get()
        if route == "/settings/reset" and method == "POST":
            return 200, self.service.settings.reset()

        # --- tags ---
        if route == "/tags" and method == "GET":
            tags = self.service.list_tags()
            for t in tags:
                t["count"] += 1
                t["tag"] = t.get("name")
            return 200, {"tags": tags}
        if route == "/tags" and method == "POST":
            name = str((body or {}).get("name", "")).strip()
            color = (body or {}).get("color")
            if not name:
                return _error("缺少标签名称")
            return 201, self.service.add_tag(name, color)
        m = re.fullmatch(r"/tags/(.+)", route)
        if m and method == "DELETE":
            name = urllib.parse.unquote(m.group(1))
            if self.service.delete_tag(name):
                return 200, {"deleted": name}
            return _error("标签不存在", 404)

        # --- export ---
        if route == "/export" and method == "GET":
            fmt = query.get("format", "json")
            data = self.service.export_graph(fmt)
            return 200, {"__download__": True, "filename": f"graph_export.{fmt}", "format": fmt, **data}

        # --- maintenance ---
        if route == "/graph/rebuild-index" and method == "POST":
            from .storage import rebuild_index_from_shards
            idx = rebuild_index_from_shards()
            self.service.invalidate_graph()
            return 200, {"rebuilt": True, "node_count": idx.meta["node_count"], "edge_count": idx.meta["edge_count"]}
        if route == "/graph/merge" and method == "POST":
            result = self.service.store.merge_shards()
            self.service.invalidate_graph()
            return 200, result

        # --- seed (demo) ---
        if route == "/seed" and method == "POST":
            from .seed import generate_demo
            result = generate_demo(self.service, **(body or {}))
            return 200, result

        return _error(f"未知路由: {method} {route}", 404)


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".map": "application/json",
}


def _serve_static(handler: BaseHTTPRequestHandler, path: str) -> None:
    # Resolve within FRONTEND_DIR and prevent path traversal.
    rel = path.lstrip("/")
    if rel in ("", "/"):
        rel = "index.html"
    safe_path = os.path.normpath(os.path.join(config.FRONTEND_DIR, rel))
    if not safe_path.startswith(os.path.normpath(config.FRONTEND_DIR)):
        handler.send_error(403, "Forbidden")
        return
    if os.path.isdir(safe_path):
        safe_path = os.path.join(safe_path, "index.html")
    if not os.path.isfile(safe_path):
        handler.send_error(404, "Not Found")
        return
    ext = os.path.splitext(safe_path)[1].lower()
    ctype = CONTENT_TYPES.get(ext, "application/octet-stream")
    try:
        with open(safe_path, "rb") as fh:
            data = fh.read()
    except OSError:
        handler.send_error(500, "Read error")
        return
    handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()
    handler.wfile.write(data)


def _serialise_export(obj, fmt: str) -> bytes:
    """Turn an export dict into JSON / GraphML / CSV bytes."""
    nodes = obj.get("nodes", [])
    edges = obj.get("edges", [])
    if fmt == "graphml":
        buf = io.StringIO()
        buf.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        buf.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
        buf.write('  <graph edgedefault="undirected">\n')
        for n in nodes:
            name = str(n.get("name", n.get("id"))).replace("&", "&amp;").replace("<", "&lt;")
            deg = n.get("degree", 0)
            comm = n.get("community", -1)
            buf.write(f'    <node id="{n["id"]}">\n')
            buf.write(f'      <data key="label">{name}</data>\n')
            buf.write(f'      <data key="degree">{deg}</data>\n')
            buf.write(f'      <data key="community">{comm}</data>\n')
            buf.write("    </node>\n")
        for e in edges:
            buf.write(f'    <edge source="{e["from"]}" target="{e["to"]}"/>\n')
        buf.write("  </graph>\n</graphml>\n")
        return buf.getvalue().encode("utf-8")
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["from", "to", "weight"])
        for e in edges:
            w.writerow([e["from"], e["to"], e.get("weight", config.EXPORT_DEFAULT_WEIGHT)])
        return buf.getvalue().encode("utf-8")
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class SocialGraphHandler(BaseHTTPRequestHandler):
    server_version = "SocialGraph/1.0"
    router: Optional[ApiRouter] = None  # injected by the server factory

    # -- helpers ------------------------------------------------------------
    def _send_json(self, obj, status: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> Optional[dict]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return None
        if length > config.MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    # -- method dispatch ----------------------------------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _handle_api(self, method: str) -> None:
        parsed = urllib.parse.urlparse(self.path)
        query = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        body = self._read_body()
        try:
            status, payload = self.router.dispatch(method, parsed.path, query, body)
        except Exception as exc:  # noqa: BLE001 - surface any bug as 500 JSON
            self._send_json({"error": f"服务器内部错误: {exc}", "trace": repr(exc)}, 500)
            return
        # Download responses (export) get a Content-Disposition header.
        if isinstance(payload, dict) and payload.get("__download__"):
            fmt = payload.get("format", "json")
            filename = payload.get("filename", f"export.{fmt}")
            data = _serialise_export(payload, fmt)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header(
                "Content-Disposition", f'attachment; filename="{filename}"'
            )
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
            return
        self._send_json(payload, status)

    def do_GET(self):
        if self.path.startswith("/api"):
            self._handle_api("GET")
        else:
            _serve_static(self, self.path)

    def do_POST(self):
        if self.path.startswith("/api"):
            self._handle_api("POST")
        else:
            self.send_error(404)

    def do_PUT(self):
        if self.path.startswith("/api"):
            self._handle_api("PUT")
        else:
            self.send_error(404)

    def do_DELETE(self):
        if self.path.startswith("/api"):
            self._handle_api("DELETE")
        else:
            self.send_error(404)

    def log_message(self, fmt, *args):  # quiet-ish logging
        if os.environ.get("GSB_VERBOSE"):
            super().log_message(fmt, *args)


def create_server(service: SocialGraphService, host: str = config.HOST, port: int = config.PORT):
    handler = type(
        "BoundHandler",
        (SocialGraphHandler,),
        {"router": ApiRouter(service)},
    )
    return ThreadingHTTPServer((host, port), handler)


def run(service: SocialGraphService) -> None:
    config.ensure_dirs()
    server = create_server(service)
    print(f"[social-graph] serving on http://{config.HOST}:{config.PORT}")
    print(f"[social-graph] data dir: {config.DATA_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[social-graph] shutting down")
        server.server_close()
