/* =========================================================================
   api.js — 后端 REST API 客户端
   =========================================================================
   薄封装，统一 fetch 行为：JSON 编解码、错误处理、下载支持。
   所有页面共享，避免在每个页面重复拼接 URL 与错误分支。
   ========================================================================= */
(function (global) {
  "use strict";

  const BASE = ""; // same origin; served by the backend

  async function request(method, path, body, opts) {
    opts = opts || {};
    const init = { method, headers: {} };
    if (body !== undefined && body !== null) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    let res;
    try {
      res = await fetch(BASE + path, init);
    } catch (e) {
      throw new Error("无法连接服务器，请确认后端已启动 (" + e.message + ")");
    }
    if (!res.ok) {
      let msg = "请求失败 (" + res.status + ")";
      try {
        const j = await res.json();
        if (j && j.error) msg = j.error;
      } catch (_) { /* ignore */ }
      throw new Error(msg);
    }
    // 下载类响应（Content-Disposition: attachment）返回原始文本，由调用方处理。
    const disp = res.headers.get("Content-Disposition") || "";
    if (disp.indexOf("attachment") !== -1) {
      const text = await res.text();
      return { _download: true, text, filename: parseFilename(disp) };
    }
    return res.json();
  }

  function parseFilename(disp) {
    const m = /filename="?([^";]+)"?/.exec(disp);
    return m ? m[1] : "download";
  }

  function qs(params) {
    const parts = [];
    for (const k in params) {
      if (params[k] === undefined || params[k] === null || params[k] === "") continue;
      parts.push(encodeURIComponent(k) + "=" + encodeURIComponent(params[k]));
    }
    return parts.length ? "?" + parts.join("&") : "";
  }

  const api = {
    health: () => request("GET", "/api/health"),

    // 用户
    listUsers: (p) => request("GET", "/api/users" + qs(p)),
    getUser: (id) => request("GET", "/api/users/" + id),
    createUser: (b) => request("POST", "/api/users", b),
    updateUser: (id, b) => request("PUT", "/api/users/" + id, b),
    deleteUser: (id) => request("DELETE", "/api/users/" + id),
    setUserTags: (id, tags) => request("POST", "/api/users/" + id + "/tags", { tags }),

    // 关系导入
    importEdges: (edges, source) => request("POST", "/api/import", { edges, source }),

    // 图
    graph: (p) => request("GET", "/api/graph" + qs(p)),
    neighborhood: (p) => request("GET", "/api/graph/neighborhood" + qs(p)),

    // 路径 & 共同好友
    shortestPath: (source, target, algorithm) =>
      request("GET", "/api/path" + qs({ source, target, algorithm })),
    commonFriends: (source, target) =>
      request("GET", "/api/common-friends" + qs({ source, target })),

    // 社群 / PageRank
    community: () => request("GET", "/api/community"),
    computeCommunity: (resolution) => request("POST", "/api/community/compute", { resolution }),
    pagerank: (p) => request("GET", "/api/pagerank" + qs(p)),

    // 推荐
    recommend: (id, p) => request("GET", "/api/recommend/" + id + qs(p)),
    recommendMany: (ids, k) => request("POST", "/api/recommend", { ids, k }),

    // 统计 / 设置 / 标签
    stats: () => request("GET", "/api/stats"),
    settings: () => request("GET", "/api/settings"),
    saveSettings: (b) => request("PUT", "/api/settings", b),
    resetSettings: () => request("POST", "/api/settings/reset"),
    tags: () => request("GET", "/api/tags"),
    addTag: (name, color) => request("POST", "/api/tags", { name, color }),
    deleteTag: (name) => request("DELETE", "/api/tags/" + encodeURIComponent(name)),

    // 导出 & 维护
    export: (fmt) => request("GET", "/api/export" + qs({ format: fmt })),
    rebuildIndex: () => request("POST", "/api/graph/rebuild-index"),
    mergeShards: () => request("POST", "/api/graph/merge"),
    seed: (p) => request("POST", "/api/seed", p || {}),
  };

  global.api = api;
})(window);
