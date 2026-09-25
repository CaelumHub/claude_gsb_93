/* =========================================================================
   common.js — 应用外壳与共享工具
   =========================================================================
   负责：侧栏/顶栏渲染、主题切换、Toast、加载态、分页、格式化、以及每页
   统一的初始化流程（`initPage`）。页面脚本在 body 上设置 `data-page`
   后调用 `initPage(fn)`，即可获得一致的导航与外壳。
   ========================================================================= */
(function (global) {
  "use strict";

  // ------------------------------------------------------------------
  // 导航定义（10 个页面）
  // ------------------------------------------------------------------
  const NAV = [
    {
      group: "图与关系",
      items: [
        { key: "graph",      ico: "🕸", label: "图可视化",      href: "graph.html" },
        { key: "import",     ico: "⇄", label: "关系导入",      href: "import.html" },
        { key: "path",       ico: "🧭", label: "路径与共同好友", href: "path.html" },
        { key: "community",  ico: "◈", label: "社群发现",      href: "community.html" },
      ],
    },
    {
      group: "智能推荐",
      items: [
        { key: "recommend",  ico: "✦", label: "个性化推荐",    href: "recommend.html" },
        { key: "users",      ico: "👤", label: "用户管理",      href: "users.html" },
        { key: "tags",       ico: "🏷", label: "标签管理",      href: "tags.html" },
      ],
    },
    {
      group: "系统",
      items: [
        { key: "stats",      ico: "📊", label: "统计面板",      href: "stats.html" },
        { key: "export",     ico: "⬇", label: "数据导出",      href: "export.html" },
        { key: "settings",   ico: "⚙", label: "系统设置",      href: "settings.html" },
      ],
    },
  ];

  const PAGE_TITLES = {
    graph: ["图可视化", "vis.js 缩放拖拽 · 路径高亮"],
    import: ["关系导入", "批量导入边 · 增量更新"],
    path: ["路径与共同好友", "BFS 最短路径 · 共同好友查询"],
    community: ["社群发现", "Louvain 结果着色"],
    recommend: ["个性化推荐", "协同过滤 · 图嵌入 · 冷启动与多样性"],
    users: ["用户管理", "用户 CRUD 与画像"],
    tags: ["标签管理", "标签体系与关联"],
    stats: ["统计面板", "图指标总览"],
    export: ["数据导出", "JSON / GraphML / CSV"],
    settings: ["系统设置", "算法与存储参数"],
  };

  // ------------------------------------------------------------------
  // 外壳渲染
  // ------------------------------------------------------------------
  function renderShell(pageKey) {
    const sidebar = document.getElementById("sidebar");
    let html = '<div class="sidebar-brand"><div class="logo">G</div><div><div class="title">社交网络分析</div><div class="subtitle">Graph &amp; Recommender</div></div></div>';
    html += '<nav class="nav">';
    for (const grp of NAV) {
      html += '<div class="nav-group-title">' + grp.group + "</div>";
      for (const it of grp.items) {
        const active = it.key === pageKey ? " active" : "";
        html +=
          '<button class="nav-item' + active + '" data-href="' + it.href + '">' +
          '<span class="ico">' + it.ico + "</span><span>" + it.label + "</span></button>";
      }
    }
    html += "</nav>";
    html += '<div class="sidebar-foot" id="sidebarFoot">等待后端…</div>';
    sidebar.innerHTML = html;

    sidebar.querySelectorAll(".nav-item").forEach((el) => {
      el.addEventListener("click", () => { location.href = el.dataset.href; });
    });

    // 顶栏
    const topbar = document.getElementById("topbar");
    const [title, desc] = PAGE_TITLES[pageKey] || ["", ""];
    topbar.innerHTML =
      '<button class="btn btn-ghost btn-sm menu-toggle" id="menuToggle">☰</button>' +
      '<span class="crumb">社交网络图分析与推荐系统 / <strong>' + title + "</strong></span>" +
      '<div class="topbar-actions">' +
      '<button class="btn btn-ghost btn-sm" id="themeToggle" title="切换深浅色">🌓</button>' +
      '<span class="badge" id="healthBadge">…</span>' +
      "</div>";

    document.getElementById("menuToggle").addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });
    document.getElementById("themeToggle").addEventListener("click", toggleTheme);

    // 页面头部区域（若存在占位）
    const headSlot = document.getElementById("pageHead");
    if (headSlot) {
      headSlot.innerHTML = '<h1>' + title + '</h1><p class="desc">' + desc + "</p>";
    }

    refreshHealth();
    applyTheme();
  }

  async function refreshHealth() {
    const badge = document.getElementById("healthBadge");
    const foot = document.getElementById("sidebarFoot");
    try {
      const h = await api.health();
      if (badge) { badge.className = "badge success"; badge.textContent = "服务正常"; }
      if (foot) { foot.textContent = "v" + h.version + " · 运行中"; }
    } catch (e) {
      if (badge) { badge.className = "badge danger"; badge.textContent = "离线"; }
      if (foot) { foot.textContent = "后端未连接"; }
    }
  }

  // ------------------------------------------------------------------
  // 主题
  // ------------------------------------------------------------------
  function applyTheme() {
    let dark = localStorage.getItem("gsb-dark") === "1";
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  }
  function toggleTheme() {
    const cur = localStorage.getItem("gsb-dark") === "1";
    localStorage.setItem("gsb-dark", cur ? "0" : "1");
    applyTheme();
  }

  // ------------------------------------------------------------------
  // Toast
  // ------------------------------------------------------------------
  function toast(message, type) {
    type = type || "info";
    let box = document.getElementById("toasts");
    if (!box) { box = document.createElement("div"); box.id = "toasts"; document.body.appendChild(box); }
    const el = document.createElement("div");
    el.className = "toast " + type;
    const ico = { success: "✓ ", error: "✕ ", warning: "⚠ ", info: "" }[type] || "";
    el.textContent = ico + message;
    box.appendChild(el);
    setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity 0.3s"; setTimeout(() => el.remove(), 300); }, 3200);
  }

  // ------------------------------------------------------------------
  // 格式化
  // ------------------------------------------------------------------
  function fmt(n) {
    if (n === null || n === undefined) return "—";
    if (typeof n === "number") {
      if (Number.isInteger(n)) return n.toLocaleString("zh-CN");
      return n.toFixed(2);
    }
    return String(n);
  }
  function fmtPct(x) { return (x * 100).toFixed(2) + "%"; }
  function fmtMs(ms) { return ms < 1000 ? ms.toFixed(1) + " ms" : (ms / 1000).toFixed(2) + " s"; }
  function escapeHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // 社群分类调色板
  const PALETTE = ["#4f8cff", "#22a06b", "#e09a2f", "#e0524d", "#8a63d2",
                   "#3aa0c9", "#d45d9a", "#6b8f5e", "#c97b2f", "#5b7a99",
                   "#a15c8a", "#4fae9e"];
  function colorFor(i) { return PALETTE[((i % PALETTE.length) + PALETTE.length) % PALETTE.length]; }

  // ------------------------------------------------------------------
  // 加载态 / 渲染辅助
  // ------------------------------------------------------------------
  function spinner() { return '<div class="loading"><span class="spinner"></span>加载中…</div>'; }
  function emptyState(big, text) {
    return '<div class="empty"><div class="big">' + (big || "🗂") + "</div><div>" + (text || "暂无数据") + "</div></div>";
  }

  // 渲染一个带分页的表（通用）
  function renderTable(container, columns, rows, opts) {
    opts = opts || {};
    let html = '<div class="table-wrap"><table class="table"><thead><tr>';
    for (const c of columns) {
      html += "<th" + (c.align ? ' style="text-align:' + c.align + '"' : "") + ">" + c.label + "</th>";
    }
    html += "</tr></thead><tbody>";
    if (!rows.length) {
      html += '<tr><td colspan="' + columns.length + '" style="text-align:center;color:var(--text-faint);padding:30px">暂无数据</td></tr>';
    } else {
      for (const r of rows) {
        html += "<tr>";
        for (const c of columns) {
          const val = c.render ? c.render(r) : (r[c.key] == null ? "—" : escapeHtml(r[c.key]));
          html += "<td" + (c.align ? ' style="text-align:' + c.align + '"' : "") + ">" + val + "</td>";
        }
        html += "</tr>";
      }
    }
    html += "</tbody></table></div>";
    container.innerHTML = html;
  }

  // ------------------------------------------------------------------
  // 每页初始化
  // ------------------------------------------------------------------
  function initPage(pageKey, fn) {
    document.addEventListener("DOMContentLoaded", function () {
      renderShell(pageKey);
      try {
        if (typeof fn === "function") fn();
      } catch (e) {
        console.error(e);
        toast("页面初始化失败: " + e.message, "error");
      }
    });
  }

  // 模态框
  function modal(title, bodyHtml, actions) {
    const backdrop = document.createElement("div");
    backdrop.className = "modal-backdrop";
    let actHtml = "";
    (actions || []).forEach((a, i) => {
      actHtml += '<button class="btn ' + (a.cls || "") + '" data-act="' + i + '">' + a.label + "</button>";
    });
    backdrop.innerHTML =
      '<div class="modal"><h3>' + escapeHtml(title) + "</h3><div>" + bodyHtml +
      '</div><div class="modal-actions">' + actHtml + "</div></div>";
    document.body.appendChild(backdrop);
    return new Promise((resolve) => {
      // 注意：在 resolve 后延迟移除 DOM（宏任务），以便调用方在 await 返回后
      // 仍能读取表单值（此时 modal 尚未从文档中移除）。
      const close = () => setTimeout(() => backdrop.remove(), 0);
      backdrop.querySelectorAll("[data-act]").forEach((btn) => {
        btn.addEventListener("click", () => { const i = +btn.dataset.act; close(); resolve(i); });
      });
      backdrop.addEventListener("click", (e) => { if (e.target === backdrop) { close(); resolve(-1); } });
    });
  }

  // 确认框
  async function confirmDialog(message) {
    const i = await modal("确认操作", "<p>" + escapeHtml(message) + "</p>", [
      { label: "取消", cls: "btn-ghost" },
      { label: "确定", cls: "btn-danger" },
    ]);
    return i === 1;
  }

  global.common = {
    NAV, PAGE_TITLES, renderShell, toast, fmt, fmtPct, fmtMs, escapeHtml,
    colorFor, PALETTE, spinner, emptyState, renderTable, initPage, modal,
    confirmDialog, applyTheme, toggleTheme, refreshHealth,
  };
})(window);
