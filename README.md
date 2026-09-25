# 社交网络图分析与推荐系统

一个 **零第三方依赖**（Python 后端纯标准库，前端仅引 vis.js CDN）的社交网络图分析
与推荐系统。前端 10 个页面覆盖用户管理、关系导入、图可视化、路径与共同好友、
社群发现、个性化推荐、统计面板、系统设置、数据导出与标签管理；后端实现邻接表图
构建、BFS 最短路径、PageRank、Louvain 社群划分，以及协同过滤 + 图嵌入 + 标签的
混合推荐。

---

## 快速开始

```bash
cd gsb3

# 1) 启动服务（空数据时自动生成演示数据）
python3 backend/run.py --seed

# 2) 打开浏览器
#    http://127.0.0.1:8080        （自动跳转到图可视化）
```

自定义端口 / 主机：

```bash
GSB_PORT=9000 GSB_HOST=0.0.0.0 python3 backend/run.py --seed
```

运行算法自测（校验 BFS / PageRank / Louvain / 推荐链路）：

```bash
python3 backend/run.py --check
```

> 前端通过 `https://unpkg.com/vis-network@9.1.9/...` 加载 vis.js（需联网）。
> 后端本身无任何 pip 依赖，仅需 Python 3.9+（开发环境为 3.12）。

---

## 目录结构

```
gsb3/
├── backend/                    # Python 后端（纯标准库）
│   ├── config.py               # 配置、设置存储、路径、原子写工具
│   ├── graph.py                # 内存高效 CSR 图（邻接表冻结为压缩数组）
│   ├── algorithms.py           # BFS/双向BFS、PageRank、Louvain、推荐算法
│   ├── storage.py              # 分片 JSON 邻接表存储、索引、增量合并
│   ├── service.py              # 业务服务层（缓存、CRUD、算法调度）
│   ├── api.py                  # HTTP 服务 + REST 路由 + 静态托管
│   ├── seed.py                 # 演示数据生成器
│   └── run.py                  # 入口（含 --check / --seed）
├── frontend/                   # 前端（10 页面 + 共享资源）
│   ├── index.html              # 入口（跳转 graph.html）
│   ├── users.html              # 1. 用户管理
│   ├── import.html             # 2. 关系导入
│   ├── graph.html              # 3. 图可视化（vis.js 缩放拖拽、路径高亮）
│   ├── path.html               # 4. 最短路径与共同好友查询
│   ├── community.html          # 5. 社群发现（Louvain 着色）
│   ├── recommend.html          # 6. 个性化推荐列表
│   ├── stats.html              # 7. 统计面板
│   ├── settings.html           # 8. 系统设置
│   ├── export.html             # 9. 数据导出
│   ├── tags.html               # 10. 标签管理
│   ├── css/style.css           # 设计系统（明暗双主题）
│   └── js/                     # api.js（客户端）+ common.js（外壳/工具）
└── data/                       # 运行期生成（分片图、画像、推荐、社群…）
    ├── graph/shard_XXXX.json   # 按用户分片的邻接表
    ├── users.json              # 用户档案
    ├── profiles.json           # 用户画像（预留扩展）
    ├── tags.json               # 标签体系
    ├── recommendations.json    # 推荐结果（单独存储）
    ├── community.json          # Louvain 结果缓存
    ├── pagerank.json           # PageRank 结果缓存
    ├── index.json              # 用户 → 分片 索引
    └── settings.json           # 系统设置
```

---

## 架构要点

### 1. 内存高效图存储（`graph.py`）

构建期使用 `dict[node -> dict[neighbour -> weight]]` 方便增删，随后 `freeze()` 将其
压缩为 **CSR（Compressed Sparse Row）**：`offsets` 与 `neighbors` 用 `array('q')`
存储（每 id 8 字节的 C 数组），权重用 `array('d')`。相比朴素 `dict[int, list[int]]`
（每整数 ~28 字节对象 + 列表指针），内存占用约降 3.5 倍，且邻居按 id 排序、去重，
保证算法可复现、利于二分查找与缓存友好。

### 2. 分片 JSON 邻接表（`storage.py`）

- 边按 `user % SHARD_COUNT` 分片到 64 个 JSON 文件，每个文件自包含（shard 元信息 +
  `edges` + 反范式化的 `users`）。
- **快速加载**：算法可逐分片流式加载，或按需加载单个分片的邻域，避免整图一次性解析。
- **增量导入**：新边只落到受影响分片，合并去重后写回；`index.json` 维护 O(1) 的
  `user -> shard` 映射，日常查询零扫描。
- **合并 / 索引重建**：`merge_shards()` 全量重写为规范形式（排序、去重），随后
  `rebuild_index_from_shards()` 重建唯一节点计数与分片映射。

### 3. 算法（`algorithms.py`）

| 算法 | 实现要点 |
| --- | --- |
| 最短路径 | 经典 BFS + **双向 BFS**（大图自动切换，搜索面 O(b^(d/2))） |
| PageRank | 幂迭代，显式处理 dangling 节点，O(n) 内存，L1 收敛判定 |
| Louvain | 两阶段模块度优化：局部移动（ΔQ 增量公式）+ 聚合，迭代至收敛，固定种子可复现，`min_improvement` 早停 |
| 协同过滤 | 朋友的朋友 + Adamic-Adar 权重去偏，仅依赖邻域规模 |
| 图嵌入 | 距离-地标（landmark）定位嵌入：L 次有界 BFS 得到低维向量，捕捉结构相似性，无需神经网络训练 |
| 冷启动 | 好友数低于阈值时退化为「热门 + 标签重叠」 |
| 多样性 | MMR 最大边际相关性重排序，λ 权衡相关性与多样性 |

### 4. 数据分层

图数据（分片邻接表）与派生数据（`recommendations.json` / `profiles.json` /
`community.json` / `pagerank.json`）**分开存储**：图变更只触发图分片的增量写与索引
刷新；推荐与社群结果作为缓存持久化，命中后零计算。

---

## REST API 摘要

所有接口返回 JSON（导出返回附件）。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查 |
| GET/POST | `/api/users` | 用户列表（分页/搜索/标签过滤）/ 创建 |
| GET/PUT/DELETE | `/api/users/<id>` | 用户详情 / 更新 / 删除 |
| POST | `/api/users/<id>/tags` | 设置用户标签 |
| POST | `/api/import` | 批量导入边 |
| GET | `/api/graph` · `/api/graph/neighborhood` | 全图 / 邻域子图 |
| GET | `/api/path` · `/api/common-friends` | 最短路径 / 共同好友 |
| GET/POST | `/api/community` · `/api/community/compute` | Louvain 结果 / 重算 |
| GET | `/api/pagerank?top=` | PageRank 中心性 |
| GET/POST | `/api/recommend/<id>` · `/api/recommend` | 单用户 / 批量推荐 |
| GET | `/api/stats` | 统计面板聚合 |
| GET/PUT | `/api/settings` | 读取 / 保存设置 |
| GET/POST/DELETE | `/api/tags` | 标签管理 |
| GET | `/api/export?format=json\|graphml\|csv` | 导出 |
| POST | `/api/graph/rebuild-index` · `/api/graph/merge` | 索引重建 / 分片合并 |
| POST | `/api/seed` | 生成演示数据 |

---

## 难点与应对

| 难点 | 应对方案 |
| --- | --- |
| **大规模图存储与快速加载** | 分片 JSON + 索引映射 O(1) 定位；CSR 压缩内存；按需/流式加载而非整图解析 |
| **图算法内存高效执行** | CSR 数组替代对象图；PageRank/Louvain 全程稀疏、避免 N×N 矩阵；双向 BFS 收缩搜索前沿 |
| **社群发现迭代收敛优化** | ΔQ 增量增益、`min_improvement` 早停、迭代轮次上限、固定种子保证可复现 |
| **推荐冷启动** | 好友数阈值判断，冷启动回退「热门 + 标签」；标签信号贯穿混合策略 |
| **推荐多样性** | MMR 重排序，λ 可调，兼顾相关性与覆盖 |
| **增量更新文件合并与索引重建** | 增量导入仅触达受影响分片；`merge_shards` 去重排序；`rebuild_index` 重建唯一计数与映射 |

---

