#!/usr/bin/env bash
#
# start.sh — 一键启动「社交网络图分析与推荐系统」前后端
#
# 前端由后端静态托管（frontend/ 目录），因此只需启动一个 Python 服务，
# 它会同时提供 REST API 与前端页面。
#
# 用法：
#   ./start.sh                     # 默认 127.0.0.1:8080，空数据自动灌入演示数据
#   ./start.sh --port 9000         # 自定义端口
#   ./start.sh --host 0.0.0.0      # 监听所有网卡（供局域网访问）
#   ./start.sh --no-seed           # 不自动灌入演示数据
#   ./start.sh --check             # 仅运行算法自测后退出
#
set -euo pipefail

# ---------------------------------------------------------------------------
# 解析参数 / 环境变量
# ---------------------------------------------------------------------------
HOST="${GSB_HOST:-127.0.0.1}"
PORT="${GSB_PORT:-8080}"
SEED="--seed"
CHECK=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)      HOST="$2"; shift 2 ;;
        --port)      PORT="$2"; shift 2 ;;
        --no-seed)   SEED="" ; shift 1 ;;
        --check)     CHECK="--check"; shift 1 ;;
        -h|--help)   sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "未知参数: $1（--help 查看用法）" >&2; exit 2 ;;
    esac
done

# 脚本所在目录（支持从任意路径调用）
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PY="${PYTHON:-python3}"

# 展示地址：0.0.0.0 不适合在浏览器里打开，统一用 127.0.0.1 展示
DISPLAY_HOST="$HOST"
[[ "$DISPLAY_HOST" == "0.0.0.0" || "$DISPLAY_HOST" == "::" ]] && DISPLAY_HOST="127.0.0.1"
FRONTEND_URL="http://${DISPLAY_HOST}:${PORT}/"

# ---------------------------------------------------------------------------
# 仅自测模式
# ---------------------------------------------------------------------------
if [[ -n "$CHECK" ]]; then
    "$PY" backend/run.py --check
    exit $?
fi

# ---------------------------------------------------------------------------
# 若服务已在运行，直接给出地址
# ---------------------------------------------------------------------------
if "$PY" - <<PYEOF "$PORT" >/dev/null 2>&1
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(1)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
    ok = True
except Exception:
    ok = False
finally:
    s.close()
sys.exit(0 if ok else 1)
PYEOF
then
    echo "⚠ 检测到端口 ${PORT} 已有服务在运行，直接使用："
    echo "  前端地址： ${FRONTEND_URL}"
    exit 0
fi

# ---------------------------------------------------------------------------
# 后台启动服务并等待就绪
# ---------------------------------------------------------------------------
echo "正在启动服务（host=${HOST} port=${PORT}）…"
GSB_HOST="$HOST" GSB_PORT="$PORT" "$PY" backend/run.py $SEED \
    > "${ROOT_DIR}/.start.log" 2>&1 &
SERVER_PID=$!

# 等待健康检查通过（最多 30 秒）
ready=0
for _ in $(seq 1 60); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "✗ 服务启动失败，日志如下：" >&2
        cat "${ROOT_DIR}/.start.log" >&2
        exit 1
    fi
    if "$PY" - <<PYEOF "$PORT" >/dev/null 2>&1
import json, sys, urllib.request
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/api/health", timeout=1) as r:
        json.load(r)
    ok = True
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PYEOF
    then
        ready=1
        break
    fi
    sleep 0.5
done

if [[ "$ready" -ne 1 ]]; then
    echo "✗ 服务未在 30 秒内就绪，日志如下：" >&2
    cat "${ROOT_DIR}/.start.log" >&2
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
fi

# ---------------------------------------------------------------------------
# 输出地址
# ---------------------------------------------------------------------------
echo ""
echo "✅ 服务已启动"
echo "   前端地址： ${FRONTEND_URL}          （自动跳转到图可视化）"
echo "   健康检查： http://${DISPLAY_HOST}:${PORT}/api/health"
echo "   停止服务： Ctrl+C"
echo ""

# 前台等待，Ctrl+C 时优雅退出
trap 'echo ""; echo "正在停止服务…"; kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; exit 0' INT TERM
wait "$SERVER_PID"
