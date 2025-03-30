#!/bin/bash

# 检查是否安装了 uv
if ! command -v uv &> /dev/null; then
  echo "uv is not installed. Please install it first."
  exit 1
fi

# 定义一个打开网页的函数，根据操作系统选择合适的命令
open_browser() {
  case "$OSTYPE" in
    darwin*) open "http://127.0.0.1:8000" ;; # macOS
    linux*) xdg-open "http://127.0.0.1:8000" ;; # Linux
    msys*) start "http://127.0.0.1:8000" ;; # Windows (Git Bash)
    cygwin*) start "http://127.0.0.1:8000" ;; # Windows (Cygwin)
    *) echo "Cannot automatically open browser on this operating system." ;;
  esac
}

case "$1" in
  build)
    uv run -- sphinx-build -M html . _build/
    ;;
  dev)
    # 使用 & 在后台运行 sphinx-autobuild，并保存其进程 ID
    uv run -- sphinx-autobuild . _build/html &
    sphinx_autobuild_pid=$!

    # 延迟一段时间，确保 sphinx-autobuild 启动完成
    sleep 2

    open_browser

    # 添加一个 trap 命令，以便在脚本退出时终止 sphinx-autobuild 进程
    trap "kill $sphinx_autobuild_pid" EXIT

    # 添加一个循环，以便用户可以通过 Ctrl+C 退出
    while true; do
      sleep 1 # 每秒检查一次
    done
    ;;
  *)
    echo "Usage: $0 {build|dev}"
    exit 1
    ;;
esac
