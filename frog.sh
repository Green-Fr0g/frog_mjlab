#!/usr/bin/env bash
#
# frog_mjlab 便捷入口脚本 —— 用法与 frog_lab/frog.sh 对齐。
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
用法:
  ./frog.sh -i                         安装依赖和本地 frog 包
  ./frog.sh -l                         列出所有 FrogMjlab 任务
  ./frog.sh -t [训练参数...]            执行 scripts/frog_rl/train.py
  ./frog.sh -p [推理参数...]            执行 scripts/frog_rl/play.py

环境变量:
  PYTHON                覆盖默认 Python 解释器（默认 python；建议指向 mjlab 的 venv）
  MJLAB_PATH            指向 mjlab 仓库根目录；若其下存在 .venv/bin/python 则自动使用
  FROG_NO_PYTHONPATH=1  不自动把 source/ 注入 PYTHONPATH（默认自动注入，免安装即可运行）

示例:
  ./frog.sh -t FrogMjlab-G1-Flat --env.scene.num-envs=4096
  ./frog.sh -p FrogMjlab-G1-Mimic --checkpoint-file logs/rsl_rl/g1_mimic/<run>/model_500.pt
EOF
}

# ---------------------------------------------------------------------------
# Python 解释器解析：PYTHON > MJLAB_PATH/.venv > python
# ---------------------------------------------------------------------------
if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_CMD=("$PYTHON")
elif [[ -n "${MJLAB_PATH:-}" && -x "${MJLAB_PATH}/.venv/bin/python" ]]; then
    PYTHON_CMD=("${MJLAB_PATH}/.venv/bin/python")
else
    PYTHON_CMD=(python)
fi

# 默认把 source/ 下两个包注入 PYTHONPATH，免安装即可运行
if [[ "${FROG_NO_PYTHONPATH:-0}" != "1" ]]; then
    export PYTHONPATH="$ROOT_DIR/source/frog_mjlab:$ROOT_DIR/source/frog_rl${PYTHONPATH:+:$PYTHONPATH}"
fi

run_python() {
    "${PYTHON_CMD[@]}" "$@"
}

install_dependencies() {
    run_python -m pip install -e "$ROOT_DIR/source/frog_mjlab" --no-deps
    run_python -m pip install -e "$ROOT_DIR/source/frog_rl"    --no-deps
}

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

mode="$1"
shift

case "$mode" in
    -i)
        [[ $# -eq 0 ]] || { echo "-i 不接受额外参数。" >&2; usage >&2; exit 2; }
        install_dependencies
        ;;
    -l)
        [[ $# -eq 0 ]] || { echo "-l 不接受额外参数。" >&2; usage >&2; exit 2; }
        run_python "$ROOT_DIR/scripts/list_envs.py" --keyword FrogMjlab
        ;;
    -t)
        run_python "$ROOT_DIR/scripts/frog_rl/train.py" "$@"
        ;;
    -p)
        run_python "$ROOT_DIR/scripts/frog_rl/play.py" "$@"
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo "未知选项: $mode" >&2
        usage >&2
        exit 2
        ;;
esac
