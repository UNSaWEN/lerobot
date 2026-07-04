#!/bin/bash
# LeRobot Orin Docker 运行脚本
# 用法: ./docker/run-orin.sh [命令]
#
# 示例:
#   ./docker/run-orin.sh                                    # 进入交互式 shell
#   ./docker/run-orin.sh bash docker/verify-orin.sh         # 运行验证脚本
#   ./docker/run-orin.sh python3 examples/phone_to_so101/teleoperate.py
#
# 详细说明见 docker/README.orin.md

set -e

# 镜像名称
IMAGE_NAME="${LEROBOT_IMAGE:-lerobot-orin}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# 检查镜像是否存在，或通过 LEROBOT_REBUILD=1 强制重建
if [ "${LEROBOT_REBUILD:-}" = "1" ] || ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    if [ "${LEROBOT_REBUILD:-}" = "1" ]; then
        echo "LEROBOT_REBUILD=1，正在重新构建镜像 $IMAGE_NAME ..."
    else
        echo "镜像 $IMAGE_NAME 不存在，正在构建..."
    fi
    docker build --network=host -f "$SCRIPT_DIR/Dockerfile.orin" -t "$IMAGE_NAME" "$REPO_ROOT"
fi

# 检测可用的设备
DEVICE_ARGS=""

# USB 串口设备 (机械臂)
for dev in /dev/ttyACM* /dev/ttyUSB*; do
    if [ -e "$dev" ]; then
        DEVICE_ARGS="$DEVICE_ARGS --device=$dev"
        echo "检测到设备: $dev"
    fi
done

# 摄像头设备
for dev in /dev/video*; do
    if [ -e "$dev" ]; then
        DEVICE_ARGS="$DEVICE_ARGS --device=$dev"
        echo "检测到摄像头: $dev"
    fi
done

# X11 显示支持
DISPLAY_ARGS=""
if [ -n "$DISPLAY" ]; then
    DISPLAY_ARGS="-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY"
    xhost +local:docker 2>/dev/null || true
fi

# 运行容器
echo "启动 LeRobot 容器..."
docker run -it --rm \
    --runtime=nvidia \
    --network=host \
    $DEVICE_ARGS \
    $DISPLAY_ARGS \
    -v "$REPO_ROOT/SO101:/opt/lerobot/SO101" \
    -v "${HOME}/.cache/huggingface:/data/models/huggingface" \
    -v "${HOME}/.cache/lerobot:/root/.cache/lerobot" \
    -w /opt/lerobot \
    "$IMAGE_NAME" \
    "${@:-/bin/bash}"
