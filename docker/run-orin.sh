#!/bin/bash
# LeRobot Orin Docker 运行脚本
# 用法: ./docker/run-orin.sh [命令]
#
# 示例:
#   ./docker/run-orin.sh                           # 进入交互式 shell
#   ./docker/run-orin.sh python3 -c "import torch; print(torch.cuda.is_available())"
#   ./docker/run-orin.sh python3 examples/phone_to_so101/teleoperate.py

set -e

# 镜像名称
IMAGE_NAME="${LEROBOT_IMAGE:-lerobot-orin}"

# 检查镜像是否存在
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "镜像 $IMAGE_NAME 不存在，正在构建..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    docker build --network=host -f "$SCRIPT_DIR/Dockerfile.orin" -t "$IMAGE_NAME" "$(dirname "$SCRIPT_DIR")"
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
    -v "$(pwd)/SO101:/opt/lerobot/SO101" \
    -v "${HOME}/.cache/huggingface:/data/models/huggingface" \
    -v "${HOME}/.cache/lerobot:/root/.cache/lerobot" \
    -w /opt/lerobot \
    "$IMAGE_NAME" \
    "${@:-/bin/bash}"
