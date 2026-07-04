# LeRobot on NVIDIA Orin (Jetson)

在 NVIDIA Jetson Orin 上构建和运行 LeRobot v0.5.2 + SO101 自定义样例的指南。

## 环境要求

| 项目 | 要求 |
|------|------|
| 硬件 | Jetson Orin Nano / NX / AGX |
| JetPack | 6.x (L4T r36.4 推荐) |
| Docker | 已安装并配置 `nvidia` runtime |
| 网络 | 构建时需要 `--network=host`（Jetson iptables 限制） |
| 存储 | 建议将 Docker 数据目录放在 SSD 上 |

验证 Docker GPU 支持：

```bash
docker run --rm --runtime=nvidia dustynv/lerobot:r36.4-cu128-24.04 \
  python3 -c "import torch; print(torch.cuda.is_available())"
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `docker/Dockerfile.orin` | Orin 专用镜像构建文件 |
| `docker/requirements-orin.txt` | Orin 补装的 Python 依赖（不含 torch/torchcodec） |
| `docker/run-orin.sh` | 一键构建 + 运行容器（自动映射设备） |
| `docker/verify-orin.sh` | 容器内验证脚本 |
| `examples/phone_to_so101/` | 手机遥操作 SO101 样例 |
| `examples/so101_to_so101_EE/` | 主臂遥操作 SO101 样例 |

## 构建镜像

在仓库根目录执行：

```bash
docker build --network=host -f docker/Dockerfile.orin -t lerobot-orin .
```

或使用运行脚本自动构建：

```bash
./docker/run-orin.sh
```

强制重新构建：

```bash
LEROBOT_REBUILD=1 ./docker/run-orin.sh
```

预计构建时间：首次约 10–30 分钟（取决于网络和 Orin 型号）。

## 验证步骤（真机）

按顺序执行以下检查。全部通过后再接机械臂。

### 步骤 1：GPU 与 LeRobot 版本

```bash
./docker/run-orin.sh python3 -c "
import torch, lerobot
print('lerobot:', lerobot.__version__)
print('torch:  ', torch.__version__)
print('cuda:   ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device: ', torch.cuda.get_device_name(0))
"
```

预期输出：

- `lerobot: 0.5.2`
- `cuda: True`
- 显示 Orin GPU 设备名

### 步骤 2：运行完整验证脚本

```bash
./docker/run-orin.sh bash docker/verify-orin.sh
```

该脚本检查：

1. Python 环境
2. PyTorch CUDA 张量运算
3. LeRobot 导入
4. placo / rerun / datasets / PyAV 等 SO101 关键依赖
5. torchcodec 未安装（预期行为）
6. SO101 样例文件和自定义处理器
7. 串口 / 摄像头设备映射
8. SO101 URDF 挂载

### 步骤 3：准备 SO101 资源

```bash
# 确保 URDF 文件存在（从 SO-ARM100 仓库获取）
ls SO101/so101_new_calib.urdf
```

URDF 来源：[SO-ARM100 Simulation URDF](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf)

### 步骤 4：串口权限

```bash
# 查看机械臂串口
ls -l /dev/ttyACM*

# 如权限不足，添加用户到 dialout 组后重新登录
sudo usermod -aG dialout $USER
```

### 步骤 5：手机遥操作验证

```bash
./docker/run-orin.sh python3 examples/phone_to_so101/teleoperate.py
```

运行前修改 `examples/phone_to_so101/teleoperate.py` 中的配置：

- `ROBOT_PORT`：串口路径（默认 `/dev/ttyACM0`）
- `PHONE_OS`：`PhoneOS.IOS` 或 `PhoneOS.ANDROID`

### 步骤 6：主臂遥操作验证

```bash
./docker/run-orin.sh python3 examples/so101_to_so101_EE/teleoperate.py
```

需要 leader 和 follower 两个串口（通常 `/dev/ttyACM0` 和 `/dev/ttyACM1`）。

### 步骤 7：录制 / 评估（可选）

```bash
# 录制
./docker/run-orin.sh python3 examples/phone_to_so101/record.py

# 回放
./docker/run-orin.sh python3 examples/phone_to_so101/replay.py

# 策略评估（需提前下载模型到 ~/.cache/huggingface）
./docker/run-orin.sh python3 examples/phone_to_so101/evaluate.py
```

## 运行脚本参数

`run-orin.sh` 自动处理：

- USB 串口 (`/dev/ttyACM*`, `/dev/ttyUSB*`)
- 摄像头 (`/dev/video*`)
- X11 显示 (`$DISPLAY`)
- SO101 目录挂载 (`./SO101` → `/opt/lerobot/SO101`)
- HuggingFace 缓存 (`~/.cache/huggingface`)
- LeRobot 缓存 (`~/.cache/lerobot`)

环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LEROBOT_IMAGE` | `lerobot-orin` | 镜像名称 |
| `LEROBOT_REBUILD` | 未设置 | 设为 `1` 强制重新构建 |

## 设计说明

### 为什么不安装 torchcodec？

lerobot v0.5.2 的 `[dataset]` extra 在 aarch64 上要求 `torchcodec>=0.11`，而 torchcodec 0.11 需要 `torch>=2.11`。Jetson 基础镜像提供的 PyTorch 通常为 2.7–2.8，无法满足该约束。

LeRobot 在缺少 torchcodec 时自动回退到 **PyAV** 解码器，不影响 SO101 录制和回放功能。

### 为什么不升级 PyTorch？

PyPI 上的 PyTorch wheel 不适用于 Jetson。必须使用 dustynv 预编译的 Jetson 优化版。Dockerfile 通过以下方式保护基础镜像的 PyTorch：

1. 从 `pyproject.toml` 剥离 `torch` / `torchvision` / `opencv`
2. `pip install --no-deps -e .` 安装 lerobot 本体
3. 通过 `requirements-orin.txt` 仅补装轻量依赖

### v0.4.4 → v0.5.2 主要变化

v0.5.2 将 `rerun-sdk`、`datasets`、`pynput`、`placo` 等从核心依赖拆到 optional extras。旧 Dockerfile 仅安装 `[feetech,phone]`，合并后会导致 SO101 样例缺少关键包。新 Dockerfile 通过 `requirements-orin.txt` 一次性补全。

## 故障排查

### 构建失败：pip 尝试安装 torch

确认 Dockerfile 中的 `sed` 步骤成功剥离了 `torch` 行。手动检查：

```bash
docker run --rm lerobot-orin grep -n '"torch' /opt/lerobot/pyproject.toml
# 应无输出
```

### CUDA not available

```bash
# 检查 nvidia runtime
docker info | grep -i runtime

# 确认使用 --runtime=nvidia
docker run --rm --runtime=nvidia lerobot-orin nvidia-smi
```

### 串口找不到

```bash
# 宿主机检查
ls /dev/ttyACM*

# 确认容器内映射
./docker/run-orin.sh ls -l /dev/ttyACM*
```

### placo 导入失败

Ubuntu 24.04 上 placo 依赖特定版本的 `cmeel-urdfdom` 和 `cmeel-tinyxml2`。`requirements-orin.txt` 已包含兼容版本约束。如仍失败：

```bash
./docker/run-orin.sh pip show placo cmeel-urdfdom cmeel-tinyxml2
```

### 视频解码慢

未安装 torchcodec 时，视频解码走 PyAV 软件路径，比 torchcodec 慢。对 SO101 录制/回放功能无影响，仅影响性能。

## 上游参考

- [dusty-nv/jetson-containers — lerobot](https://github.com/dusty-nv/jetson-containers/tree/master/packages/physicalAI/lerobot)
- [NVIDIA Jetson AI Lab — LeRobot](https://www.jetson-ai-lab.com/archive/lerobot.html)
- [LeRobot 上游 Docker 文档](README.md)
