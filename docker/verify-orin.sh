#!/usr/bin/env bash
# LeRobot Orin 容器内验证脚本
# 用法:
#   ./docker/run-orin.sh bash docker/verify-orin.sh
#   或在容器内: bash /opt/lerobot/docker/verify-orin.sh

set -euo pipefail

PASS=0
FAIL=0
WARN=0

ok()   { echo "[PASS] $*"; PASS=$((PASS + 1)); }
fail() { echo "[FAIL] $*"; FAIL=$((FAIL + 1)); }
warn() { echo "[WARN] $*"; WARN=$((WARN + 1)); }

section() {
    echo ""
    echo "======== $* ========"
}

section "1. 基础环境"
python3 --version || fail "python3 不可用"
ok "python3 可用"

section "2. GPU / CUDA"
python3 - <<'PY'
import sys
import torch

print(f"torch       : {torch.__version__}")
print(f"cuda avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda device : {torch.cuda.get_device_name(0)}")
    x = torch.randn(2, 2, device="cuda")
    print(f"cuda tensor : {x.device}")
else:
    print("CUDA 不可用 — 请检查 --runtime=nvidia 和 JetPack 驱动")
    sys.exit(1)
PY
ok "PyTorch CUDA 正常"

section "3. LeRobot 版本"
python3 - <<'PY'
import lerobot

print(f"lerobot version: {lerobot.__version__}")
PY
ok "lerobot 可导入"

section "4. SO101 关键依赖"
python3 - <<'PY'
import importlib
import sys

modules = {
    "placo": "kinematics (RobotKinematics)",
    "rerun": "visualization (init_rerun)",
    "datasets": "LeRobotDataset",
    "av": "PyAV video decoder (torchcodec fallback)",
    "serial": "pyserial (机械臂串口)",
    "pynput": "keyboard listener",
    "hebi": "phone teleop (hebi-py)",
}

failed = []
for mod, desc in modules.items():
    try:
        importlib.import_module(mod)
        print(f"  OK  {mod:12s} — {desc}")
    except ImportError as e:
        print(f"  FAIL {mod:12s} — {desc}: {e}")
        failed.append(mod)

if failed:
    sys.exit(1)
PY
ok "SO101 关键依赖齐全"

section "5. torchcodec 应未安装 (预期行为)"
if python3 -c "import torchcodec" 2>/dev/null; then
    warn "torchcodec 已安装 — aarch64 上可能与 Jetson PyTorch 版本不兼容"
else
    ok "torchcodec 未安装，将使用 PyAV 解码 (符合 Orin 策略)"
fi

section "6. 自定义 SO101 样例"
python3 - <<'PY'
import sys
from pathlib import Path

root = Path("/opt/lerobot")
examples = [
    root / "examples/phone_to_so101/teleoperate.py",
    root / "examples/phone_to_so101/so101_phone_processor.py",
    root / "examples/so101_to_so101_EE/teleoperate.py",
]
missing = [p for p in examples if not p.exists()]
for p in examples:
    status = "OK" if p.exists() else "MISSING"
    print(f"  {status}  {p}")

if missing:
    sys.exit(1)
PY
ok "SO101 样例文件存在"

section "7. 导入 SO101 处理器"
python3 - <<'PY'
import sys
sys.path.insert(0, "/opt/lerobot/examples/phone_to_so101")
from so101_phone_processor import MapPhoneActionToRobotActionSO101
print(f"processor: {MapPhoneActionToRobotActionSO101.__name__}")
PY
ok "SO101 自定义处理器可导入"

section "8. 串口 / 摄像头设备 (宿主机映射检查)"
for pattern in /dev/ttyACM* /dev/ttyUSB*; do
    [ -e "$pattern" ] && echo "  串口: $pattern" || true
done
for pattern in /dev/video*; do
    [ -e "$pattern" ] && echo "  摄像头: $pattern" || true
done
if ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | head -1 >/dev/null; then
    ok "检测到串口设备"
else
    warn "未检测到串口设备 — 遥操作前请确认 --device 映射"
fi
if ls /dev/video* 2>/dev/null | head -1 >/dev/null; then
    ok "检测到摄像头设备"
else
    warn "未检测到摄像头 — 录制/评估前请确认 --device 映射"
fi

section "9. SO101 URDF (可选)"
if [ -f /opt/lerobot/SO101/so101_new_calib.urdf ]; then
    ok "SO101 URDF 已挂载: /opt/lerobot/SO101/so101_new_calib.urdf"
else
    warn "SO101 URDF 未挂载 — 运行遥操作前请挂载: -v \$(pwd)/SO101:/opt/lerobot/SO101"
fi

echo ""
echo "========================================"
echo "验证完成: PASS=$PASS  FAIL=$FAIL  WARN=$WARN"
echo "========================================"

if [ "$FAIL" -gt 0 ]; then
    echo "存在失败项，请查看上方输出。"
    exit 1
fi

echo "全部通过。可继续真机遥操作验证，参见 docker/README.orin.md"
