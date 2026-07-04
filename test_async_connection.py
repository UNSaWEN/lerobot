#!/usr/bin/env python3
"""
测试 RobotClient 到远程 PolicyServer 的异步推理连接。

测试分为三个阶段：
  1. gRPC 基础连通性测试（Ready 握手）
  2. PolicyInstructions 发送测试（需要服务器能加载模型）
  3. 完整异步推理管线测试（观测→推理→动作回传）

使用 MockRobot 模拟机器人（无需真实硬件），可单独运行连通性测试。
"""

import argparse
import logging
import sys
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("async_test")


def test_grpc_connectivity(server_address: str) -> bool:
    """阶段1：测试基础 gRPC 连通性（Ready 握手）"""
    import grpc
    from google.protobuf import empty_pb2

    from lerobot.transport import services_pb2_grpc
    from lerobot.transport.utils import grpc_channel_options

    logger.info("=" * 60)
    logger.info(f"阶段1：测试 gRPC 连通性 → {server_address}")
    logger.info("=" * 60)

    try:
        channel = grpc.insecure_channel(server_address, grpc_channel_options())
        stub = services_pb2_grpc.AsyncInferenceStub(channel)

        start = time.perf_counter()
        stub.Ready(empty_pb2.Empty(), timeout=10)
        elapsed = time.perf_counter() - start

        logger.info(f"[成功] Ready 握手成功！延迟: {elapsed * 1000:.2f}ms")
        channel.close()
        return True

    except grpc.RpcError as e:
        logger.error(f"[失败] gRPC 连接失败: {e}")
        logger.error("请确认：")
        logger.error(f"  1. PolicyServer 正在 {server_address} 上运行")
        logger.error("  2. 防火墙允许该端口的流量")
        logger.error("  3. 网络连通（可 ping 试试）")
        return False

    except Exception as e:
        logger.error(f"[失败] 未预期的错误: {e}")
        return False


def test_full_async_inference(
    server_address: str,
    policy_type: str,
    pretrained_path: str,
    policy_device: str,
    actions_per_chunk: int,
    run_duration: float,
) -> bool:
    """阶段2+3：测试完整异步推理管线"""
    from lerobot.async_inference.configs import RobotClientConfig
    from lerobot.async_inference.robot_client import RobotClient

    # 动态导入 MockRobot（在 tests 目录中）
    sys.path.insert(0, ".")
    from tests.mocks.mock_robot import MockRobotConfig

    logger.info("=" * 60)
    logger.info("阶段2：发送 PolicyInstructions（服务器将加载模型）")
    logger.info("=" * 60)
    logger.info(f"  policy_type:  {policy_type}")
    logger.info(f"  pretrained:   {pretrained_path}")
    logger.info(f"  device:       {policy_device}")
    logger.info(f"  actions/chunk:{actions_per_chunk}")

    # 使用 MockRobot（6个电机，无摄像头）来模拟机器人
    robot_config = MockRobotConfig(n_motors=6)

    client_config = RobotClientConfig(
        server_address=server_address,
        robot=robot_config,
        chunk_size_threshold=0.0,  # 总是发送观测
        policy_type=policy_type,
        pretrained_name_or_path=pretrained_path,
        actions_per_chunk=actions_per_chunk,
        policy_device=policy_device,
        client_device="cpu",
        fps=30,
    )

    try:
        client = RobotClient(client_config)
    except Exception as e:
        logger.error(f"[失败] 创建 RobotClient 失败: {e}")
        return False

    logger.info("RobotClient 创建成功，尝试 start()...")

    if not client.start():
        logger.error("[失败] client.start() 失败（Ready 或 SendPolicyInstructions 失败）")
        return False

    logger.info("[成功] client.start() 成功！服务器已加载模型。")

    # 阶段3：运行异步推理
    logger.info("=" * 60)
    logger.info(f"阶段3：运行完整异步推理管线 ({run_duration}秒)")
    logger.info("=" * 60)

    action_chunks_received = {"count": 0}

    # 包装 receive_actions 以计数
    original_aggregate = client._aggregate_action_queues

    def counting_aggregate(*args, **kwargs):
        action_chunks_received["count"] += 1
        logger.info(f"  收到动作块 #{action_chunks_received['count']}")
        return original_aggregate(*args, **kwargs)

    client._aggregate_action_queues = counting_aggregate  # type: ignore[method-assign]

    # 启动线程
    action_thread = threading.Thread(target=client.receive_actions, daemon=True)
    control_thread = threading.Thread(target=client.control_loop, args=("测试任务",), daemon=True)

    action_thread.start()
    control_thread.start()

    logger.info(f"管线已启动，运行 {run_duration} 秒...")

    # 等待指定时间
    time.sleep(run_duration)

    # 停止
    client.stop()
    action_thread.join(timeout=5)
    control_thread.join(timeout=5)

    logger.info("=" * 60)
    logger.info("测试结果")
    logger.info("=" * 60)
    logger.info(f"  收到动作块数: {action_chunks_received['count']}")
    logger.info(f"  动作队列记录: {len(client.action_queue_size)} 条")

    if action_chunks_received["count"] > 0:
        logger.info("[成功] 异步推理管线工作正常！")
        return True
    else:
        logger.warning("[警告] 未收到任何动作块。可能原因：")
        logger.warning("  - 服务器模型加载失败")
        logger.warning("  - 观测格式与模型不匹配（MockRobot 的特征可能不符）")
        logger.warning("  - 网络延迟过大")
        return False


def main():
    parser = argparse.ArgumentParser(description="测试 RobotClient 到远程 PolicyServer 的异步推理")
    parser.add_argument("--host", default="192.168.50.3", help="PolicyServer 主机地址")
    parser.add_argument("--port", type=int, default=8080, help="PolicyServer 端口")
    parser.add_argument(
        "--connectivity-only",
        action="store_true",
        help="仅测试 gRPC 连通性（不发送 PolicyInstructions 和推理）",
    )
    parser.add_argument("--policy-type", default="act", help="策略类型 (act, smolvla, diffusion, pi0...)")
    parser.add_argument("--pretrained-path", default="", help="预训练模型路径或 HuggingFace 名称")
    parser.add_argument("--policy-device", default="cuda", help="服务器端推理设备 (cuda/cpu/mps)")
    parser.add_argument("--actions-per-chunk", type=int, default=20, help="每个块的动作数")
    parser.add_argument("--duration", type=float, default=10.0, help="运行异步推理的时间（秒）")

    args = parser.parse_args()
    server_address = f"{args.host}:{args.port}"

    logger.info(f"目标 PolicyServer: {server_address}")

    # 阶段1：连通性测试
    if not test_grpc_connectivity(server_address):
        logger.error("连通性测试失败，终止。")
        sys.exit(1)

    if args.connectivity_only:
        logger.info("仅连通性测试模式，测试完成。")
        sys.exit(0)

    # 阶段2+3：完整异步推理测试
    if not args.pretrained_path:
        logger.error("需要指定 --pretrained-path 才能进行完整推理测试。")
        logger.info("示例: python test_async_connection.py --pretrained-path user/model_name")
        logger.info("如仅测试连通性，请使用 --connectivity-only")
        sys.exit(1)

    success = test_full_async_inference(
        server_address=server_address,
        policy_type=args.policy_type,
        pretrained_path=args.pretrained_path,
        policy_device=args.policy_device,
        actions_per_chunk=args.actions_per_chunk,
        run_duration=args.duration,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
