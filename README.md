# BI-V150 GPU gRPC 检测服务

这是可独立交付的 GPU 检测服务项目。运行时不依赖原始 YOLOv5 仓库；所需的
YOLOv5 推理源码快照、模型、gRPC 契约、服务代码和测试均位于本目录内。

## 运行前提

- 天垓 BI-V150，驱动及 CoreX 4.4.0 SDK 已安装到 `/usr/local/corex`
- x86_64 Linux、Python 3.10
- 能够访问内部或公网 PyPI 镜像以安装应用层 Python 包

不要从公开 PyPI 安装 `torch`、`torchvision`、`torchaudio`、`triton`、
`numpy`、`cuda-*`、`cupy` 或 `nvidia-*`。这些加速包必须使用 CoreX SDK
随服务器提供的版本。

以上依赖约束针对 GPU 环境。CPU 端联调复制 [`cpu_client/`](cpu_client/README.md)
和 [`shared/`](shared/README.md) 两个同级文件夹，包含可编辑客户端、JSON 配置和共享接口模块。
不需要在 CPU 上安装本 GPU 项目或复制整个交付包。

## 首次部署

```bash
cd /opt/gpu-grpc-service
scripts/bootstrap_corex.sh
```

脚本会创建本项目自己的 `.venv`、安装固定版本的应用依赖、以 editable
方式安装服务、生成 protobuf Python 文件，并检查包来源与真实 GPU/NMS。
如果项目目录被移动，脚本会先把带有旧绝对路径的环境保存为
`.venv.stale-<时间>-<进程号>`，然后创建新的 `.venv`。虚拟环境不能作为
交付物复制到另一目录。

## 开发运行

```bash
cd /path/to/gpu_grpc_service
source scripts/corex_env.sh
export DETECTOR_AUTH_TOKEN='replace-with-a-development-secret'
scripts/run_gpu_detector.sh
```

另开一个终端：

```bash
cd /opt/gpu-grpc-service
source scripts/corex_env.sh
curl -fsS http://127.0.0.1:8081/health/ready
python -m gpu_detector.client \
  --target 127.0.0.1:50051 \
  --token "$DETECTOR_AUTH_TOKEN" \
  --image tests/fixtures/bus.jpg \
  --count 3
```

持续低负载联调（默认单路、5 FPS、10 秒）：

```bash
python -m gpu_detector.continuous_client \
  --target 127.0.0.1:50051 \
  --token "$DETECTOR_AUTH_TOKEN" \
  --streams 1 \
  --fps 5 \
  --duration 30
```

该命令用于模拟 CPU 后端的连续 gRPC 流并汇总完成帧数、结果码及
p50/p95/p99 延迟，不用于测量 YOLO 独占 GPU 时的极限吞吐。

## 验证与交付

```bash
source scripts/corex_env.sh
python -m unittest discover -s tests/gpu_detector -v
python scripts/smoke_test_yolov5.py --half
scripts/build_delivery_bundle.sh
```

无论开发目录名是什么，交付文件和压缩包内的顶层目录始终使用
`gpu-grpc-service`。

交付 `dist/gpu-grpc-service-0.2.0.tar.gz` 及其 `.sha256` 文件，不要交付
`.venv`。完整配置、协议与 systemd 部署说明见
[`docs/operations.md`](docs/operations.md)。

## 目录边界

- `src/gpu_detector/`：服务、调度、推理 adapter 和进程生命周期
- `shared/detector_contract/`：唯一的 CPU/GPU protobuf 契约、生成代码和公共通信配置类
- `cpu_client/`：CPU 客户端和联调配置，通过本地 Python 包依赖 `shared/`
- `src/models/`、`src/utils/`：内置 YOLOv5 推理实现快照
- `models/`、`config/`：可信模型和类别配置
- `deploy/systemd/`：客户服务器部署模板

项目包含 GPL-3.0 授权的 YOLOv5 代码，整体按 GPL-3.0-only 分发。交付时
必须保留源码和授权文件；详见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
