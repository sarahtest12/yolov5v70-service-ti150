# BI-V150 GPU gRPC 检测服务开发与运行

> 本文只针对独立项目 `gpu-grpc-service`。服务运行时不读取或导入外部
> YOLOv5 仓库，默认使用 CoreX PyTorch、GPU 0、FP16、单帧批次。

## 1. 已实现范围

- gRPC 双向流 `detector.v1.Detector/Detect`。
- JPEG 大小、字段、真实解码尺寸校验。
- YOLOv5 模型启动时加载一次并完成模型与 NMS warmup。
- FP32/FP16、letterbox、推理、NMS、原图坐标恢复和归一化 `xyxy`。
- 全局有界队列及可配置微批处理；队列满时返回 `OVERLOADED`。
- 到达服务时已经过旧或在 GPU 队列等待超时的帧返回 `EXPIRED`。
- 单帧错误返回结果错误码，不关闭整条 gRPC 流。
- 可选 Bearer token 鉴权。
- gRPC 标准健康检查。
- HTTP `/health/live`、`/health/ready` 和 `/metrics`。
- 模型 SHA-256 校验、结构化进程日志、固定 JPEG 测试客户端。

## 2. 模块与接口

```text
gRPC stream
    │
    ▼
GrpcDetectorServicer         协议、鉴权、校验和错误码 adapter
    │ submit(frame)
    ▼
InferenceScheduler           有界队列、过载、串行 GPU 访问、微批处理
    │ detect_batch(frames)
    ▼
YoloDetector                 JPEG → tensor → YOLOv5 → NMS → normalized xyxy
```

外部推理 seam 只有 `YoloDetector.detect_batch()`；传输层不接触 letterbox、Tensor 或 NMS。调度 seam 只有同步的 `submit()`，gRPC handler 不需要了解队列线程和 batch 形成规则。

主要文件：

- `shared/detector_contract/detector.proto`：CPU/GPU 唯一共享契约。
- `shared/detector_contract/config.py`：两端公共通信配置定义。
- `src/gpu_detector/detector.py`：模型推理模块。
- `src/gpu_detector/scheduler.py`：有界微批调度模块。
- `src/gpu_detector/grpc_server.py`：gRPC adapter。
- `src/gpu_detector/application.py`：进程生命周期。
- `src/gpu_detector/client.py`：测试客户端。
- `src/models/`、`src/utils/`：项目内置的 YOLOv5 推理实现快照。

## 3. 开发环境

每个新终端先执行：

```bash
cd /path/to/gpu_grpc_service
scripts/bootstrap_corex.sh
```

开发目录可以移动，但 `.venv` 不能跟随项目复制后继续使用。bootstrap
检测到环境中的旧绝对路径时，会把旧环境保存为 `.venv.stale-*` 并重建。
生产环境仍统一安装到 `/opt/gpu-grpc-service`。

安装或核对应用依赖：

```bash
source scripts/corex_env.sh
python -m pip install -r requirements-corex.txt
python -m pip install --no-deps --editable .
```

修改 `shared/detector_contract/detector.proto` 后重新生成 stub：

```bash
source scripts/corex_env.sh
scripts/generate_detector_proto.sh
```

生成端和运行端必须保持 `grpcio==1.83.1`、`grpcio-tools==1.83.1`、`protobuf==7.36.1`。

## 4. 本地启动

CPU 服务器或 Windows 本地联调请复制项目内同级的 `cpu_client/` 和 `shared/` 源码目录，
按 [`cpu_client/README.md`](../cpu_client/README.md) 创建普通 CPU 虚拟环境、
修改 `config.json` 并执行 `demo.py`。无需向 CPU 复制 GPU 交付压缩包。
下面的 CoreX 启动和旧客户端命令仅用于 GPU 本机开发环境。

开发模式至少设置一个 token：

```bash
export DETECTOR_AUTH_TOKEN='replace-with-a-development-secret'
scripts/run_gpu_detector.sh
```

默认监听：

- gRPC：`0.0.0.0:50051`
- HTTP 健康检查：`127.0.0.1:8081`
- GPU：设备 `0`
- 模型：项目内 `models/yolov5s.pt`

另开终端测试：

```bash
source scripts/corex_env.sh

curl -fsS http://127.0.0.1:8081/health/live
curl -fsS http://127.0.0.1:8081/health/ready
curl -fsS http://127.0.0.1:8081/metrics | grep detector_ready

python -m gpu_detector.client \
  --target 127.0.0.1:50051 \
  --token "$DETECTOR_AUTH_TOKEN" \
  --image tests/fixtures/bus.jpg \
  --count 3
```

客户端对每帧输出 JSON，包含 `stream_id`、`frame_id`、检测类别、置信度、归一化框和三个阶段耗时。

模拟 CPU 后端以单路 5 FPS 持续联调：

```bash
python -m gpu_detector.continuous_client \
  --target 127.0.0.1:50051 \
  --token "$DETECTOR_AUTH_TOKEN" \
  --streams 1 \
  --fps 5 \
  --duration 30
```

输出汇总包含发送/完成数量、各结果码数量、检测框总数、有效完成 FPS 和
客户端观测到的 p50/p95/p99 延迟。该工具默认是联调负载，不是极限压测。

## 5. 配置

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `DETECTOR_GRPC_HOST` | `0.0.0.0` | gRPC 监听地址 |
| `DETECTOR_GRPC_PORT` | `50051` | gRPC 端口 |
| `DETECTOR_HEALTH_HOST` | `127.0.0.1` | HTTP 健康检查地址 |
| `DETECTOR_HEALTH_PORT` | `8081` | HTTP 健康检查端口 |
| `DETECTOR_DEVICE` | `0` | 当前进程使用的 GPU |
| `DETECTOR_WEIGHTS` | `models/yolov5s.pt` | 可信模型绝对路径 |
| `DETECTOR_WEIGHTS_SHA256` | 空 | 非空时启动必须匹配 |
| `DETECTOR_DATA` | `config/coco80.yaml` | 类别配置 |
| `DETECTOR_IMAGE_SIZE` | `640` | 推理输入尺寸 |
| `DETECTOR_CONF_THRESHOLD` | `0.25` | 置信度阈值 |
| `DETECTOR_IOU_THRESHOLD` | `0.45` | NMS IoU 阈值 |
| `DETECTOR_MAX_DETECTIONS` | `300` | 单帧最大框数 |
| `DETECTOR_FP16` | `true` | 使用 FP16 |
| `DETECTOR_BATCH_SIZE` | `1` | 最大微批大小 |
| `DETECTOR_BATCH_WAIT_MS` | `0` | 等待同批后续帧的最长时间 |
| `DETECTOR_QUEUE_CAPACITY` | `8` | 全局待处理帧上限 |
| `DETECTOR_MAX_QUEUE_WAIT_MS` | `250` | 最大 GPU 排队时间；`0` 禁用 |
| `DETECTOR_MAX_FRAME_AGE_MS` | `1000` | 到达时允许的最大帧龄；`0` 禁用 |
| `DETECTOR_MAX_FRAME_BYTES` | `4194304` | 单帧 JPEG 上限 |
| `DETECTOR_MAX_DIMENSION` | `16384` | 声明宽高上限 |
| `DETECTOR_GRPC_WORKERS` | `16` | 同时占用的 gRPC stream 线程上限 |
| `DETECTOR_AUTH_TOKEN` | 空 | Bearer token；生产环境必须设置 |

第一阶段保持 `BATCH_SIZE=1` 和 `BATCH_WAIT_MS=0`。批量值调整必须重新测试延迟、吞吐和显存。

## 6. 协议约束

- CPU 请求 metadata：`authorization: Bearer <DETECTOR_AUTH_TOKEN>`。
- 单连接内按请求顺序返回结果。
- `width`、`height` 必须与 JPEG 实际解码尺寸一致。
- `frame_id` 必须大于 0；同一路流由 CPU 保证单调递增。
- `time_base_num` 和 `time_base_den` 必须同时为 0，或同时为正数。
- 所有框为相对于解码帧的 `[0,1]` normalized `xyxy`。
- `OK + detections=[]` 表示正常但未发现目标。
- `INVALID_FRAME`、`OVERLOADED`、`EXPIRED`、`INFERENCE_ERROR` 是帧级结果，不关闭连接。
- `observed_at_unix_ms=0` 表示发送端没有提供时间，此时不检查到达帧龄。
- 启用帧龄限制时，CPU/GPU 服务器必须使用 NTP/chrony 保持时钟同步。
- 排队时限只取消尚未进入 GPU 的帧，不中断已经开始的 GPU kernel。
- Bearer token 错误是连接级 `UNAUTHENTICATED`。

## 7. 测试

```bash
source scripts/corex_env.sh

python -m unittest discover -s tests/gpu_detector -v
bash tests/tooling/test_corex_env_relocation.sh
python scripts/smoke_test_yolov5.py
python scripts/smoke_test_yolov5.py --half
RUN_GPU_INTEGRATION=1 python -m unittest tests.gpu_detector.test_gpu_integration -v
```

真实 gRPC 测试需要先启动服务，再运行第 4 节客户端。

## 8. systemd 部署

仓库提供：

- `deploy/systemd/gpu-detector.service`
- `deploy/systemd/gpu-detector.env.example`

需要管理员执行的安装步骤：

```bash
sudo useradd --system --home /opt/gpu-grpc-service --shell /usr/sbin/nologin gpu-detector
sudo chown -R gpu-detector:gpu-detector /opt/gpu-grpc-service
sudo install -m 0644 deploy/systemd/gpu-detector.service /etc/systemd/system/gpu-detector.service
sudo install -m 0600 deploy/systemd/gpu-detector.env.example /etc/gpu-detector.env
sudo editor /etc/gpu-detector.env
sudo systemctl daemon-reload
sudo systemctl enable --now gpu-detector
sudo systemctl status gpu-detector
```

启用前必须替换 token，并确认服务安装在 `/opt/gpu-grpc-service`。GPU gRPC
端口只允许 CPU 服务器私网 IP 访问；若两台服务器不在可信专网，需要在
gRPC 前增加 mTLS。

## 9. 独立性交付检查

交付前从一个不包含原 YOLOv5 仓库的临时目录验证压缩包：

```bash
scripts/build_delivery_bundle.sh
mkdir -p /tmp/gpu-detector-delivery-check
tar -xzf dist/gpu-grpc-service-0.2.0.tar.gz \
  -C /tmp/gpu-detector-delivery-check
cd /tmp/gpu-detector-delivery-check/gpu-grpc-service
python3 -m venv .venv
source scripts/corex_env.sh
python -m pip install -r requirements-corex.txt
python -m pip install --no-deps --editable .
scripts/generate_detector_proto.sh
python scripts/check_corex_env.py
python -m unittest discover -s tests/gpu_detector -v
```

交付物必须包含 `LICENSE`、`THIRD_PARTY_NOTICES.md`、`shared/`、`src/models/`、
`src/utils/` 和模型文件，且不得包含 `.venv` 或指向开发机原 YOLOv5
仓库的路径。

## 10. 双 GPU 扩展

第一版一个进程只使用一张卡。使用两张 BI-V150 时，运行两个独立实例：

```text
实例 A：DETECTOR_DEVICE=0，DETECTOR_GRPC_PORT=50051
实例 B：DETECTOR_DEVICE=1，DETECTOR_GRPC_PORT=50052
```

CPU 后端按稳定哈希 `stream_id -> instance` 路由。同一 GPU 不启动多个普通进程重复加载模型。
