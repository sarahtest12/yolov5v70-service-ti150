# CPU 联调源码模块

把 `cpu_client/` 和 `shared/` 两个文件夹一起复制到 Windows、WSL 或 Linux CPU 服务器，保持同级。
可直接打开源码修改；不需要复制 GPU 交付包、模型、原 YOLOv5 仓库或 GPU `.venv`。
Python 使用 3.10–3.12；CPU 端安装普通 PyPI 包，不运行 `bootstrap_corex.sh`。

如果本地已经复制过上一版，请先把旧的 `cpu_client/detector_contract/` 移到项目外备份，
再安装新版依赖，避免 Python 优先导入旧副本。新版该目录只存在于 `shared/` 下。

## 文件说明

```text
你的本地目录/
├── shared/                      CPU/GPU 共同依赖的唯一接口模块
│   ├── pyproject.toml
│   └── detector_contract/
│       ├── config.py            公共通信配置类
│       ├── detector.proto       可阅读的消息字段、结果码、RPC 定义
│       ├── __init__.py
│       ├── detector_pb2.py       已生成的消息类
│       └── detector_pb2_grpc.py  已生成的 gRPC 绑定
└── cpu_client/
    ├── README.md
    ├── requirements.txt         安装 ../shared 及 CPU 图片依赖
    ├── config.example.json      配置模板，复制为 config.json 后编辑
    ├── detector_client.py       Web 后端可直接复用的 gRPC 客户端
    ├── demo.py                  单帧 / 持续低帧率图片联调入口
    └── tests/test_client.py     不加载 GPU 的本地 gRPC 联调测试
```

你自己放一张 `test.jpg` 在本目录，或修改配置的 `image`。也支持 PNG 等
OpenCV 可解码的图片，示例会重新编码为 JPEG。这里没有附带二进制测试图片。

## Windows PowerShell

在复制出的 `cpu_client` 文件夹打开 PowerShell（依赖清单中的 `../shared` 按当前目录解析）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item config.example.json config.json
notepad config.json
```

按下一节填写配置，然后运行：

```powershell
# 如果在 config.json 里填了 token，可以省略这行。
$env:DETECTOR_AUTH_TOKEN = '与GPU服务一致的token'
.\.venv\Scripts\python.exe demo.py --config config.json
```

直接调用虚拟环境解释器，不需要修改 PowerShell 执行策略或设置 PYTHONPATH。

## Linux / WSL

```bash
cd /你的路径/cpu_client
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp config.example.json config.json
# 用编辑器修改 config.json，并放入 test.jpg。
export DETECTOR_AUTH_TOKEN='与GPU服务一致的token'
.venv/bin/python demo.py --config config.json
```

Windows 和 WSL 的虚拟环境分别创建，不要跨环境复制 `.venv`。

## 配置

| 字段 | 用途 |
|---|---|
| `target` | CPU 实际可达的 gRPC `主机:端口`，不加 `http://` |
| `token` | 与 GPU 的 `DETECTOR_AUTH_TOKEN` 一致；同名环境变量优先，包括空值 |
| `image` | 测试图片；相对路径按配置文件所在目录解析 |
| `stream_id` | 摄像头标识，本例默认 `cpu-test-camera` |
| `count` | 发送帧数，`1` 为单帧验证，默认 `15` |
| `fps` | 发送频率，默认每秒 `5` 帧 |
| `connect_timeout_seconds` | 建立 gRPC 连接的等待上限，默认 `5` 秒 |
| `rpc_timeout_seconds` | 整条 Detect 调用的时限，默认 `30` 秒；不是每帧时限 |

首次复制配置即可看到全部字段。`config.json` 已加入本目录的 Git 忽略规则，
GPU 交付脚本也排除该文件。不要在 `config.example.json` 中保存实际 token。

Windows 图片路径建议写成 `C:/images/test.jpg`；如果用反斜杠，在 JSON 中要写成 `C:\\images\\test.jpg`。

默认 `127.0.0.1:50051` 仅适合客户端所在环境中已有服务或有效隧道的情况。
当前平台 SSH 入口已返回 `port forwarding is disabled`；本模块不会解除该限制。
从 CPU 调用仍需管理员提供可达的 TCP 入口，填入 `target`。
Windows SSH 建立的本地隧道应先配合 Windows 客户端测试，不要把 WSL 的回环地址视为同一个地址。

持续运行 60 秒可以设置 `count=300`、`fps=5`、`rpc_timeout_seconds=90`。
这是重复发送固定图片的低负载联调，并不读取 RTSP。

输出为逐帧 JSON 和最后一份统计：发送/接收帧数、各结果码、完成 FPS、
客户端观测延迟 p50/p95/p99。退出码 `0` 表示指定帧数全部成功，`1` 表示
帧级失败或结果缺失，`2` 表示配置/图片/连接/RPC 错误，`130` 表示手动中断。
protobuf JSON 中 `uint64`（例如 `frame_id`）按字符串输出；默认值字段可能省略。

## 放入 Web 后端使用

将 `detector_client.py` 放入 Web 后端的 Python 入口目录，在后端虚拟环境中
执行 `python -m pip install -e /你的路径/shared`，业务代码直接导入。
不要把 `detector_contract` 再复制到客户端目录。共享包会安装 `grpcio` 和
`protobuf`；OpenCV 和 numpy 用于示例的图片编码。

```python
import os
import time

from detector_client import DetectorClient, frame_from_jpeg, pb

# 来自 RTSP 解码并编码好的 JPEG；width/height 必须与 JPEG 的真实尺寸一致。
captured_ms = time.time_ns() // 1_000_000  # 实际项目在采集/解码时记录
frame = frame_from_jpeg(
    jpeg_bytes,
    width=width,
    height=height,
    stream_id="camera-1",
    frame_id=1,
    observed_at_unix_ms=captured_ms,
    source_pts=source_pts,
    time_base_num=time_base_num,
    time_base_den=time_base_den,
)

with DetectorClient(
    os.environ["DETECTOR_TARGET"],
    token=os.environ["DETECTOR_AUTH_TOKEN"],
) as client:
    for result in client.detect([frame], rpc_timeout_seconds=10):
        if result.code == pb.RESULT_CODE_OK:
            for box in result.detections:
                print(box.label, box.confidence, box.x1, box.y1, box.x2, box.y2)
        else:
            print(result.frame_id, result.code, result.error_message)
```

这里的 `jpeg_bytes`、宽高及 PTS 来自你的解码器，不是预定义变量。
没有 PTS 时省略 `source_pts` 和两个 `time_base` 参数。
`detect()` 可传入持续产出请求的迭代器，不要每帧重连；持续调用可以不传
`rpc_timeout_seconds`，并在业务层负责停止和断线重连。离开 `with` 会关闭连接。
这是同步客户端，接入异步 Web 框架时在后台线程运行，避免阻塞事件循环。

真实视频每路只保留最新待发送帧，检测关闭时停止产出请求，视频播放可继续。
排队和网络积压不能靠重新填写时间戳掩盖。采集时间由 CPU 记录，CPU/GPU
需同步时钟。GPU 返回的 `source_pts` 和 time base 用于关联视频时间线；
`stream_id + frame_id` 用于关联请求和结果。

## 共享接口约定和更新

RPC：`detector.v1.Detector/Detect`，双向流，一条 RPC 内按请求顺序返回。

- 请求：JPEG（默认不超过 4 MiB）、真实宽高、非空 `stream_id`、大于 0 的 `frame_id`。
- `observed_at_unix_ms` 为 Unix 毫秒；`0` 表示未提供，GPU 不检查到达帧龄。
- `time_base_num` / `time_base_den` 要么同时为 0，要么同时为正数。
- 检测框为原图 `[0,1]` 归一化 `xyxy`；画框时乘以实际显示的图像宽高，另处理留白偏移。
- `OK` 且空 detections 表示没有目标。`EXPIRED` 表示帧龄或排队超时，
  `OVERLOADED` 表示队列满，`INVALID_FRAME` 表示请求错误，`INFERENCE_ERROR` 表示推理失败。
- 帧级错误不会关闭流；鉴权失败通过 gRPC `UNAUTHENTICATED` 抛出。
- `DetectorClient.detect()` 原样返回 protobuf 消息；未知新增结果码也应按非成功处理。

接口唯一维护源是 `shared/detector_contract/detector.proto`。
GPU 和 CPU 都通过 Python 包引用这里的消息、服务和公共通信配置定义。
`requirements.txt` 中的 `-e ../shared` 会直接引用这份源码，不生成第二份副本。
公共配置类见 [`shared/detector_contract/config.py`](../shared/detector_contract/config.py)。

接口变更时，在 GPU 项目根目录执行：

```bash
source scripts/corex_env.sh
# 编辑 shared/detector_contract/detector.proto 后执行：
scripts/generate_detector_proto.sh
```

生成脚本仅更新 `shared/` 中的绑定。跨机器部署时更新相同版本的 `shared/`
目录；不要手动修改 `*_pb2*.py`，也不要分别维护两套 `.proto`。
CPU 只消费已生成源码时不需要安装 `grpcio-tools`。
GPU 设备/模型配置和 CPU 连接/图片配置属于各自运行环境，不共用一份配置值文件。

## 独立测试

在本目录执行（Windows 将解释器路径替换为 `.\.venv\Scripts\python.exe`）：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

测试使用本机临时 gRPC 服务验证消息、鉴权和错误结果，不加载 YOLO 或 GPU。
实际 GPU 联调以运行 `demo.py` 收到 `RESULT_CODE_OK` 为准。
