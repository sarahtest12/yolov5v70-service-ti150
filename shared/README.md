# CPU/GPU 共享模块

本目录是 `detector_contract` 的唯一源码位置。GPU 服务和 CPU 客户端都安装
这个普通 Python 包，不再各自保存协议副本，也不再运行同步脚本。

```text
shared/
├── pyproject.toml             包定义及 gRPC/protobuf 依赖版本
└── detector_contract/
    ├── __init__.py
    ├── config.py              TransportConfig、公共默认值和鉴权环境变量名
    ├── detector.proto         消息、结果码和 RPC 的唯一维护源
    ├── detector_pb2.py         生成的消息类
    └── detector_pb2_grpc.py    生成的客户端/服务端绑定
```

在包含 `shared/` 的目录执行：

```bash
python -m pip install -e ./shared
```

`-e` 表示直接引用本地源码；修改源码后，新启动的 Python 进程就会使用修改。
两端的导入语句相同：

```python
from detector_contract import detector_pb2, detector_pb2_grpc
from detector_contract.config import TransportConfig, DEFAULT_TRANSPORT
```

`TransportConfig` 集中定义端口、帧大小和 gRPC 消息大小的默认值，以及
双方使用的 gRPC 配置选项。GPU 可通过自己的环境变量覆盖这些默认值；
CPU 可把自定义 `TransportConfig` 传给 `DetectorClient(transport=...)`。
修改一端的实际配置不会自动修改另一端。

CPU 连接地址、测试图片、帧率仍由 `cpu_client/config.json` 管理；GPU 的
设备、权重路径、队列和监听地址仍由 `DetectorConfig` / 部署环境变量管理。
公共配置类型只有一份，各进程运行时的配置值独立，密钥不写入共享源码。

协议修改后，在 GPU 项目根目录执行：

```bash
source scripts/corex_env.sh
scripts/generate_detector_proto.sh
```

脚本只在本目录生成绑定。也可在安装了 `./shared[codegen]` 的普通 Python
环境中从项目根目录生成：

```bash
python -m grpc_tools.protoc -Ishared --python_out=shared --grpc_python_out=shared shared/detector_contract/detector.proto
```

两台服务器各自安装同一版本的 `shared/` 源码，这是共享同一份定义，
不是跨机器访问同一个磁盘文件。现在可以复制源码目录；之后也可将本包
发布到内部 Python 包仓库，通过版本号管理。协议变更时需按兼容性要求
更新两端部署，不能假定本地编辑会自动传到远端。
