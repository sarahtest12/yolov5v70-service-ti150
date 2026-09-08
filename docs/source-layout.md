# 源码精简范围

本项目交付的是 BI-V150/CoreX 上的 YOLOv5 检测 gRPC 服务。
`src/models/` 和 `src/utils/` 是随服务部署的上游推理依赖，不是第二套服务。
共享接口只保存在 `shared/detector_contract/`。

## 2026-09-08 清理

本次删除 38 个不被保留源码导入的上游附带文件，共 5,464 行、222,607 字节：

| 删除位置 | 原用途 |
|---|---|
| `src/models/tf.py` | TensorFlow 模型转换实现 |
| `src/utils/loggers/` | W&B、ClearML、Comet 训练日志和调参集成 |
| `src/utils/aws/` | 原 YOLOv5 的 AWS 部署和训练恢复脚本 |
| `src/utils/docker/` | 原 YOLOv5 的通用 Docker 镜像示例 |
| `src/utils/flask_rest_api/` | 原 YOLOv5 Flask 推理示例 |
| `src/utils/google_app_engine/` | 原 YOLOv5 Google App Engine 部署示例 |
| `src/utils/autobatch.py`、`callbacks.py`、`loss.py` | 训练批量估算、回调和损失计算 |
| `src/utils/segment/{augmentations,dataloaders,loss,metrics,plots}.py` | 分割训练和评估辅助实现 |

删除前检查了保留源码的普通导入、函数内导入和相对导入，未发现对这些
候选文件的依赖；并用本项目的 `models/yolov5s.pt` 在真实 BI-V150 上推理，
核对了实际加载的模块。修改没有改动模型权重或检测算法。

本项目不提供模型训练、TensorFlow 转换或这些上游部署示例。需要这些能力
时应在原 YOLOv5 测试/训练项目中开发，而不是重新混入客户 gRPC 服务。

## 仍需保留的文件

推理导入关系中有容易误删的间接依赖：

```text
gpu_detector.detector
├── models.common → utils.plots → utils.segment.general
├── models.experimental → models.yolo → utils.autoanchor
└── utils.dataloaders → utils.augmentations / utils.general / utils.torch_utils
```

- `models/common.py`、`yolo.py`、`experimental.py`：模型加载、层定义和执行。
  PyTorch 检查点还会按类的模块路径动态加载对象，不能仅按静态搜索结果删除类。
- `utils/segment/general.py` 及其 `__init__.py`：虽然服务只做目标检测，仍被上游绘图模块导入。
- `utils/plots.py`、`metrics.py`、`autoanchor.py`、`dataloaders.py` 等：当前导入链依赖。
- `utils/activations.py`：保留可信 YOLOv5 检查点中的历史激活类兼容性。
- `utils/triton.py`：仍被保留的 `DetectMultiBackend` 可选分支引用。
- `src/models/` 的模型 YAML：保留上游模型结构配置；没有据单次推理记录删除这些配置。
- `matplotlib`、`seaborn` 等依赖：仍被上游模块导入，本次不从依赖清单移除。
- 测试、模型、类别文件和许可证：分别用于验证、推理和交付，不属于冗余代码。

`src/gpu_detector/client.py`、`continuous_client.py` 与 `cpu_client/` 存在功能交叉，
但前两者仍有公开命令入口、运维文档和测试引用；其中持续客户端还支持多路逻辑流。
本次保留现有 CLI 兼容性。跨服务器开发优先使用 `cpu_client/`。
若后续统一客户端，需要先迁移命令参数和调用方，不能只删除文件。

## 环境和产物

`.venv.stale-*` 是环境迁移时保留的旧虚拟环境，`dist/` 是历史交付包，
`__pycache__/` 是解释器缓存，均不是业务源码。本次只清理了已删除源码的
对应缓存，没有清理环境备份和历史交付包，也没有修改当前 `.venv` 的包安装、
用户的 `run.sh` 或运行中的服务。
已有 `dist/` 压缩包不会自动更新；需要新交付包时重新运行打包脚本。

## 验证与恢复

```bash
source scripts/corex_env.sh
python -m unittest discover -s tests/gpu_detector -v
RUN_GPU_INTEGRATION=1 python -m unittest tests.gpu_detector.test_gpu_integration -v
cd cpu_client
python -m unittest discover -s tests -v
```

删除的文件均已保存在本独立项目的 Git 提交 `a651493` 中。若需恢复，
先确认目标路径没有新修改，再使用 `git restore --source=a651493 -- <具体文件路径>`。
恢复单个文件时仍需核对它所依赖的其他已移除文件。
