"""可直接复制到 CPU 后端的客户端；只依赖 grpcio 和共享 protobuf。"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator

import grpc

from detector_contract import detector_pb2 as pb
from detector_contract import detector_pb2_grpc
from detector_contract.config import DEFAULT_TRANSPORT, TransportConfig


def frame_from_jpeg(
    jpeg: bytes,
    *,
    width: int,
    height: int,
    stream_id: str,
    frame_id: int,
    observed_at_unix_ms: int | None = None,
    source_pts: int = 0,
    time_base_num: int = 0,
    time_base_den: int = 0,
) -> pb.DetectionFrame:
    """组装请求；宽高必须与 JPEG 一致，真实视频应传入采集时间和 PTS。"""
    return pb.DetectionFrame(
        stream_id=stream_id,
        frame_id=frame_id,
        observed_at_unix_ms=(
            time.time_ns() // 1_000_000
            if observed_at_unix_ms is None else observed_at_unix_ms
        ),
        width=width,
        height=height,
        jpeg=jpeg,
        source_pts=source_pts,
        time_base_num=time_base_num,
        time_base_den=time_base_den,
    )


class DetectorClient:
    """用 with 管理连接，detect() 接收请求迭代器并逐帧返回 protobuf 结果。

    连接错误以 grpc 异常抛出；EXPIRED 等帧级结果原样返回，由业务处理。
    此客户端用于可信私网或已获准的 SSH 隧道，传输本身不启用 TLS。
    """

    def __init__(
        self, target: str, *, token: str = "", connect_timeout_seconds: float = 5.0,
        transport: TransportConfig = DEFAULT_TRANSPORT,
    ):
        self._target = target
        self._metadata = (("authorization", f"Bearer {token}"),) if token else None
        self._connect_timeout = connect_timeout_seconds
        self._transport = transport
        self._channel: grpc.Channel | None = None

    def __enter__(self) -> DetectorClient:
        if self._channel is not None:
            raise RuntimeError("client is already connected")
        channel = grpc.insecure_channel(self._target, options=self._transport.grpc_options())
        try:
            grpc.channel_ready_future(channel).result(timeout=self._connect_timeout)
        except BaseException:
            channel.close()
            raise
        self._channel = channel
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None

    def detect(
        self,
        frames: Iterable[pb.DetectionFrame],
        *,
        rpc_timeout_seconds: float | None = None,
    ) -> Iterator[pb.DetectionResult]:
        """时限作用于整条 RPC；None 表示无整体时限，并非每帧超时。"""
        if self._channel is None:
            raise RuntimeError("use DetectorClient inside a with block")
        call = detector_pb2_grpc.DetectorStub(self._channel).Detect(
            iter(frames), metadata=self._metadata, timeout=rpc_timeout_seconds,
        )
        try:
            yield from call
        finally:
            call.cancel()
