"""CPU/GPU 共用的通信配置定义；不读取环境变量，也不依赖 GPU SDK。"""

from dataclasses import dataclass


AUTH_TOKEN_ENV = "DETECTOR_AUTH_TOKEN"


@dataclass(frozen=True, slots=True)
class TransportConfig:
    grpc_port: int = 50051
    max_frame_bytes: int = 4 * 1024 * 1024
    max_message_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        if type(self.grpc_port) is not int or not 0 <= self.grpc_port <= 65535:
            raise ValueError("grpc_port must be an integer between 0 and 65535")
        if type(self.max_frame_bytes) is not int or self.max_frame_bytes <= 0:
            raise ValueError("max_frame_bytes must be a positive integer")
        if type(self.max_message_bytes) is not int or self.max_message_bytes < self.max_frame_bytes:
            raise ValueError("max_message_bytes must be an integer at least max_frame_bytes")

    def grpc_options(self) -> tuple[tuple[str, int], ...]:
        return (
            ("grpc.max_send_message_length", self.max_message_bytes),
            ("grpc.max_receive_message_length", self.max_message_bytes),
        )


DEFAULT_TRANSPORT = TransportConfig()
