#!/usr/bin/env python3
"""Validate the BI-V150/CoreX Python environment used by the GPU service."""

from __future__ import annotations

import argparse
import os
import platform
import re
import sys
import sysconfig
from importlib import metadata
from pathlib import Path
from typing import Any


EXPECTED_COREX_PACKAGES = {
    "numpy": "1.26.4",
    "torch": "2.7.1+corex.4.4.0",
    "torchaudio": "2.7.1+corex.4.4.0",
    "torchvision": "0.22.1+corex.4.4.0",
    "triton": "3.1.0+corex.4.4.0",
}

EXPECTED_VENV_PACKAGES = {
    "grpcio": "1.83.1",
    "grpcio-health-checking": "1.83.1",
    "grpcio-tools": "1.83.1",
    "matplotlib": "3.10.9",
    "opencv-python-headless": "4.11.0.86",
    "protobuf": "7.36.1",
    "seaborn": "0.13.2",
}

FORBIDDEN_VENV_PACKAGES = {
    "cuda-bindings",
    "cuda-pathfinder",
    "cuda-toolkit",
    "cupy",
    "cupy-cuda11x",
    "cupy-cuda12x",
    "torch",
    "torchaudio",
    "torchvision",
    "triton",
}


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def distributions_at(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in metadata.distributions(path=[str(path)]):
        name = distribution.metadata.get("Name")
        if name:
            packages[normalize_package_name(name)] = distribution.version
    return packages


def is_below(path: str | os.PathLike[str], parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="GPU index used for the tensor and NMS checks")
    parser.add_argument("--skip-gpu", action="store_true", help="only validate packages and import paths")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    corex_home = Path(os.environ.get("COREX_HOME", "/usr/local/corex"))
    corex_python = corex_home / "lib64/python3/dist-packages"
    service_root = Path(__file__).resolve().parents[1]
    service_src = service_root / "src"
    expected_venv = service_root / ".venv"
    venv_site = Path(sysconfig.get_paths()["purelib"])

    require(sys.version_info[:2] == (3, 10), f"Python 3.10 is required, found {platform.python_version()}")
    require(platform.machine() == "x86_64", f"x86_64 is required, found {platform.machine()}")
    require(Path(sys.prefix).resolve() == expected_venv.resolve(),
            f"Python must run from {expected_venv}, found sys.prefix={sys.prefix}")
    require(corex_python.is_dir(), f"CoreX Python package directory is missing: {corex_python}")
    require(os.environ.get("YOLOv5_AUTOINSTALL", "").lower() == "false",
            "YOLOv5_AUTOINSTALL must be false")
    require(str(corex_python) in os.environ.get("PYTHONPATH", "").split(":"),
            f"PYTHONPATH must contain {corex_python}")
    require(str(service_src) in os.environ.get("PYTHONPATH", "").split(":"),
            f"PYTHONPATH must contain {service_src}")
    require((service_root / "models/yolov5s.pt").is_file(),
            f"model weights are missing: {service_root / 'models/yolov5s.pt'}")
    require((service_root / "config/coco80.yaml").is_file(),
            f"class metadata is missing: {service_root / 'config/coco80.yaml'}")

    local_packages = distributions_at(venv_site)
    for name, expected_version in EXPECTED_VENV_PACKAGES.items():
        actual_version = local_packages.get(name)
        require(actual_version == expected_version,
                f"venv package {name} must be {expected_version}, found {actual_version or 'missing'}")

    forbidden = sorted(
        name for name in local_packages
        if name in FORBIDDEN_VENV_PACKAGES or name.startswith("nvidia-")
    )
    require(not forbidden, f"venv contains forbidden accelerator packages: {', '.join(forbidden)}")

    for name, expected_version in EXPECTED_COREX_PACKAGES.items():
        try:
            actual_version = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual_version = None
        require(actual_version == expected_version,
                f"CoreX package {name} must be {expected_version}, found {actual_version or 'missing'}")

    modules: dict[str, Any] = {}
    for name in (
        "cv2",
        "detector_contract",
        "gpu_detector",
        "grpc",
        "matplotlib",
        "models",
        "numpy",
        "prometheus_client",
        "seaborn",
        "torch",
        "torchvision",
    ):
        try:
            modules[name] = __import__(name)
        except Exception as error:  # report every failed import in one run
            failures.append(f"cannot import {name}: {error!r}")

    for name in ("numpy", "torch", "torchvision"):
        module = modules.get(name)
        if module is not None:
            require(is_below(module.__file__, corex_python),
                    f"{name} must load from CoreX, found {module.__file__}")

    for name in ("cv2", "grpc", "matplotlib", "seaborn"):
        module = modules.get(name)
        if module is not None:
            require(is_below(module.__file__, venv_site),
                    f"{name} must load from the active venv, found {module.__file__}")

    for name in ("detector_contract", "gpu_detector", "models"):
        module = modules.get(name)
        if module is not None:
            require(is_below(module.__file__, service_src),
                    f"{name} must load from this service project, found {module.__file__}")

    torch = modules.get("torch")
    torchvision = modules.get("torchvision")
    if not args.skip_gpu and torch is not None and torchvision is not None:
        try:
            require(torch.cuda.is_available(), "torch.cuda.is_available() is false")
            device_count = torch.cuda.device_count()
            require(device_count > args.device,
                    f"GPU index {args.device} is unavailable; detected {device_count} device(s)")
            if device_count > args.device:
                device = torch.device(f"cuda:{args.device}")
                matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
                require((matrix @ matrix).cpu().tolist() == [[7.0, 10.0], [15.0, 22.0]],
                        "CoreX GPU matrix multiplication returned an unexpected result")

                boxes = torch.tensor(
                    [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0], [20.0, 20.0, 30.0, 30.0]],
                    device=device,
                )
                scores = torch.tensor([0.9, 0.8, 0.7], device=device)
                kept = torchvision.ops.nms(boxes, scores, 0.5).cpu().tolist()
                require(kept == [0, 2], f"CoreX torchvision NMS returned {kept}, expected [0, 2]")
                torch.cuda.synchronize(device)
        except Exception as error:
            failures.append(f"GPU operation failed: {error!r}")

    print(f"python={platform.python_version()} executable={sys.executable}")
    print(f"service_root={service_root}")
    print(f"corex_home={corex_home.resolve()}")
    if torch is not None:
        print(f"torch={metadata.version('torch')} path={torch.__file__}")
        print(f"cuda_compatible={torch.version.cuda} devices={torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"gpu[{index}]={torch.cuda.get_device_name(index)}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    print("CoreX environment check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
