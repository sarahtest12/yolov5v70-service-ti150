#!/usr/bin/env python3
"""Check that the project virtual environment still belongs to this checkout."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "bi150_gpu_grpc_detector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=PROJECT_ROOT / ".venv")
    parser.add_argument("--quiet", action="store_true", help="suppress diagnostics; use only the exit status")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    venv_dir = args.venv.resolve()
    expected_python = venv_dir / "bin/python"

    if not expected_python.is_file():
        failures.append(f"virtual environment Python is missing: {expected_python}")
    else:
        completed = subprocess.run(
            [str(expected_python), "-c", "import sys; print(sys.prefix)"],
            check=False,
            capture_output=True,
            text=True,
        )
        actual_prefix = completed.stdout.strip()
        if completed.returncode or Path(actual_prefix).resolve() != venv_dir:
            failures.append(f"venv sys.prefix is {actual_prefix or '<unavailable>'}, expected {venv_dir}")

    activate = venv_dir / "bin/activate"
    expected_assignment = f"VIRTUAL_ENV={venv_dir}"
    if not activate.is_file():
        failures.append(f"activation script is missing: {activate}")
    elif expected_assignment not in activate.read_text():
        failures.append(f"activation script does not target {venv_dir}")

    for command_name in ("pip", "gpu-detector", "gpu-detector-client"):
        command = venv_dir / "bin" / command_name
        if not command.is_file():
            if command_name == "pip":
                failures.append(f"venv command is missing: {command}")
            continue
        first_line = command.read_text(errors="replace").splitlines()[0]
        launcher_python = Path(first_line[2:] if first_line.startswith("#!") else first_line)
        if not first_line.startswith("#!") or launcher_python.parent != venv_dir / "bin":
            failures.append(f"{command_name} launcher points outside the current venv: {first_line}")

    site_packages = venv_dir / "lib/python3.10/site-packages"
    for editable_path in site_packages.glob(f"__editable__.{PACKAGE_NAME}-*.pth"):
        for configured_path in editable_path.read_text().splitlines():
            if configured_path and Path(configured_path).resolve() != (PROJECT_ROOT / "src").resolve():
                failures.append(f"editable install points outside this project: {configured_path}")

    for direct_url in site_packages.glob(f"{PACKAGE_NAME}-*.dist-info/direct_url.json"):
        try:
            installed_url = json.loads(direct_url.read_text()).get("url", "")
        except (json.JSONDecodeError, OSError) as error:
            failures.append(f"cannot read editable install metadata {direct_url}: {error}")
            continue
        expected_url = PROJECT_ROOT.as_uri()
        if installed_url and installed_url != expected_url:
            failures.append(f"editable install URL is {installed_url}, expected {expected_url}")

    if failures:
        if not args.quiet:
            for failure in failures:
                print(f"FAIL: {failure}")
        return 1

    if not args.quiet:
        print(f"Virtual environment location check: PASS ({venv_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
