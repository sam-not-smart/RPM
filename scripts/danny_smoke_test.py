"""Read-only environment and numerical smoke test for the Danny workstation.

Run from the repository root with the intended Conda environment activated:

    python scripts/danny_smoke_test.py

The test does not write to the repository or run a sustained benchmark.  A
temporary Matplotlib image is created in the system temporary directory and
removed immediately.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from importlib import import_module
from pathlib import Path


EXPECTED_PACKAGES = ("numpy", "scipy", "matplotlib")


def gibibytes(byte_count: int) -> float:
    return byte_count / 1024**3


def total_memory_bytes() -> int | None:
    """Return physical memory using only the standard library."""
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return int(result.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.total_physical)
        except (AttributeError, OSError):
            return None
    return None


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def print_system_info() -> None:
    print("Danny environment smoke test")
    print("=" * 28)
    print(f"Hostname:          {socket.gethostname()}")
    print(f"Operating system:  {platform.platform()}")
    print(f"Python:            {sys.version.split()[0]} ({sys.executable})")
    print(f"Conda environment: {os.environ.get('CONDA_DEFAULT_ENV', 'not detected')}")
    print(f"Logical CPUs:      {os.cpu_count() or 'unknown'}")

    memory = total_memory_bytes()
    print(f"Physical memory:   {gibibytes(memory):.1f} GiB" if memory else "Physical memory:   unknown")

    disk = shutil.disk_usage(Path.cwd())
    print(f"Disk at repo:      {gibibytes(disk.free):.1f} GiB free / {gibibytes(disk.total):.1f} GiB")

    gpu = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    print(f"NVIDIA GPU:        {gpu if gpu else 'not detected (not required by this project)'}")


def check_packages() -> dict[str, object]:
    print("\nRequired packages")
    print("-" * 28)
    packages: dict[str, object] = {}
    for name in EXPECTED_PACKAGES:
        try:
            module = import_module(name)
        except Exception as exc:  # Include binary/linker failures, not only ImportError.
            print(f"FAIL {name}: {exc}")
            continue
        packages[name] = module
        print(f"PASS {name} {getattr(module, '__version__', 'version unknown')}")
    return packages


def check_numerics(packages: dict[str, object]) -> list[str]:
    failures: list[str] = []
    print("\nNumerical checks")
    print("-" * 28)

    if "numpy" in packages:
        np = packages["numpy"]
        matrix = np.arange(1.0, 10.0).reshape(3, 3)
        observed = float(np.linalg.det(matrix + np.eye(3)))
        expected = -2.0
        if np.isclose(observed, expected):
            print(f"PASS NumPy linear algebra (determinant={observed:.1f})")
        else:
            failures.append(f"NumPy determinant was {observed}, expected {expected}")
            print(f"FAIL {failures[-1]}")

    if "numpy" in packages and "scipy" in packages:
        np = packages["numpy"]
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import expm_multiply

            generator = csr_matrix([[0.0, -1.0], [1.0, 0.0]])
            observed = expm_multiply(generator * (np.pi / 2), np.array([1.0, 0.0]))
            expected = np.array([0.0, 1.0])
            if np.allclose(observed, expected, atol=1e-12):
                print("PASS SciPy sparse evolution (90-degree rotation)")
            else:
                failures.append(f"SciPy sparse evolution returned {observed}")
                print(f"FAIL {failures[-1]}")
        except Exception as exc:
            failures.append(f"SciPy sparse evolution raised {exc}")
            print(f"FAIL {failures[-1]}")

    if "matplotlib" in packages:
        try:
            matplotlib = packages["matplotlib"]
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt

            with tempfile.TemporaryDirectory(prefix="danny-smoke-") as directory:
                output = Path(directory) / "smoke.png"
                figure, axis = plt.subplots(figsize=(2, 2))
                axis.plot([0, 1], [0, 1])
                figure.savefig(output)
                plt.close(figure)
                if not output.is_file() or output.stat().st_size == 0:
                    raise RuntimeError("Matplotlib produced no output")
            print("PASS Matplotlib headless PNG rendering")
        except Exception as exc:
            failures.append(f"Matplotlib rendering raised {exc}")
            print(f"FAIL {failures[-1]}")

    return failures


def main() -> int:
    print_system_info()
    packages = check_packages()
    failures = [f"could not import {name}" for name in EXPECTED_PACKAGES if name not in packages]
    failures.extend(check_numerics(packages))

    print("\nResult")
    print("-" * 28)
    if failures:
        print(f"FAILED ({len(failures)} problem(s))")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PASSED: Python, required packages, and numerical kernels are working.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
