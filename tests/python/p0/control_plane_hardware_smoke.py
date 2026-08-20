#!/usr/bin/env python3
"""Repeatable P0 control-plane smoke test against real accelerator hardware.

The test owns an isolated hi.cpp master process and only terminates PIDs that
were returned by that process.  It validates the device ledger, dynamic Worker
lifecycle, crash convergence, generation recovery, proxy inference, graceful
shutdown, and the INFO/WARN/ERROR logging contract.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


class SmokeFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class HttpResult:
    status: int
    body: Any
    text: str


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(level: str, message: str) -> None:
    print(f"{now_text()} | {level:<5} | {message}", flush=True)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def process_resource_sample(process_id: int) -> dict[str, int]:
    """Return parent-process RSS, private bytes, handles/fds, and threads."""
    if os.name == "nt":
        from ctypes import wintypes

        process_query_information = 0x0400
        process_vm_read = 0x0010
        th32cs_snapthread = 0x00000004

        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("page_fault_count", wintypes.DWORD),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
                ("private_usage", ctypes.c_size_t),
            ]

        class ThreadEntry32(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ThreadID", wintypes.DWORD),
                ("th32OwnerProcessID", wintypes.DWORD),
                ("tpBasePri", wintypes.LONG),
                ("tpDeltaPri", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.GetProcessHandleCount.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCountersEx),
            wintypes.DWORD,
        ]
        process = kernel32.OpenProcess(
            process_query_information | process_vm_read, False, process_id
        )
        require(bool(process), f"cannot open server process {process_id} for sampling")
        try:
            handles = wintypes.DWORD()
            require(
                bool(kernel32.GetProcessHandleCount(process, ctypes.byref(handles))),
                f"GetProcessHandleCount failed for PID {process_id}",
            )
            memory = ProcessMemoryCountersEx()
            memory.cb = ctypes.sizeof(memory)
            require(
                bool(psapi.GetProcessMemoryInfo(process, ctypes.byref(memory), memory.cb)),
                f"GetProcessMemoryInfo failed for PID {process_id}",
            )
        finally:
            kernel32.CloseHandle(process)

        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel32.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry32)]
        kernel32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry32)]
        snapshot = kernel32.CreateToolhelp32Snapshot(th32cs_snapthread, 0)
        invalid_handle = ctypes.c_void_p(-1).value
        require(snapshot not in (None, invalid_handle), "cannot snapshot Windows threads")
        thread_count = 0
        try:
            entry = ThreadEntry32()
            entry.dwSize = ctypes.sizeof(entry)
            found = bool(kernel32.Thread32First(snapshot, ctypes.byref(entry)))
            while found:
                if entry.th32OwnerProcessID == process_id:
                    thread_count += 1
                found = bool(kernel32.Thread32Next(snapshot, ctypes.byref(entry)))
        finally:
            kernel32.CloseHandle(snapshot)
        return {
            "rss_bytes": int(memory.working_set_size),
            "private_bytes": int(memory.private_usage),
            "handles": int(handles.value),
            "threads": thread_count,
        }

    proc_dir = Path("/proc") / str(process_id)
    if proc_dir.is_dir():
        status: dict[str, str] = {}
        for line in (proc_dir / "status").read_text(encoding="utf-8").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                status[key] = value.strip()

        def kibibytes(name: str) -> int:
            raw = status.get(name, "0 kB").split()[0]
            return int(raw) * 1024

        return {
            "rss_bytes": kibibytes("VmRSS"),
            "private_bytes": kibibytes("RssAnon"),
            "handles": len(list((proc_dir / "fd").iterdir())),
            "threads": int(status.get("Threads", "0")),
        }

    result = subprocess.run(
        ["ps", "-o", "rss=,nlwp=", "-p", str(process_id)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    require(len(result) >= 2, f"cannot parse process sample for PID {process_id}")
    return {
        "rss_bytes": int(result[0]) * 1024,
        "private_bytes": 0,
        "handles": 0,
        "threads": int(result[1]),
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    default_executable = (
        repo_root
        / "cmake-build-release-visual-studio"
        / "bin"
        / "Release"
        / "houmo-llm-hi-cpp-win.exe"
    )
    default_config = (
        repo_root / "cmake-build-release-visual-studio" / "bin" / "config"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-executable", type=Path, default=default_executable)
    parser.add_argument("--config-dir", type=Path, default=default_config)
    parser.add_argument("--primary-model", type=Path, required=True)
    parser.add_argument(
        "--primary-mmproj",
        type=Path,
        help="Optional multimodal projector used by the main-process model.",
    )
    parser.add_argument("--worker-model", type=Path)
    parser.add_argument(
        "--worker-mmproj",
        type=Path,
        help="Optional multimodal projector used by the dynamic Worker.",
    )
    parser.add_argument("--primary-device", type=int, default=0)
    parser.add_argument("--worker-device", type=int, default=1)
    parser.add_argument(
        "--allow-shared-device",
        action="store_true",
        help="Allow the primary model and dynamic Worker to share one device.",
    )
    parser.add_argument(
        "--skip-resource-rejection",
        action="store_true",
        help="Skip the deliberate insufficient-capacity request.",
    )
    parser.add_argument("--port", type=int, default=17901)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--load-timeout", type=float, default=240.0)
    parser.add_argument("--inference-timeout", type=float, default=180.0)
    parser.add_argument("--convergence-timeout", type=float, default=20.0)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument(
        "--skip-pre-ready-crash",
        action="store_true",
        help="Skip the timing-sensitive real Worker failure before READY.",
    )
    parser.add_argument(
        "--soak-cycles",
        type=int,
        default=0,
        help="Additional dynamic Worker load/infer/unload cycles after the smoke flow.",
    )
    parser.add_argument(
        "--soak-duration-seconds",
        type=float,
        default=0.0,
        help="Run Worker lifecycle soak until this duration has elapsed; checked between cycles.",
    )
    parser.add_argument(
        "--soak-inference-every",
        type=int,
        default=10,
        help="Run proxy inference every N soak cycles; 0 disables soak inference.",
    )
    parser.add_argument(
        "--check-stream-cancellation",
        action="store_true",
        help=(
            "Verify P1-6 dispatch metrics and actively disconnect embedded and "
            "Worker SSE streams after their first chunk."
        ),
    )
    parser.add_argument(
        "--chat-benchmark-samples",
        type=int,
        default=0,
        help="Measure this many sequential chat requests on each loaded runtime; 0 disables it.",
    )
    parser.add_argument(
        "--chat-benchmark-warmup",
        type=int,
        default=5,
        help="Warm-up requests before collecting chat latency samples.",
    )
    parser.add_argument("--soak-progress-every", type=int, default=10)
    parser.add_argument("--max-handle-growth", type=int, default=32)
    parser.add_argument("--max-thread-growth", type=int, default=4)
    parser.add_argument("--max-rss-growth-mb", type=int, default=128)
    parser.add_argument("--max-private-growth-mb", type=int, default=128)
    parser.add_argument(
        "--max-tail-growth-mb",
        type=int,
        default=8,
        help="Maximum RSS/private growth during the final 20%% of soak cycles.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=(
            repo_root
            / "artifacts"
            / "p0_hardware_smoke"
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        ),
    )
    return parser.parse_args()


class HardwareSmoke:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.base_url = f"http://127.0.0.1:{args.port}"
        self.server: subprocess.Popen[bytes] | None = None
        self.server_log = None
        self.server_log_path = args.artifact_dir / "server-console.log"
        self.summary_path = args.artifact_dir / "summary.json"
        self.steps: list[dict[str, Any]] = []
        # Only currently owned/live Worker PIDs belong here. Remove a PID as
        # soon as supervisor convergence or graceful shutdown is observed so
        # cleanup can never touch a historical PID that the OS may have reused.
        self.worker_pids: set[int] = set()
        self.loaded_model_ids: list[str] = []
        self.baseline_health: dict[str, Any] = {}
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def record(self, name: str, **details: Any) -> None:
        entry = {"name": name, "status": "passed", **details}
        self.steps.append(entry)
        log("PASS", name)

    def validate_inputs(self) -> None:
        paths = {
            "server executable": self.args.server_executable,
            "config directory": self.args.config_dir,
            "primary model": self.args.primary_model,
        }
        if self.args.primary_mmproj is not None:
            paths["primary mmproj"] = self.args.primary_mmproj
        if self.args.worker_mmproj is not None:
            paths["worker mmproj"] = self.args.worker_mmproj
        if self.args.worker_model is None:
            self.args.worker_model = self.args.primary_model
        paths["worker model"] = self.args.worker_model
        for label, path in paths.items():
            require(path.exists(), f"{label} does not exist: {path}")
        require(self.args.server_executable.is_file(), "server executable is not a file")
        require(self.args.config_dir.is_dir(), "config directory is not a directory")
        require(
            self.args.primary_device != self.args.worker_device
            or self.args.allow_shared_device,
            "primary and Worker devices must be different unless --allow-shared-device is set",
        )
        require(1 <= self.args.port <= 65533, "port must leave room for Worker ports")
        require(self.args.soak_cycles >= 0, "soak cycles cannot be negative")
        require(
            self.args.soak_duration_seconds >= 0,
            "soak duration cannot be negative",
        )
        require(self.args.soak_inference_every >= 0, "soak inference interval cannot be negative")
        require(self.args.soak_progress_every > 0, "soak progress interval must be positive")
        require(self.args.chat_benchmark_samples >= 0, "benchmark samples cannot be negative")
        require(self.args.chat_benchmark_warmup >= 0, "benchmark warm-up cannot be negative")
        require(not self.port_open(self.args.port), f"port {self.args.port} is already in use")

    @staticmethod
    def port_open(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    def start_server(self) -> None:
        self.args.artifact_dir.mkdir(parents=True, exist_ok=True)
        (self.args.artifact_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.server_log = self.server_log_path.open("wb")
        env = os.environ.copy()
        env.update(
            {
                "HLIE__SERVER__PORT": str(self.args.port),
                "HLIE__SERVER__DEVICES": ",".join(
                    dict.fromkeys(
                        (str(self.args.primary_device), str(self.args.worker_device))
                    )
                ),
                "HLIE__MODEL__CUR_MODEL": str(self.args.primary_model.parent),
                "HLIE__MODEL__DEFAULT": "",
                "HLIE__MODEL__DEFAULT_MMPROJ": "",
                "HLIE__MODEL__AUTO_LOAD": "false",
                "HLIE__MODEL__AUTO_UNLOAD": "false",
                "HLIE__MODEL__DISCOVERY_SCAN_INTERVAL_SEC": "0",
                "HLIE__REPLICA__COUNT": "1",
                "HLIE__LOG__LEVEL": "info",
                "HLIE__LOG__LOCATION": str(self.args.artifact_dir / "logs"),
            }
        )
        command = [
            str(self.args.server_executable.resolve()),
            "--config",
            str(self.args.config_dir.resolve()),
            "--port",
            str(self.args.port),
        ]
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        log("INFO", f"Starting isolated server on {self.base_url}")
        self.server = subprocess.Popen(
            command,
            cwd=self.args.server_executable.resolve().parent,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=self.server_log,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        self.wait_until(
            "server /live",
            self.args.startup_timeout,
            lambda: self.request("GET", "/live", timeout=2.0).status == 200,
        )
        require(self.server.poll() is None, "server exited during startup")
        self.record("isolated server became live", pid=self.server.pid, port=self.args.port)

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> HttpResult:
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with self.opener.open(request, timeout=timeout) as response:
                raw = response.read()
                status = response.status
        except urllib.error.HTTPError as error:
            raw = error.read()
            status = error.code
        text = raw.decode("utf-8", errors="replace")
        parsed: Any = None
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
        return HttpResult(status=status, body=parsed, text=text)

    def request_ok(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> Any:
        result = self.request(method, path, body, timeout)
        require(
            200 <= result.status < 300,
            f"{method} {path} failed with HTTP {result.status}: {result.text[:2000]}",
        )
        require(isinstance(result.body, dict), f"{method} {path} returned non-object JSON")
        return result.body

    def wait_until(
        self,
        description: str,
        timeout: float,
        predicate: Callable[[], Any],
    ) -> Any:
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.server is not None and self.server.poll() is not None:
                raise SmokeFailure(
                    f"server exited with code {self.server.returncode} while waiting for {description}"
                )
            try:
                value = predicate()
                if value:
                    return value
            except (OSError, urllib.error.URLError, TimeoutError, SmokeFailure) as error:
                last_error = error
            time.sleep(self.args.poll_interval)
        suffix = f": {last_error}" if last_error else ""
        raise SmokeFailure(f"timed out waiting for {description}{suffix}")

    def health(self) -> dict[str, Any]:
        body = self.request_ok("GET", "/health")
        self.check_ledger_invariants(body)
        return body

    @staticmethod
    def prometheus_value(metrics_text: str, series: str) -> int:
        prefix = f"{series} "
        for line in metrics_text.splitlines():
            if not line.startswith(prefix):
                continue
            raw = line[len(prefix) :].strip()
            try:
                value = float(raw)
            except ValueError as error:
                raise SmokeFailure(
                    f"Prometheus series {series!r} has invalid value {raw!r}"
                ) from error
            require(value >= 0 and value.is_integer(), f"invalid metric {line!r}")
            return int(value)
        raise SmokeFailure(f"Prometheus series {series!r} is missing")

    def dispatch_metrics(self) -> dict[str, int]:
        health = self.health()
        raw_dispatch = health.get("inference_dispatch")
        require(
            isinstance(raw_dispatch, dict),
            "/health.inference_dispatch is missing or is not an object",
        )
        series_by_field = {
            "active_sessions": "hlie_inference_dispatch_active_sessions",
            "dispatched": "hlie_inference_dispatch_total",
            "draining_rejected": (
                'hlie_inference_dispatch_rejected_total{reason="draining"}'
            ),
            "model_not_found": (
                'hlie_inference_dispatch_rejected_total{reason="model_not_found"}'
            ),
            "model_not_ready": (
                'hlie_inference_dispatch_rejected_total{reason="model_not_ready"}'
            ),
            "runtime_unavailable": (
                'hlie_inference_dispatch_rejected_total{reason="runtime_unavailable"}'
            ),
            "endpoint_unavailable": (
                'hlie_inference_dispatch_rejected_total{reason="endpoint_unavailable"}'
            ),
            "unsupported_operation": (
                'hlie_inference_dispatch_rejected_total{reason="unsupported_operation"}'
            ),
            "endpoint_failures": "hlie_inference_dispatch_failures_total",
            "cached_endpoints": "hlie_inference_dispatch_cached_endpoints",
        }
        dispatch: dict[str, int] = {}
        for field in series_by_field:
            value = raw_dispatch.get(field)
            require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"invalid /health.inference_dispatch.{field}: {value!r}",
            )
            dispatch[field] = value

        metrics = self.request("GET", "/metrics")
        require(
            metrics.status == 200,
            f"GET /metrics failed with HTTP {metrics.status}: {metrics.text[:2000]}",
        )
        for field, series in series_by_field.items():
            exported = self.prometheus_value(metrics.text, series)
            require(
                exported == dispatch[field],
                f"dispatch metric mismatch for {field}: health={dispatch[field]}, "
                f"prometheus={exported}",
            )
        return dispatch

    @staticmethod
    def device(health: dict[str, Any], device_id: int) -> dict[str, Any]:
        devices = health.get("devices")
        require(isinstance(devices, list), "health.devices is not a list")
        for device in devices:
            if isinstance(device, dict) and device.get("id") == device_id:
                return device
        raise SmokeFailure(f"device {device_id} is missing from /health")

    @staticmethod
    def ledger_state(device: dict[str, Any]) -> tuple[int, int]:
        return int(device["pending_reserved_mb"]), int(device["committed_reserved_mb"])

    @staticmethod
    def check_ledger_invariants(health: dict[str, Any]) -> None:
        for device in health.get("devices", []):
            fields = (
                "total_mb",
                "observed_used_mb",
                "pending_reserved_mb",
                "committed_reserved_mb",
                "committed_unobserved_mb",
                "effective_used_mb",
                "admissible_mb",
            )
            for field in fields:
                require(
                    isinstance(device.get(field), int) and device[field] >= 0,
                    f"invalid ledger field {field}: {device}",
                )
            expected_effective = (
                device["observed_used_mb"]
                + device["pending_reserved_mb"]
                + device["committed_unobserved_mb"]
            )
            require(
                device["effective_used_mb"] == expected_effective,
                f"effective_used invariant failed: {device}",
            )
            require(
                device["committed_unobserved_mb"] <= device["committed_reserved_mb"],
                f"committed_unobserved exceeds committed: {device}",
            )
            require(
                device["admissible_mb"] <= device["total_mb"],
                f"admissible exceeds capacity: {device}",
            )

    @staticmethod
    def load_body(model: Path, mmproj: Path | None, device: int) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model_type": "llm",
            "model_path": str(model.resolve()),
            "devices": str(device),
        }
        if mmproj is not None:
            body["params"] = {"mm_proj_path": str(mmproj.resolve())}
        return body

    def load_model(self, model: Path, mmproj: Path | None, device: int) -> dict[str, Any]:
        body = self.request_ok(
            "POST",
            "/models/load",
            self.load_body(model, mmproj, device),
            timeout=self.args.load_timeout,
        )
        require(body.get("code") == 0, f"model load returned failure: {body}")
        return body

    def models(self) -> list[dict[str, Any]]:
        body = self.request_ok("GET", "/v1/models")
        data = body.get("data")
        require(isinstance(data, list), "/v1/models data is not a list")
        return [item for item in data if isinstance(item, dict)]

    def loaded_model_on_port(self, port: int, process_id: int | None = None) -> str | None:
        for model in self.models():
            status = model.get("status", {})
            if not isinstance(status, dict) or status.get("value") != "loaded":
                continue
            for instance in model.get("instances", []):
                if not isinstance(instance, dict) or instance.get("port") != port:
                    continue
                if process_id is not None and instance.get("process_id") != process_id:
                    continue
                model_id = model.get("id")
                if isinstance(model_id, str) and model_id:
                    return model_id
        return None

    def status_processor(self, process_id: int) -> dict[str, Any] | None:
        body = self.request_ok("GET", "/hlie/v1/status")
        processors = body.get("processor")
        require(isinstance(processors, list), "status.processor is not a list")
        for processor in processors:
            if isinstance(processor, dict) and processor.get("process_id") == process_id:
                return processor
        return None

    def wait_loaded(self, load: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        port = int(load["port"])
        process_id = int(load.get("process_id", 0))
        model_id = self.wait_until(
            f"model READY on port {port}",
            self.args.load_timeout,
            lambda: self.loaded_model_on_port(port, process_id if process_id else None),
        )
        processor = self.status_processor(process_id) if process_id else None
        return str(model_id), processor

    @staticmethod
    def launch_manifest(load: dict[str, Any]) -> Path | None:
        raw = load.get("json_info_file")
        return Path(raw) if isinstance(raw, str) and raw else None

    def wait_manifest_removed(self, load: dict[str, Any]) -> None:
        manifest = self.launch_manifest(load)
        if manifest is None:
            return
        self.wait_until(
            f"one-time launch manifest cleanup ({manifest.name})",
            self.args.convergence_timeout,
            lambda: not manifest.exists(),
        )

    def chat(self, model_id: str) -> float:
        started = time.perf_counter()
        response = self.request_ok(
            "POST",
            "/v1/chat/completions",
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "Reply with P0_OK only."}],
                "temperature": 0,
                "max_tokens": 16,
                "stream": False,
            },
            timeout=self.args.inference_timeout,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        choices = response.get("choices")
        require(isinstance(choices, list) and choices, f"chat response has no choices: {response}")
        return elapsed_ms

    def benchmark_chat(self, model_id: str, runtime_kind: str) -> None:
        sample_count = self.args.chat_benchmark_samples
        if sample_count <= 0:
            return
        require(
            self.args.chat_benchmark_warmup >= 0,
            "--chat-benchmark-warmup cannot be negative",
        )
        for _ in range(self.args.chat_benchmark_warmup):
            self.chat(model_id)

        latencies_ms = [self.chat(model_id) for _ in range(sample_count)]
        ordered = sorted(latencies_ms)

        def percentile(quantile: float) -> float:
            index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
            return ordered[index]

        self.record(
            f"{runtime_kind} sequential chat latency benchmark",
            model=model_id,
            samples=sample_count,
            warmup=self.args.chat_benchmark_warmup,
            mean_ms=round(sum(ordered) / len(ordered), 3),
            p50_ms=round(percentile(0.50), 3),
            p99_ms=round(percentile(0.99), 3),
            min_ms=round(ordered[0], 3),
            max_ms=round(ordered[-1], 3),
            latencies_ms=[round(value, 3) for value in latencies_ms],
        )

    def check_stream_cancellation(self, model_id: str, runtime_kind: str) -> None:
        before = self.dispatch_metrics()
        require(
            before["active_sessions"] == 0,
            f"{runtime_kind} stream started with active dispatch sessions: {before}",
        )
        request_body = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write a long numbered list from 1 through 512, one item at a "
                        "time. Do not stop early."
                    ),
                }
            ],
            "temperature": 0,
            "max_tokens": 512,
            "stream": True,
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
                "Connection": "close",
            },
            method="POST",
        )

        response = None
        first_event = b""
        try:
            response = self.opener.open(request, timeout=self.args.inference_timeout)
            require(response.status == 200, f"{runtime_kind} stream returned HTTP {response.status}")
            content_type = response.headers.get("Content-Type", "")
            require(
                "text/event-stream" in content_type,
                f"{runtime_kind} stream returned unexpected Content-Type {content_type!r}",
            )

            active = self.health().get("inference_dispatch", {}).get("active_sessions")
            require(
                isinstance(active, int) and active >= before["active_sessions"] + 1,
                f"{runtime_kind} stream did not retain its dispatch session: active={active}",
            )
            for _ in range(32):
                line = response.readline()
                if not line:
                    break
                if line.lstrip().startswith(b"data:"):
                    first_event = line.strip()
                    break
            require(first_event, f"{runtime_kind} stream ended before its first SSE data event")
        finally:
            if response is not None:
                response.close()

        def session_released() -> dict[str, int] | None:
            metrics = self.dispatch_metrics()
            if metrics["active_sessions"] == before["active_sessions"]:
                return metrics
            return None

        released = self.wait_until(
            f"{runtime_kind} stream dispatch session release",
            self.args.convergence_timeout,
            session_released,
        )
        require(
            released["dispatched"] >= before["dispatched"] + 1,
            f"{runtime_kind} stream was not counted as dispatched: {before} -> {released}",
        )

        # A disconnect must cancel only the request, not poison the model runtime.
        self.chat(model_id)
        after_recovery = self.dispatch_metrics()
        require(
            after_recovery["active_sessions"] == 0
            and after_recovery["dispatched"] >= before["dispatched"] + 2,
            f"{runtime_kind} post-cancellation inference did not recover: {after_recovery}",
        )
        self.record(
            f"{runtime_kind} stream disconnect releases dispatch session",
            model=model_id,
            first_event_bytes=len(first_event),
            dispatched_before=before["dispatched"],
            dispatched_after=after_recovery["dispatched"],
            cached_endpoints=after_recovery["cached_endpoints"],
        )

    def kill_worker(self, process_id: int) -> None:
        require(process_id > 0, f"invalid Worker PID: {process_id}")
        self.worker_pids.add(process_id)
        log("INFO", f"Injecting Worker crash into PID {process_id}")
        try:
            os.kill(process_id, signal.SIGTERM)
        except ProcessLookupError as error:
            raise SmokeFailure(f"Worker PID {process_id} disappeared before injection") from error

    def wait_worker_converged(
        self,
        baseline_running: int,
        minimum_unexpected_exits: int,
        device_id: int,
        baseline_ledger: tuple[int, int],
    ) -> dict[str, Any]:
        def converged() -> dict[str, Any] | None:
            health = self.health()
            workers = health["workers"]
            ledger = self.ledger_state(self.device(health, device_id))
            if (
                workers["running"] == baseline_running
                and workers["unexpected_exits"] >= minimum_unexpected_exits
                and ledger == baseline_ledger
            ):
                return health
            return None

        return self.wait_until(
            "Worker crash convergence and ledger rollback",
            self.args.convergence_timeout,
            converged,
        )

    def unload(self, model_id: str) -> None:
        result = self.request(
            "POST",
            "/models/unload",
            {"model_id": model_id},
            timeout=self.args.load_timeout,
        )
        require(
            200 <= result.status < 300,
            f"unload {model_id} failed with HTTP {result.status}: {result.text[:2000]}",
        )
        if model_id in self.loaded_model_ids:
            self.loaded_model_ids.remove(model_id)

    def run_worker_soak(
        self,
        baseline_workers: int,
        worker_baseline: tuple[int, int],
        previous_generation: int,
    ) -> int:
        cycle_limit = self.args.soak_cycles
        duration_seconds = self.args.soak_duration_seconds
        if cycle_limit <= 0 and duration_seconds <= 0:
            return previous_generation
        require(self.server is not None, "server process is unavailable for soak sampling")
        meta_cache = self.args.worker_model.parent / "meta_cache"
        manifests_before = (
            {path.resolve() for path in meta_cache.glob("model_launch_*.hlif")}
            if meta_cache.is_dir()
            else set()
        )
        sample_before = process_resource_sample(self.server.pid)
        samples = [{"cycle": 0, **sample_before}]
        forced_baseline = int(self.health()["workers"]["forced_shutdowns"])
        soak_started = time.monotonic()
        soak_deadline = (
            soak_started + duration_seconds if duration_seconds > 0 else None
        )
        limit_text = (
            f"{cycle_limit} cycle(s)"
            if cycle_limit > 0
            else f"{duration_seconds:.0f} second(s)"
        )
        log("INFO", f"Starting dynamic Worker lifecycle soak for {limit_text}")

        cycle = 0
        while True:
            if cycle_limit > 0 and cycle >= cycle_limit:
                break
            if soak_deadline is not None and time.monotonic() >= soak_deadline:
                break
            cycle += 1
            load = self.load_model(
                self.args.worker_model, self.args.worker_mmproj, self.args.worker_device
            )
            process_id = int(load["process_id"])
            self.worker_pids.add(process_id)
            model_id, processor = self.wait_loaded(load)
            self.loaded_model_ids.append(model_id)
            require(processor is not None, f"soak cycle {cycle} Worker is missing from status")
            generation = int(processor.get("generation", 0))
            require(
                generation > previous_generation,
                f"soak generation did not advance: {previous_generation} -> {generation}",
            )
            previous_generation = generation
            self.wait_manifest_removed(load)
            if (
                self.args.soak_inference_every > 0
                and cycle % self.args.soak_inference_every == 0
            ):
                self.chat(model_id)

            loaded_health = self.health()
            loaded_device = self.device(loaded_health, self.args.worker_device)
            require(
                loaded_device["pending_reserved_mb"] == 0
                and loaded_device["committed_reserved_mb"] > worker_baseline[1],
                f"soak cycle {cycle} did not reach a committed Worker ledger state",
            )
            graceful_before = int(loaded_health["workers"]["graceful_shutdowns"])
            self.unload(model_id)

            def cycle_released() -> dict[str, Any] | None:
                health = self.health()
                if (
                    health["workers"]["running"] == baseline_workers
                    and health["workers"]["graceful_shutdowns"] >= graceful_before + 1
                    and health["workers"]["forced_shutdowns"] == forced_baseline
                    and self.ledger_state(self.device(health, self.args.worker_device))
                    == worker_baseline
                ):
                    return health
                return None

            self.wait_until(
                f"soak cycle {cycle} Worker release",
                self.args.convergence_timeout,
                cycle_released,
            )
            self.worker_pids.discard(process_id)
            self.wait_manifest_removed(load)
            if cycle % self.args.soak_progress_every == 0:
                sample = process_resource_sample(self.server.pid)
                samples.append({"cycle": cycle, **sample})
                elapsed = int(time.monotonic() - soak_started)
                log(
                    "INFO",
                    "Soak progress "
                    f"cycle={cycle}, elapsed_sec={elapsed}: handles={sample['handles']}, "
                    f"threads={sample['threads']}, rss_mb={sample['rss_bytes'] // (1024 * 1024)}",
                )

        completed_cycles = cycle
        require(completed_cycles > 0, "soak duration ended before one cycle completed")
        if samples[-1]["cycle"] != completed_cycles:
            final_sample = process_resource_sample(self.server.pid)
            samples.append({"cycle": completed_cycles, **final_sample})

        manifests_after = (
            {path.resolve() for path in meta_cache.glob("model_launch_*.hlif")}
            if meta_cache.is_dir()
            else set()
        )
        leaked_manifests = sorted(str(path) for path in manifests_after - manifests_before)
        require(not leaked_manifests, f"Worker launch manifests leaked: {leaked_manifests}")
        sample_after = samples[-1]
        require(
            sample_after["handles"] <= sample_before["handles"] + self.args.max_handle_growth,
            f"server handle growth exceeded limit: {sample_before} -> {sample_after}",
        )
        require(
            sample_after["threads"] <= sample_before["threads"] + self.args.max_thread_growth,
            f"server thread growth exceeded limit: {sample_before} -> {sample_after}",
        )
        max_rss_growth = self.args.max_rss_growth_mb * 1024 * 1024
        require(
            sample_after["rss_bytes"] <= sample_before["rss_bytes"] + max_rss_growth,
            f"server RSS growth exceeded limit: {sample_before} -> {sample_after}",
        )
        max_private_growth = self.args.max_private_growth_mb * 1024 * 1024
        if sample_before["private_bytes"] > 0 and sample_after["private_bytes"] > 0:
            require(
                sample_after["private_bytes"]
                <= sample_before["private_bytes"] + max_private_growth,
                f"server private-byte growth exceeded limit: {sample_before} -> {sample_after}",
            )
        tail_start_cycle = max(
            0, completed_cycles - max(10, completed_cycles // 5)
        )
        tail_sample = next(
            (sample for sample in samples if sample["cycle"] >= tail_start_cycle),
            sample_before,
        )
        max_tail_growth = self.args.max_tail_growth_mb * 1024 * 1024
        require(
            sample_after["rss_bytes"] <= tail_sample["rss_bytes"] + max_tail_growth,
            f"server RSS kept growing in the soak tail: {tail_sample} -> {sample_after}",
        )
        if tail_sample["private_bytes"] > 0 and sample_after["private_bytes"] > 0:
            require(
                sample_after["private_bytes"]
                <= tail_sample["private_bytes"] + max_tail_growth,
                f"server private bytes kept growing in the soak tail: {tail_sample} -> {sample_after}",
            )
        self.record(
            f"{completed_cycles}-cycle dynamic Worker lifecycle soak",
            cycles=completed_cycles,
            configured_cycle_limit=cycle_limit,
            configured_duration_seconds=duration_seconds,
            elapsed_seconds=int(time.monotonic() - soak_started),
            first_sample=sample_before,
            final_sample={key: value for key, value in sample_after.items() if key != "cycle"},
            samples=samples,
            leaked_manifests=0,
            tail_start_cycle=tail_start_cycle,
        )
        return previous_generation

    def run(self) -> None:
        self.validate_inputs()
        self.start_server()
        self.baseline_health = self.health()
        baseline_workers = int(self.baseline_health["workers"]["running"])
        require(baseline_workers == 0, "isolated server unexpectedly has running Workers")
        primary_baseline = self.ledger_state(
            self.device(self.baseline_health, self.args.primary_device)
        )
        worker_baseline = self.ledger_state(
            self.device(self.baseline_health, self.args.worker_device)
        )
        require(primary_baseline == (0, 0), f"primary device is not idle: {primary_baseline}")
        require(worker_baseline == (0, 0), f"Worker device is not idle: {worker_baseline}")
        self.record("baseline health and ledger invariants")

        primary_load = self.load_model(
            self.args.primary_model, self.args.primary_mmproj, self.args.primary_device
        )
        primary_id, _ = self.wait_loaded(primary_load)
        self.loaded_model_ids.append(primary_id)
        self.chat(primary_id)
        if self.args.check_stream_cancellation:
            self.check_stream_cancellation(primary_id, "embedded")
        self.benchmark_chat(primary_id, "embedded")
        primary_health = self.health()
        primary_ledger = self.device(primary_health, self.args.primary_device)
        require(primary_ledger["pending_reserved_mb"] == 0, "local load left pending memory")
        require(
            primary_ledger["committed_reserved_mb"] > primary_baseline[1],
            "local load did not commit a reservation",
        )
        self.record(
            "main-process model load, inference, and reservation commit",
            model=primary_id,
            committed_mb=primary_ledger["committed_reserved_mb"],
        )

        if self.args.primary_device == self.args.worker_device:
            worker_baseline = self.ledger_state(primary_ledger)

        after_rejection = self.health()
        if not self.args.skip_resource_rejection:
            before_device = self.ledger_state(
                self.device(after_rejection, self.args.primary_device)
            )
            rejection = self.request(
                "POST",
                "/models/load",
                self.load_body(
                    self.args.worker_model,
                    self.args.worker_mmproj,
                    self.args.primary_device,
                ),
                timeout=self.args.load_timeout,
            )
            require(
                rejection.status == 503,
                f"capacity rejection expected HTTP 503, got {rejection.status}: {rejection.text}",
            )
            after_rejection = self.health()
            require(
                self.ledger_state(self.device(after_rejection, self.args.primary_device))
                == before_device,
                "capacity rejection changed pending/committed ledger state",
            )
            require(
                after_rejection["workers"]["running"] == baseline_workers,
                "capacity rejection started a Worker",
            )
            self.record("resource rejection leaves the ledger unchanged", http_status=503)

        unexpected = int(after_rejection["workers"]["unexpected_exits"])
        first_generation = 0
        if not self.args.skip_pre_ready_crash:
            pre_ready_load = self.load_model(
                self.args.worker_model, self.args.worker_mmproj, self.args.worker_device
            )
            pre_ready_pid = int(pre_ready_load["process_id"])
            self.worker_pids.add(pre_ready_pid)
            processor = self.wait_until(
                "pre-READY Worker process registration",
                5.0,
                lambda: self.status_processor(pre_ready_pid),
            )
            first_generation = int(processor.get("generation", 0))
            pending_health = self.health()
            pending_ledger = self.device(pending_health, self.args.worker_device)
            require(
                pending_ledger["pending_reserved_mb"] > worker_baseline[0]
                and pending_ledger["committed_reserved_mb"] == worker_baseline[1],
                "Worker reached READY before pre-READY crash injection; rerun with a larger model "
                "or use --skip-pre-ready-crash",
            )
            self.kill_worker(pre_ready_pid)
            unexpected += 1
            self.wait_worker_converged(
                baseline_workers, unexpected, self.args.worker_device, worker_baseline
            )
            self.wait_manifest_removed(pre_ready_load)
            self.worker_pids.discard(pre_ready_pid)
            self.record(
                "pre-READY Worker crash rolls back pending reservation",
                process_id=pre_ready_pid,
                generation=first_generation,
                pending_mb=pending_ledger["pending_reserved_mb"],
            )

        worker_load = self.load_model(
            self.args.worker_model, self.args.worker_mmproj, self.args.worker_device
        )
        worker_pid = int(worker_load["process_id"])
        self.worker_pids.add(worker_pid)
        worker_id, worker_processor = self.wait_loaded(worker_load)
        self.wait_manifest_removed(worker_load)
        self.loaded_model_ids.append(worker_id)
        require(worker_processor is not None, "loaded Worker is missing from status")
        ready_generation = int(worker_processor.get("generation", 0))
        if first_generation:
            require(
                ready_generation > first_generation,
                f"Worker generation did not advance: {first_generation} -> {ready_generation}",
            )
        self.chat(worker_id)
        if self.args.check_stream_cancellation:
            self.check_stream_cancellation(worker_id, "process")
        self.benchmark_chat(worker_id, "process")
        worker_ready_health = self.health()
        worker_ready_ledger = self.device(worker_ready_health, self.args.worker_device)
        require(worker_ready_ledger["pending_reserved_mb"] == 0, "READY Worker left pending memory")
        require(
            worker_ready_ledger["committed_reserved_mb"] > worker_baseline[1],
            "READY Worker did not commit a reservation",
        )
        self.record(
            "dynamic Worker READY and proxy inference",
            model=worker_id,
            process_id=worker_pid,
            generation=ready_generation,
        )

        self.kill_worker(worker_pid)
        unexpected += 1
        self.wait_worker_converged(
            baseline_workers, unexpected, self.args.worker_device, worker_baseline
        )
        self.wait_manifest_removed(worker_load)
        self.worker_pids.discard(worker_pid)
        if worker_id in self.loaded_model_ids:
            self.loaded_model_ids.remove(worker_id)
        self.record(
            "READY Worker crash converges registry, route, process, and reservation",
            process_id=worker_pid,
        )

        recovered_load = self.load_model(
            self.args.worker_model, self.args.worker_mmproj, self.args.worker_device
        )
        recovered_pid = int(recovered_load["process_id"])
        self.worker_pids.add(recovered_pid)
        recovered_id, recovered_processor = self.wait_loaded(recovered_load)
        self.wait_manifest_removed(recovered_load)
        self.loaded_model_ids.append(recovered_id)
        require(recovered_processor is not None, "recovered Worker is missing from status")
        recovered_generation = int(recovered_processor.get("generation", 0))
        require(
            recovered_generation > ready_generation,
            f"recovery generation did not advance: {ready_generation} -> {recovered_generation}",
        )
        self.chat(recovered_id)
        before_graceful = self.health()
        graceful_before = int(before_graceful["workers"]["graceful_shutdowns"])
        forced_before = int(before_graceful["workers"]["forced_shutdowns"])
        self.unload(recovered_id)

        def graceful_done() -> dict[str, Any] | None:
            health = self.health()
            if (
                health["workers"]["running"] == baseline_workers
                and health["workers"]["graceful_shutdowns"] >= graceful_before + 1
                and health["workers"]["forced_shutdowns"] == forced_before
                and self.ledger_state(self.device(health, self.args.worker_device))
                == worker_baseline
            ):
                return health
            return None

        self.wait_until(
            "graceful Worker shutdown and reservation release",
            self.args.convergence_timeout,
            graceful_done,
        )
        self.worker_pids.discard(recovered_pid)
        self.wait_manifest_removed(recovered_load)
        self.record(
            "generation recovery and graceful Worker unload",
            process_id=recovered_pid,
            generation=recovered_generation,
        )

        recovered_generation = self.run_worker_soak(
            baseline_workers, worker_baseline, recovered_generation
        )

        self.unload(primary_id)

        def all_released() -> dict[str, Any] | None:
            health = self.health()
            primary_released = (
                self.ledger_state(self.device(health, self.args.primary_device))
                == primary_baseline
            )
            worker_released = (
                self.args.primary_device == self.args.worker_device
                or self.ledger_state(self.device(health, self.args.worker_device))
                == worker_baseline
            )
            if primary_released and worker_released:
                return health
            return None

        final_health = self.wait_until(
            "all model reservations released",
            self.args.convergence_timeout,
            all_released,
        )
        self.record(
            "final ledger returned to baseline",
            unexpected_exits=final_health["workers"]["unexpected_exits"],
            graceful_shutdowns=final_health["workers"]["graceful_shutdowns"],
        )

    def cleanup(self) -> None:
        if self.server is not None and self.server.poll() is None:
            for model_id in list(reversed(self.loaded_model_ids)):
                try:
                    self.request(
                        "POST",
                        "/models/unload",
                        {"model_id": model_id},
                        timeout=10.0,
                    )
                except Exception:
                    pass
            try:
                self.server.terminate()
                self.server.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait(timeout=5.0)
        for process_id in list(self.worker_pids):
            try:
                os.kill(process_id, signal.SIGTERM)
            except (OSError, ProcessLookupError, SystemError):
                pass
            finally:
                self.worker_pids.discard(process_id)
        if self.server_log is not None:
            self.server_log.close()
            self.server_log = None

    def verify_logs(self) -> None:
        require(self.server_log_path.exists(), "server console log was not created")
        content = self.server_log_path.read_text(encoding="utf-8", errors="replace")
        lowered = content.lower()
        require("[debug]" not in lowered, "INFO run emitted DEBUG log lines")
        require(
            "model service not initialized" not in lowered,
            "empty health polling emitted the old uninitialized-service warning",
        )
        expected = [
            ("[error]", "Worker exited:"),
            ("[info]", "Worker exited gracefully:"),
            ("[info]", "released device reservation"),
        ]
        if not self.args.skip_resource_rejection:
            expected.insert(0, ("[warning]", "Device reservation rejected"))
        for level, message in expected:
            require(
                any(level in line.lower() and message.lower() in line.lower() for line in content.splitlines()),
                f"missing expected {level} log for {message!r}",
            )
        self.record("INFO/WARN/ERROR logging contract")

    def write_summary(self, status: str, error: str | None = None) -> None:
        self.args.artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": status,
            "base_url": self.base_url,
            "server_executable": str(self.args.server_executable),
            "primary_model": str(self.args.primary_model),
            "primary_mmproj": (
                str(self.args.primary_mmproj)
                if self.args.primary_mmproj is not None
                else ""
            ),
            "worker_model": str(self.args.worker_model),
            "worker_mmproj": (
                str(self.args.worker_mmproj)
                if self.args.worker_mmproj is not None
                else ""
            ),
            "primary_device": self.args.primary_device,
            "worker_device": self.args.worker_device,
            "soak_cycles": self.args.soak_cycles,
            "soak_duration_seconds": self.args.soak_duration_seconds,
            "check_stream_cancellation": self.args.check_stream_cancellation,
            "chat_benchmark_samples": self.args.chat_benchmark_samples,
            "chat_benchmark_warmup": self.args.chat_benchmark_warmup,
            "steps": self.steps,
            "error": error,
        }
        self.summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    args = parse_args()
    runner = HardwareSmoke(args)
    failure: str | None = None
    try:
        runner.run()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        log("FAIL", failure)
        traceback.print_exc()
    finally:
        runner.cleanup()

    if failure is None:
        try:
            runner.verify_logs()
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            log("FAIL", failure)
            traceback.print_exc()

    runner.write_summary("passed" if failure is None else "failed", failure)
    if failure is not None:
        log("INFO", f"Artifacts: {runner.args.artifact_dir}")
        return 1
    log("PASS", f"P0 hardware smoke completed; artifacts: {runner.args.artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
