#!/usr/bin/env python3
"""Shared helpers for packing deployment artifacts into a GGUF bundle."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def import_gguf(explicit_path: Path | None, script_dir: Path) -> Any:
    """Import gguf-py from an explicit path, the environment, or the repo."""
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(explicit_path)
    if os.environ.get("GGUF_PY_PATH"):
        candidates.append(Path(os.environ["GGUF_PY_PATH"]))
    candidates.extend(
        (
            script_dir / "gguf-py",
            script_dir.parent / "gguf-py",
            script_dir.parents[2] / "llama.cpp" / "gguf-py",
        )
    )

    for candidate in candidates:
        if (candidate / "gguf").is_dir():
            sys.path.insert(0, str(candidate))
            break

    try:
        return importlib.import_module("gguf")
    except ModuleNotFoundError as exc:
        searched = "\n  ".join(str(path) for path in candidates)
        raise RuntimeError(
            "Cannot import gguf. Install gguf-py or pass --gguf-py.\n"
            f"Searched:\n  {searched}"
        ) from exc


class GGUFBundleWriter:
    """Collect typed tensors and raw component files into one GGUF file."""

    def __init__(
        self,
        output_path: Path,
        architecture: str,
        gguf_module: Any,
    ) -> None:
        self.output_path = output_path.resolve()
        self.architecture = architecture
        self.gguf = gguf_module
        self.tensors: list[tuple[str, np.ndarray, Any | None]] = []
        self.file_info: dict[str, dict[str, Any]] = {}

    def add_raw_file(self, name: str, path: Path) -> None:
        path = path.resolve()
        data = np.memmap(path, mode="r", dtype=np.uint8)
        self.tensors.append(
            (name, data, self.gguf.GGMLQuantizationType.I8)
        )
        self.file_info[name] = {
            "source_name": path.name,
            "size": path.stat().st_size,
            "format": path.suffix.lower(),
            "storage": "raw_bytes",
        }
        print(f"Added {name}: {data.nbytes / 1024 / 1024:.2f} MiB")

    def add_tensor(
        self,
        name: str,
        data: np.ndarray,
        source_name: str,
    ) -> None:
        if not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)
        self.tensors.append((name, data, None))
        self.file_info[name] = {
            "source_name": source_name,
            "size": data.nbytes,
            "format": "tensor",
            "storage": "typed_tensor",
            "dtype": str(data.dtype),
            "shape": list(data.shape),
        }
        print(
            f"Added {name}: shape={tuple(data.shape)}, dtype={data.dtype}, "
            f"{data.nbytes / 1024 / 1024:.2f} MiB"
        )

    def add_binary_tensor(
        self,
        name: str,
        data: np.ndarray,
        source_name: str,
    ) -> None:
        """Store a tensor as a flat byte resource while retaining its layout metadata."""
        if not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)
        binary = data.view(np.uint8).reshape(-1)
        self.tensors.append(
            (name, binary, self.gguf.GGMLQuantizationType.I8)
        )
        self.file_info[name] = {
            "source_name": source_name,
            "size": binary.nbytes,
            "format": ".bin",
            "storage": "raw_bytes",
            "dtype": str(data.dtype),
            "shape": list(data.shape),
        }
        print(
            f"Added {name}: binary shape={tuple(data.shape)}, dtype={data.dtype}, "
            f"{binary.nbytes / 1024 / 1024:.2f} MiB"
        )

    def write(
        self,
        model_name: str,
        model_type: str,
        add_metadata: Any,
    ) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        total_bytes = sum(data.nbytes for _, data, _ in self.tensors)
        print(
            f"Writing {len(self.tensors)} tensors "
            f"({total_bytes / 1024 / 1024:.2f} MiB) to {self.output_path}"
        )

        writer = self.gguf.GGUFWriter(
            path=None,
            arch=self.architecture,
            endianess=self.gguf.GGUFEndian.LITTLE,
        )
        writer.add_string("model_name", model_name)
        writer.add_string("model_type", model_type)
        writer.add_int32("total_file_count", len(self.tensors))
        writer.add_string(
            "file_info", json.dumps(self.file_info, ensure_ascii=False)
        )
        add_metadata(writer)

        for name, data, raw_dtype in self.tensors:
            if raw_dtype is None:
                writer.add_tensor(name=name, tensor=data)
            else:
                writer.add_tensor(name=name, tensor=data, raw_dtype=raw_dtype)

        writer.write_header_to_file(path=self.output_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        print(
            f"Done: {self.output_path} "
            f"({self.output_path.stat().st_size / 1024 / 1024:.2f} MiB)"
        )
