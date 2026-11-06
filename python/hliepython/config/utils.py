# Copyright 2024-2026 HLIEPython Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import os


def as_dict(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def as_list(value: Any, field_name: str) -> List[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    return value


def as_int(value: Any, default: int, field_name: str) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def normalize_model_root_dir(model_root_dir: Any) -> str:
    return str(Path(str(model_root_dir)).expanduser())


def resolve_model_path(model_path: Any, model_root_dir: str) -> str:
    normalized_path = str(model_path).strip()
    if not normalized_path:
        raise ValueError("models[].instances[].model_path must not be empty")

    path_obj = Path(normalized_path).expanduser()
    if path_obj.is_absolute():
        resolved_path = path_obj
    else:
        resolved_path = Path(model_root_dir).expanduser() / path_obj

    if resolved_path.suffix.lower() == ".gguf":
        return resolve_gguf_runtime_dir(resolved_path)

    return str(resolved_path)


def resolve_model_file_path(model_path: Any, model_root_dir: str) -> str:
    normalized_path = str(model_path).strip()
    if not normalized_path:
        raise ValueError("model file path must not be empty")

    path_obj = Path(normalized_path).expanduser()
    if path_obj.is_absolute():
        return str(path_obj)

    return str(Path(model_root_dir).expanduser() / path_obj)


def resolve_gguf_runtime_dir(gguf_path: Path) -> str:
    gguf_path = gguf_path.expanduser()
    if not gguf_path.is_absolute():
        gguf_path = gguf_path.resolve()

    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF 模型文件不存在: {gguf_path}")
    if not gguf_path.is_file():
        raise ValueError(f"GGUF 模型路径不是文件: {gguf_path}")

    runtime_root = gguf_path.parent / ".gguf_model_dirs"
    runtime_dir = runtime_root / gguf_path.stem
    runtime_dir.mkdir(parents=True, exist_ok=True)

    link_path = runtime_dir / gguf_path.name
    if link_path.is_symlink():
        if os.readlink(link_path) != str(gguf_path):
            link_path.unlink()
            link_path.symlink_to(gguf_path)
    elif link_path.exists():
        if link_path.resolve() != gguf_path.resolve():
            raise FileExistsError(f"GGUF 运行目录中已存在同名非目标文件: {link_path}")
    else:
        link_path.symlink_to(gguf_path)

    return str(runtime_dir)
