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

import gc
import glob
import json
import os
from pathlib import Path
from typing import Any

try:
    from gguf.gguf_reader import GGUFReader
except ImportError:  # 兼容部分 gguf 版本
    from gguf import GGUFReader


_CONFIG_TENSOR_SUFFIXES = (".json", ".txt", ".jinja", ".md")


def resolve_gguf_path(model_path: str) -> str:
    """解析 GGUF 路径。

    - 如果传入文件，直接返回绝对路径；
    - 如果传入目录，查找目录下唯一的 HiMode*.gguf 文件。
    """
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("model_path must be a non-empty string")

    abs_path = os.path.abspath(model_path.strip())
    if os.path.isfile(abs_path):
        return abs_path

    if os.path.isdir(abs_path):
        gguf_files = sorted(glob.glob(os.path.join(abs_path, "HiMode*.gguf")))
        if not gguf_files:
            raise FileNotFoundError(f"目录下未找到 HiMode*.gguf 文件: {abs_path}")
        if len(gguf_files) > 1:
            raise ValueError(f"目录下存在多个 HiMode*.gguf 文件，无法确定唯一模型: {gguf_files}")
        return os.path.abspath(gguf_files[0])

    raise FileNotFoundError(f"路径不存在或不是有效文件/目录: {abs_path}")


def _to_python_value(value: Any) -> Any:
    """将 numpy 标量/数组转换为 Python 可序列化对象。"""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _read_field_value(field: Any) -> Any:
    """按 GGUFReader 字段结构读取字段值。"""
    if hasattr(field, "parts") and hasattr(field, "data") and field.parts is not None:
        try:
            return _to_python_value(field.parts[field.data[0]])
        except Exception:
            pass

    if hasattr(field, "parts") and field.parts is not None:
        parts = [_to_python_value(part) for part in field.parts]
        return parts[0] if len(parts) == 1 else parts

    if hasattr(field, "value"):
        return _to_python_value(field.value)

    return _to_python_value(field)


def get_gguf_metadata(
    model_path: str,
    *,
    include_tensors: bool = False,
    include_config_tensors: bool = True,
) -> dict[str, Any]:
    """读取 GGUF 元信息。

    Args:
        model_path: GGUF 文件路径，或包含唯一 HiMode*.gguf 的目录。
        include_tensors: 是否返回张量元信息。
        include_config_tensors: include_tensors=True 时，是否包含 .json/.txt/.jinja/.md 等配置张量。

    Returns:
        {
            "model_path": str,
            "model_dir": str,
            "metadata": dict,
            "tensors": list[dict],  # 仅 include_tensors=True 时返回
        }
    """
    gguf_path = resolve_gguf_path(model_path)
    if not Path(gguf_path).exists():
        raise FileNotFoundError(f"GGUF 文件不存在: {gguf_path}")

    reader = None
    try:
        reader = GGUFReader(gguf_path)
        result: dict[str, Any] = {
            "model_path": gguf_path,
            "model_dir": os.path.dirname(gguf_path),
            "metadata": {
                key: _read_field_value(field)
                for key, field in reader.fields.items()
            },
        }

        if include_tensors:
            tensors = []
            for tensor in reader.tensors:
                name = str(tensor.name)
                is_config_tensor = name.endswith(_CONFIG_TENSOR_SUFFIXES)
                if is_config_tensor and not include_config_tensors:
                    continue

                shape = []
                for item in list(tensor.shape):
                    if hasattr(item, "item"):
                        shape.append(int(item.item()))
                    elif hasattr(item, "tolist"):
                        shape.append(item.tolist())
                    else:
                        shape.append(int(item))

                tensors.append(
                    {
                        "name": name,
                        "shape": tuple(shape),
                        "n_elements": int(tensor.n_elements),
                        "quantization": str(tensor.tensor_type.name),
                        "data_offset": int(tensor.data_offset),
                        "is_config_tensor": is_config_tensor,
                    }
                )

            result["tensors"] = tensors

        return result
    finally:
        close_reader = getattr(reader, "close", None) if reader is not None else None
        if callable(close_reader):
            close_reader()
        gc.collect()


def get_gguf_metainfo(model_path: str) -> dict[str, Any]:
    """兼容旧命名：仅返回 GGUF metadata 字典。"""
    return get_gguf_metadata(model_path)["metadata"]


def _decode_bytes_like(value: Any) -> str | None:
    """将 GGUF 中常见的 bytes/list[int] 字符串表示解码为文本。"""
    value = _to_python_value(value)

    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, bytearray):
        return bytes(value).decode("utf-8")
    if isinstance(value, list):
        try:
            if all(isinstance(item, int) for item in value):
                return bytes(value).decode("utf-8")
        except ValueError:
            return None

    return None


def get_hmm_info_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """从 GGUF metadata 的 hmm.info 字段解析 HMM 信息。"""
    raw_hmm_info = metadata.get("hmm.info")
    if raw_hmm_info is None:
        return {}

    hmm_info_text = _decode_bytes_like(raw_hmm_info)
    if not hmm_info_text:
        return {}

    try:
        hmm_info = json.loads(hmm_info_text)
    except json.JSONDecodeError:
        return {}

    return hmm_info if isinstance(hmm_info, dict) else {}


def get_hmm_info(model_path: str) -> dict[str, Any]:
    """读取 GGUF metadata 中的 hmm.info JSON 对象。"""
    return get_hmm_info_from_metadata(get_gguf_metainfo(model_path))


def _positive_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        return None
    return int_value if int_value > 0 else None


def get_vllm_defaults_from_hmm_info(model_path: str) -> dict[str, int]:
    """从 GGUF hmm.info 提取可用于 vLLM 的默认启动参数。"""
    hmm_info = get_hmm_info(model_path)
    batch = _positive_int_or_none(hmm_info.get("batch"))
    context_length = _positive_int_or_none(hmm_info.get("context_length"))

    defaults: dict[str, int] = {}
    if batch is not None:
        defaults["max_num_seqs"] = batch
    if context_length is not None:
        defaults["max_model_len"] = context_length
        defaults["max_num_batched_tokens"] = context_length
    return defaults


def apply_gguf_metadata_defaults_to_vllm_engine_args(
    model_path: str,
    vllm_engine_args: dict[str, Any],
) -> dict[str, Any]:
    """当用户未配置相关 vLLM 参数时，用 GGUF hmm.info 中的值补齐。

    - hmm.info.batch -> max_num_seqs
    - hmm.info.context_length -> max_model_len / max_num_batched_tokens
    - metadata 值为 None、0 或非法值时不设置，交给 vLLM 默认值。
    """
    engine_args = dict(vllm_engine_args or {})
    defaults = get_vllm_defaults_from_hmm_info(model_path)

    for key, value in defaults.items():
        engine_args.setdefault(key, value)

    return engine_args
