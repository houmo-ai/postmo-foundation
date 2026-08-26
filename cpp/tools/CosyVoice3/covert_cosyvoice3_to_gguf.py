#!/usr/bin/env python3
"""Convert a CosyVoice3 deployment directory into one GGUF file.

The five PyTorch embedding files are converted to FP16 bytes in memory. Only
their .bin tensor names are written to GGUF; no intermediate .bin files and no
.pt files are included in the output.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = SCRIPT_DIR / "Fun-CosyVoice3-0.5B-2512"

EMBEDDINGS = {
    "quant_embedding.bin": ("quant_embedding.pt", (151936, 896), "weight"),
    "flow_input_embedding.bin": ("flow_input_embedding.pt", (6561, 80), None),
    "llm_sos_embedding.bin": ("llm_sos_embedding.pt", (1, 1, 896), None),
    "llm_speech_embedding.bin": ("llm_speech_embedding.pt", (6761, 896), None),
    "llm_task_id_embedding.bin": (
        "llm_task_id_embedding.pt",
        (1, 1, 896),
        None,
    ),
}

HMM_FILES = (
    "cosyvoice3_campplus.hmm",
    "cosyvoice3_flow_decoder.hmm",
    "cosyvoice3_flow_encoder.hmm",
    "cosyvoice3_flow_spk.hmm",
    "cosyvoice3_hift_part1.hmm",
    "cosyvoice3_hift_part2.hmm",
    "cosyvoice3_llm_decoder.hmm",
    "cosyvoice3_llm_qwen2_decode.hmm",
    "cosyvoice3_llm_qwen2_prefill.hmm",
    "cosyvoice3_speech_tokenizer.hmm",
)

TOKENIZER_FILES = (
    "config.json",
    "vocab.json",
    "tokenizer_config.json",
    "merges.txt",
    "generation_config.json",
)


def import_gguf(explicit_path: Path | None) -> Any:
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    if os.environ.get("GGUF_PY_PATH"):
        candidates.append(Path(os.environ["GGUF_PY_PATH"]))
    candidates.extend(
        (
            SCRIPT_DIR / "gguf-py",
            SCRIPT_DIR.parents[3] / "llama.cpp" / "gguf-py",
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


def resolve_model_dirs(input_dir: Path) -> tuple[Path, Path]:
    input_dir = input_dir.resolve()
    if (input_dir / "hmquant").is_dir():
        return input_dir, input_dir / "hmquant"
    if input_dir.name.lower() == "hmquant":
        return input_dir.parent, input_dir
    raise FileNotFoundError(
        f"{input_dir} is neither a model directory containing hmquant nor an "
        "hmquant directory"
    )


def find_tokenizer_dir(config_dir: Path) -> Path:
    direct = config_dir / "CosyVoice-BlankEN"
    if direct.is_dir():
        return direct
    if all((config_dir / name).is_file() for name in TOKENIZER_FILES):
        return config_dir
    raise FileNotFoundError(
        f"Cannot find CosyVoice-BlankEN tokenizer files under {config_dir}"
    )


def require_files(directory: Path, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"Missing files under {directory}:\n  {joined}")


class GGUFPacker:
    def __init__(self, output_path: Path, gguf_module: Any):
        self.output_path = output_path
        self.gguf = gguf_module
        self.tensors: list[tuple[str, np.ndarray]] = []
        self.file_info: dict[str, dict[str, Any]] = {}
        self.vocab: list[str] = []
        self.merges: list[str] = []

    def add_raw_file(self, name: str, path: Path) -> None:
        data = np.memmap(path, mode="r", dtype=np.uint8)
        self.tensors.append((name, data))
        self.file_info[name] = {
            "original_path": str(path),
            "size": path.stat().st_size,
            "format": path.suffix,
        }
        print(f"Added {name}: {data.nbytes / 1024 / 1024:.2f} MiB")

    def add_embedding(
        self,
        output_name: str,
        path: Path,
        expected_shape: tuple[int, ...],
        state_key: str | None,
    ) -> None:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if state_key is not None:
            if not isinstance(value, dict) or state_key not in value:
                raise ValueError(f"{path} does not contain key {state_key!r}")
            value = value[state_key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{path} contains {type(value).__name__}, not a tensor")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{path} has shape {tuple(value.shape)}, expected {expected_shape}"
            )

        tensor = value.detach().to(dtype=torch.float16).contiguous().cpu()
        data = tensor.numpy().view(np.uint8).reshape(-1)
        self.tensors.append((output_name, data))
        self.file_info[output_name] = {
            "original_path": str(path),
            "size": data.nbytes,
            "format": ".bin",
            "dtype": "float16",
            "shape": list(expected_shape),
        }
        print(
            f"Converted {path.name} -> {output_name}: "
            f"shape={expected_shape}, {data.nbytes / 1024 / 1024:.2f} MiB"
        )

    def load_tokenizer_metadata(self, tokenizer_dir: Path) -> None:
        with (tokenizer_dir / "vocab.json").open(encoding="utf-8") as stream:
            vocab_map = json.load(stream)
        if not isinstance(vocab_map, dict):
            raise TypeError("vocab.json must contain a JSON object")

        ids = sorted(vocab_map.values())
        if ids != list(range(len(vocab_map))):
            raise ValueError("vocab.json token IDs must be contiguous from zero")
        self.vocab = [
            token for token, _ in sorted(vocab_map.items(), key=lambda item: item[1])
        ]

        with (tokenizer_dir / "merges.txt").open(encoding="utf-8") as stream:
            self.merges = [line.strip() for line in stream if line.strip()]
        print(f"Loaded tokenizer metadata: {len(self.vocab)} tokens, "
              f"{len(self.merges)} merges")

    def write(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        total_bytes = sum(data.nbytes for _, data in self.tensors)
        print(
            f"Writing {len(self.tensors)} tensors "
            f"({total_bytes / 1024 / 1024:.2f} MiB) to {self.output_path}"
        )

        writer = self.gguf.GGUFWriter(
            path=None,
            arch="cosyvoice3",
            endianess=self.gguf.GGUFEndian.LITTLE,
        )
        writer.add_string("model_name", "CosyVoice3")
        writer.add_string("model_type", "cosyvoice3_tts")
        writer.add_int32("total_file_count", len(self.tensors))
        writer.add_string("file_info", json.dumps(self.file_info, ensure_ascii=False))
        writer.add_int32("vocab_size", len(self.vocab))
        writer.add_array("vocab", self.vocab)
        writer.add_int32("merges_size", len(self.merges))
        writer.add_array("merges", self.merges)

        for name, data in self.tensors:
            writer.add_tensor(
                name=name,
                tensor=data,
                raw_dtype=self.gguf.GGMLQuantizationType.I8,
            )

        writer.write_header_to_file(path=self.output_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        print(
            f"Done: {self.output_path} "
            f"({self.output_path.stat().st_size / 1024 / 1024:.2f} MiB)"
        )


def convert(
    input_dir: Path,
    config_dir: Path,
    output_path: Path,
    gguf_py: Path | None = None,
) -> None:
    model_dir, weight_dir = resolve_model_dirs(input_dir)
    tokenizer_dir = find_tokenizer_dir(config_dir.resolve())

    require_files(model_dir, HMM_FILES)
    require_files(weight_dir, tuple(spec[0] for spec in EMBEDDINGS.values()))
    require_files(tokenizer_dir, TOKENIZER_FILES)

    packer = GGUFPacker(output_path.resolve(), import_gguf(gguf_py))
    for output_name, (source_name, shape, state_key) in EMBEDDINGS.items():
        packer.add_embedding(
            output_name,
            weight_dir / source_name,
            shape,
            state_key,
        )
    for name in HMM_FILES:
        packer.add_raw_file(name, model_dir / name)
    for name in TOKENIZER_FILES:
        packer.add_raw_file(name, tokenizer_dir / name)
    packer.load_tokenizer_metadata(tokenizer_dir)
    packer.write()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CosyVoice3 HMM/PT files into one GGUF file"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Model root containing hmquant, or the hmquant directory itself",
    )
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument(
        "-c",
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help=f"Official model configuration directory (default: {DEFAULT_CONFIG_DIR})",
    )
    parser.add_argument(
        "--gguf-py",
        type=Path,
        help="Path containing the gguf Python package",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(args.input, args.config_dir, args.output, args.gguf_py)


if __name__ == "__main__":
    main()
