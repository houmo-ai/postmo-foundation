#!/usr/bin/env python3
"""Convert a Qwen3-TTS HMM deployment into one GGUF bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from gguf_bundle import GGUFBundleWriter, import_gguf  # noqa: E402


COMMON_HMM_COMPONENTS = {
    "qwen3_tts.text_projection.hmm": "text_projection.hmm",
    "qwen3_tts.talker.prefill.hmm": "talker_prefill.hmm",
    "qwen3_tts.talker.decode.hmm": "talker_decode.hmm",
    "qwen3_tts.code_predictor.prefill.hmm": "code_predictor_prefill.hmm",
    "qwen3_tts.code_predictor.decode.hmm": "code_predictor_decode.hmm",
    "qwen3_tts.speech_tokenizer.hmm": "speech_tokenizer.hmm",
    "qwen3_tts.stateful_decoder.hmm": "stateful_decoder.hmm",
}

VARIANT_HMM_COMPONENTS = {
    "custom_voice": {},
    "base": {
        "qwen3_tts.speech_tokenizer_encoder.hmm":
            "speech_tokenizer_encoder.hmm",
        "qwen3_tts.speaker_encoder.hmm": "speaker_encoder.hmm",
    },
}

CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "speech_tokenizer/config.json",
    "speech_tokenizer/configuration.json",
    "speech_tokenizer/preprocessor_config.json",
)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def require_files(root: Path, relative_names: tuple[str, ...]) -> None:
    missing = [name for name in relative_names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing files under {root}:\n  " + "\n  ".join(missing)
        )


def resolve_hmm_files(model_dir: Path, variant: str) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    hmm_files = list(model_dir.glob("*.hmm"))
    components = {
        **COMMON_HMM_COMPONENTS,
        **VARIANT_HMM_COMPONENTS[variant],
    }
    for bundle_name, suffix in components.items():
        matches = [path for path in hmm_files if path.name.endswith(suffix)]
        if len(matches) != 1:
            names = ", ".join(path.name for path in matches) or "none"
            raise FileNotFoundError(
                f"Expected exactly one *{suffix} under {model_dir}, found {names}"
            )
        resolved[bundle_name] = matches[0]
    return resolved


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not value:
        raise TypeError(f"{path} does not contain a non-empty state dict")
    for key, tensor in value.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{path} contains an invalid state entry {key!r}")
    return value


def as_fp16_numpy(tensor: torch.Tensor, source: Path) -> np.ndarray:
    if tensor.dtype != torch.float16:
        print(f"Converting {source.name} from {tensor.dtype} to float16")
    return tensor.detach().to(dtype=torch.float16).contiguous().cpu().numpy()


def build_vocabulary(config_dir: Path) -> tuple[list[str], list[str]]:
    vocab_map = load_json(config_dir / "vocab.json")
    tokenizer_config = load_json(config_dir / "tokenizer_config.json")
    if not isinstance(vocab_map, dict):
        raise TypeError("vocab.json must contain an object")

    token_by_id: dict[int, str] = {}
    for token, token_id in vocab_map.items():
        if not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"Invalid token ID for {token!r}: {token_id!r}")
        previous = token_by_id.get(token_id)
        if previous is not None and previous != token:
            raise ValueError(
                f"Duplicate tokenizer ID {token_id}: {previous!r} and {token!r}"
            )
        token_by_id[token_id] = token

    added = tokenizer_config.get("added_tokens_decoder", {})
    if not isinstance(added, dict):
        raise TypeError("tokenizer_config.added_tokens_decoder must be an object")
    for token_id_text, item in added.items():
        token_id = int(token_id_text)
        content = item.get("content") if isinstance(item, dict) else None
        if not isinstance(content, str):
            raise ValueError(f"Invalid added token at ID {token_id}")
        if token_id in token_by_id and token_by_id[token_id] != content:
            raise ValueError(f"Conflicting token values at ID {token_id}")
        token_by_id[token_id] = content

    max_id = max(token_by_id)
    missing = [token_id for token_id in range(max_id + 1) if token_id not in token_by_id]
    if missing:
        raise ValueError(f"Tokenizer IDs are not contiguous; first missing ID is {missing[0]}")
    vocab = [token_by_id[token_id] for token_id in range(max_id + 1)]

    with (config_dir / "merges.txt").open(encoding="utf-8") as stream:
        merges = [
            line.strip()
            for line in stream
            if line.strip() and not line.startswith("#")
        ]
    return vocab, merges


def validate_config(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if config.get("model_type") != "qwen3_tts":
        raise ValueError("config.json is not a qwen3_tts configuration")
    variant = config.get("tts_model_type")
    if not isinstance(variant, str) or variant not in VARIANT_HMM_COMPONENTS:
        raise ValueError(
            "Unsupported Qwen3-TTS variant "
            f"{variant!r}; expected 'custom_voice' or 'base'"
        )
    if config.get("tts_model_size") not in {"0b6", "1b7"}:
        raise ValueError(
            "Unsupported Qwen3-TTS model size "
            f"{config.get('tts_model_size')!r}; expected '0b6' or '1b7'"
        )

    talker = config.get("talker_config")
    if not isinstance(talker, dict):
        raise ValueError("config.json is missing talker_config")
    required = (
        "hidden_size",
        "text_hidden_size",
        "text_vocab_size",
        "vocab_size",
        "num_code_groups",
        "codec_eos_token_id",
        "codec_bos_id",
        "codec_pad_id",
        "codec_language_id",
        "spk_id",
        "spk_is_dialect",
    )
    missing = [key for key in required if key not in talker]
    if missing:
        raise ValueError("talker_config is missing: " + ", ".join(missing))
    if talker["num_code_groups"] != 16:
        raise ValueError("Expected num_code_groups=16")
    for key in ("spk_id", "spk_is_dialect", "codec_language_id"):
        if not isinstance(talker[key], dict):
            raise ValueError(f"talker_config.{key} must be an object")
    if variant == "custom_voice" and not talker["spk_id"]:
        raise ValueError("CustomVoice requires a non-empty talker_config.spk_id")
    if variant == "base":
        speaker_encoder = config.get("speaker_encoder_config")
        if not isinstance(speaker_encoder, dict):
            raise ValueError("Base config is missing speaker_encoder_config")
        for key in ("enc_dim", "sample_rate"):
            value = speaker_encoder.get(key)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"speaker_encoder_config.{key} must be a positive integer"
                )
        if speaker_encoder["enc_dim"] != talker["hidden_size"]:
            raise ValueError(
                "speaker encoder dimension must match Talker hidden size: "
                f"{speaker_encoder['enc_dim']} != {talker['hidden_size']}"
            )
    return variant, talker


def validate_deployment_size(
    hmm_files: dict[str, Path], configured_size: str
) -> None:
    expected_tag = {"0b6": "0.6b", "1b7": "1.7b"}[configured_size]
    conflicting_tag = {"0b6": "1.7b", "1b7": "0.6b"}[configured_size]
    names = [path.name.lower() for path in hmm_files.values()]
    if any(conflicting_tag in name for name in names):
        raise ValueError(
            f"HMM filenames identify a {conflicting_tag} deployment, but "
            f"config.json identifies {expected_tag}"
        )


def validate_deployment_variant(
    hmm_files: dict[str, Path], configured_variant: str
) -> None:
    names = [path.name.lower() for path in hmm_files.values()]
    if configured_variant == "base" and any(
        "customvoice" in name or "custom_voice" in name for name in names
    ):
        raise ValueError(
            "HMM filenames identify a CustomVoice deployment, but config.json "
            "identifies Base"
        )
    if configured_variant == "custom_voice" and any(
        "-base_" in name or "_base_" in name for name in names
    ):
        raise ValueError(
            "HMM filenames identify a Base deployment, but config.json "
            "identifies CustomVoice"
        )


def find_sampling_rate(config: dict[str, Any]) -> int:
    direct = config.get("sampling_rate")
    if isinstance(direct, int):
        return direct
    for value in config.values():
        if isinstance(value, dict):
            nested = find_sampling_rate(value)
            if nested > 0:
                return nested
    return 0


def add_embeddings(
    packer: GGUFBundleWriter,
    weight_dir: Path,
    talker: dict[str, Any],
) -> None:
    talker_state = load_state_dict(weight_dir / "quant_embedding.pt")
    if set(talker_state) != {"weight"}:
        raise ValueError("quant_embedding.pt must contain only 'weight'")
    talker_weight = talker_state["weight"]
    expected_talker = (talker["vocab_size"], talker["hidden_size"])
    if tuple(talker_weight.shape) != expected_talker:
        raise ValueError(
            f"Talker embedding shape {tuple(talker_weight.shape)} != {expected_talker}"
        )
    packer.add_binary_tensor(
        "quant_embedding.bin",
        as_fp16_numpy(talker_weight, weight_dir / "quant_embedding.pt"),
        "quant_embedding.pt:weight",
    )

    text_state = load_state_dict(weight_dir / "text_embedding.pt")
    if set(text_state) != {"weight"}:
        raise ValueError("text_embedding.pt must contain only 'weight'")
    text_weight = text_state["weight"]
    expected_text = (talker["text_vocab_size"], talker["text_hidden_size"])
    if tuple(text_weight.shape) != expected_text:
        raise ValueError(
            f"Text embedding shape {tuple(text_weight.shape)} != {expected_text}"
        )
    packer.add_binary_tensor(
        "text_embedding.bin",
        as_fp16_numpy(text_weight, weight_dir / "text_embedding.pt"),
        "text_embedding.pt:weight",
    )

    predictor_path = weight_dir / "quant_embedding_code_predictor.pt"
    predictor_state = load_state_dict(predictor_path)
    expected_keys = {
        f"{index}.weight" for index in range(talker["num_code_groups"] - 1)
    }
    if set(predictor_state) != expected_keys:
        raise ValueError(
            "Code Predictor embedding keys do not match num_code_groups - 1"
        )
    predictor_vocab = talker["code_predictor_config"]["vocab_size"]
    predictor_weights: list[np.ndarray] = []
    for index in range(talker["num_code_groups"] - 1):
        key = f"{index}.weight"
        tensor = predictor_state[key]
        expected = (predictor_vocab, talker["hidden_size"])
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"Code Predictor embedding {key} shape {tuple(tensor.shape)} "
                f"!= {expected}"
            )
        predictor_weights.append(as_fp16_numpy(tensor, predictor_path))
    packer.add_binary_tensor(
        "quant_embedding_code_predictor.bin",
        np.stack(predictor_weights, axis=0),
        predictor_path.name,
    )


def convert(
    input_dir: Path,
    config_dir: Path,
    output_path: Path,
    gguf_py: Path | None,
    deployment_meta_path: Path | None,
) -> None:
    model_dir = input_dir.resolve()
    weight_dir = model_dir / "hmquant"
    config_dir = config_dir.resolve()
    if not weight_dir.is_dir():
        raise FileNotFoundError(f"Missing hmquant directory under {model_dir}")

    require_files(
        weight_dir,
        (
            "quant_embedding.pt",
            "quant_embedding_code_predictor.pt",
            "text_embedding.pt",
            "decode_padding_shapes.json",
        ),
    )
    require_files(config_dir, CONFIG_FILES)
    config = load_json(config_dir / "config.json")
    variant, talker = validate_config(config)
    hmm_files = resolve_hmm_files(model_dir, variant)
    validate_deployment_size(hmm_files, config["tts_model_size"])
    validate_deployment_variant(hmm_files, variant)
    generation = load_json(config_dir / "generation_config.json")
    speech_tokenizer_config = load_json(config_dir / "speech_tokenizer/config.json")
    sample_rate = find_sampling_rate(speech_tokenizer_config)
    if sample_rate <= 0:
        raise ValueError("speech tokenizer config does not define sampling_rate")
    if (
        variant == "base"
        and config["speaker_encoder_config"]["sample_rate"] != sample_rate
    ):
        raise ValueError(
            "speaker encoder and speech tokenizer sample rates differ: "
            f"{config['speaker_encoder_config']['sample_rate']} != {sample_rate}"
        )
    decode_padding_shapes = load_json(weight_dir / "decode_padding_shapes.json")
    deployment_meta: dict[str, Any] = {}
    if deployment_meta_path is not None:
        value = load_json(deployment_meta_path.resolve())
        if not isinstance(value, dict):
            raise TypeError("deployment meta must contain a JSON object")
        deployment_meta = value
    samples_per_frame = deployment_meta.get("stateful_samples_per_frame", 1920)
    initial_skip_frames = deployment_meta.get(
        "stateful_initial_output_skip_frames", 4
    )
    if not isinstance(samples_per_frame, int) or samples_per_frame <= 0:
        raise ValueError("stateful_samples_per_frame must be a positive integer")
    if not isinstance(initial_skip_frames, int) or initial_skip_frames < 0:
        raise ValueError(
            "stateful_initial_output_skip_frames must be a non-negative integer"
        )
    vocab, merges = build_vocabulary(config_dir)
    if len(vocab) > talker["text_vocab_size"]:
        raise ValueError(
            f"Tokenizer size {len(vocab)} exceeds text embedding rows "
            f"{talker['text_vocab_size']}"
        )

    gguf_module = import_gguf(gguf_py, SCRIPT_DIR)
    packer = GGUFBundleWriter(output_path, "qwen3_tts", gguf_module)
    add_embeddings(packer, weight_dir, talker)
    for bundle_name, path in hmm_files.items():
        packer.add_raw_file(bundle_name, path)
    packer.add_raw_file(
        "qwen3_tts.decode_padding_shapes.json",
        weight_dir / "decode_padding_shapes.json",
    )
    for relative_name in CONFIG_FILES:
        packer.add_raw_file(
            f"qwen3_tts.resource.{relative_name.replace('/', '.')}",
            config_dir / relative_name,
        )

    def add_metadata(writer: Any) -> None:
        writer.add_string("qwen3_tts.variant", variant)
        writer.add_string("qwen3_tts.model_size", config["tts_model_size"])
        writer.add_int32("qwen3_tts.sample_rate", sample_rate)
        writer.add_int32("qwen3_tts.num_code_groups", talker["num_code_groups"])
        writer.add_int32(
            "qwen3_tts.stateful.samples_per_frame", samples_per_frame
        )
        writer.add_int32(
            "qwen3_tts.stateful.initial_output_skip_frames", initial_skip_frames
        )
        # CustomVoice keeps acoustic codebooks greedy for timbre stability.
        # Base follows its official voice-clone generation configuration.
        code_predictor_sampling = (
            variant == "base" and generation.get("subtalker_dosample") is True
        )
        writer.add_int32(
            "qwen3_tts.sampling.code_predictor",
            int(code_predictor_sampling),
        )
        writer.add_int32("qwen3_tts.talker.hidden_size", talker["hidden_size"])
        writer.add_int32("qwen3_tts.text.hidden_size", talker["text_hidden_size"])
        writer.add_int32("qwen3_tts.text.vocab_size", talker["text_vocab_size"])
        writer.add_int32("qwen3_tts.tokenizer.vocab_size", len(vocab))
        writer.add_int32("qwen3_tts.codec.vocab_size", talker["vocab_size"])
        writer.add_int32(
            "qwen3_tts.code_predictor.vocab_size",
            talker["code_predictor_config"]["vocab_size"],
        )
        for key in (
            "tts_bos_token_id",
            "tts_eos_token_id",
            "tts_pad_token_id",
        ):
            writer.add_int32(f"qwen3_tts.{key}", config[key])
        for key in (
            "codec_bos_id",
            "codec_eos_token_id",
            "codec_think_id",
            "codec_nothink_id",
            "codec_pad_id",
            "codec_think_bos_id",
            "codec_think_eos_id",
        ):
            writer.add_int32(f"qwen3_tts.{key}", talker[key])
        writer.add_string(
            "qwen3_tts.codec_language_id",
            json.dumps(talker["codec_language_id"], ensure_ascii=False),
        )
        writer.add_string(
            "qwen3_tts.spk_id", json.dumps(talker["spk_id"], ensure_ascii=False)
        )
        writer.add_string(
            "qwen3_tts.spk_is_dialect",
            json.dumps(talker["spk_is_dialect"], ensure_ascii=False),
        )
        if variant == "base":
            speaker_encoder = config["speaker_encoder_config"]
            writer.add_int32(
                "qwen3_tts.speaker_encoder.embedding_dim",
                speaker_encoder["enc_dim"],
            )
            writer.add_int32(
                "qwen3_tts.speaker_encoder.sample_rate",
                speaker_encoder["sample_rate"],
            )
        writer.add_string(
            "qwen3_tts.generation_config",
            json.dumps(generation, ensure_ascii=False),
        )
        writer.add_string(
            "qwen3_tts.decode_padding_shapes",
            json.dumps(decode_padding_shapes, separators=(",", ":")),
        )
        writer.add_int32("vocab_size", len(vocab))
        writer.add_array("vocab", vocab)
        writer.add_int32("merges_size", len(merges))
        writer.add_array("merges", merges)

    display_size = {"0b6": "0.6B", "1b7": "1.7B"}[config["tts_model_size"]]
    display_variant = {
        "custom_voice": "CustomVoice",
        "base": "Base",
    }[variant]
    packer.write(
        f"Qwen3-TTS-12Hz-{display_size}-{display_variant}",
        "qwen3_tts",
        add_metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-TTS CustomVoice or Base HMM/PT files into GGUF"
    )
    parser.add_argument("-i", "--input", required=True, type=Path)
    parser.add_argument("-c", "--config-dir", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument(
        "--deployment-meta",
        type=Path,
        help="Optional PTQ meta.json containing Stateful Decoder export parameters",
    )
    parser.add_argument("--gguf-py", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(
        args.input,
        args.config_dir,
        args.output,
        args.gguf_py,
        args.deployment_meta,
    )


if __name__ == "__main__":
    main()
