from __future__ import annotations

import base64
import importlib.util
import io
import os
import re
import sys
import json
import time
import math
import regex
import uuid
import threading
from scipy.signal import get_window
import wave
import whisper
import torch
import soundfile as sf
from contextlib import nullcontext
import numpy as np
from torch import nn
from torch.nn import functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Iterator, List

import struct

from transformers import AutoTokenizer
from fastapi import Request
from fastapi.responses import Response
from loguru import logger
import inflect
from functools import partial

from librosa.filters import mel as librosa_mel_fn
from typing import List, Generator, Optional, Dict, Any

import tcim_lite as tcim

from hliepython.model_server.engines.base_engine import InferenceEngine
from hliepython.model_server.engines.protocol.audio_speech import (
    AudioSpeechBase64Response,
    AudioSpeechRequest,
    AudioSpeechResponse,
    DEFAULT_PROMPT_WAV_PATH,
)
from hliepython.model_server.utils.convert_gguf_to_hmm import parse_gguf_to_dict

from enum import Enum

TOKENS_FILE_PATH = os.path.join(os.path.dirname(__file__), "files/special_tokens.json")
mel_basis = {}
hann_window = {}
TTS_MAX_INPUT_CHARS = 45



class EmbeddingDtype(Enum):
    INT16 = 0    # 对应C++的INT16
    FLOAT16 = 1  # 对应C++的FLOAT16

def load_embedding_from_binary(file_path, offset=0):
    """
    从二进制数据加载Embedding权重（完全对齐C++逻辑，INT16保持原始类型，修复只读警告）
    :param binary_data: 二进制权重数据（bytes类型）
    :param offset: 偏移量（对应C++的offset参数）
    :return: torch.Tensor - 构建好的embedding权重张量
    """
    with open(file_path, "rb") as f:
        buffer = io.BytesIO(f.read())

    # 跳过偏移量（对应C++的f.seekg(offset, std::ios::beg)）
    buffer.seek(offset)

    # 1. 读取维度数（uint32_t）
    dim_count_bytes = buffer.read(4)
    if len(dim_count_bytes) != 4:
        raise RuntimeError("读取维度数失败：二进制数据长度不足")
    dim_count = struct.unpack('<I', dim_count_bytes)[0]  # 小端序解析（C++默认）

    # 2. 读取各维度值
    shape = []
    for i in range(dim_count):
        dim_bytes = buffer.read(4)
        if len(dim_bytes) != 4:
            raise RuntimeError(f"读取第{i}个维度值失败：二进制数据长度不足")
        shape.append(struct.unpack('<I', dim_bytes)[0])

    # 3. 读取数据类型标识
    dtype_bytes = buffer.read(4)
    if len(dtype_bytes) != 4:
        raise RuntimeError("读取数据类型标识失败：二进制数据长度不足")
    dtype_val = struct.unpack('<I', dtype_bytes)[0]

    # 校验支持的类型
    try:
        embedding_dtype = EmbeddingDtype(dtype_val)
    except ValueError:
        raise RuntimeError(f"不支持的数据类型：{dtype_val}（仅支持INT16/FLOAT16）")

    if embedding_dtype not in [EmbeddingDtype.INT16, EmbeddingDtype.FLOAT16]:
        raise RuntimeError(f"不支持的数据类型：{dtype_val}（仅支持INT16/FLOAT16）")

    # 4. 计算总元素数并读取权重数据
    elem_count = 1
    for d in shape:
        elem_count *= d
    data_size = elem_count * 2  # INT16/FLOAT16均为2字节

    weight_bytes = buffer.read(data_size)
    if len(weight_bytes) != data_size:
        raise RuntimeError(
            f"读取权重数据失败：期望{data_size}字节，实际{len(weight_bytes)}字节，剩余数据{buffer.read()}")

    # 5. 转换为torch张量（修复只读警告，INT16保持原始类型）
    if embedding_dtype == EmbeddingDtype.INT16:
        # 方案：使用torch.frombuffer + clone() 避免只读警告，性能更优
        weight_tensor = torch.frombuffer(weight_bytes, dtype=torch.int16).reshape(shape).clone()
    else:  # FLOAT16
        weight_tensor = torch.frombuffer(weight_bytes, dtype=torch.float16).reshape(shape).clone()

    print(f"权重数据类型: {weight_tensor.dtype}, 形状: {weight_tensor.shape}")
    return weight_tensor






def load_wav(wav, target_sr, min_sr=16000):
    """加载音频文件，将其转换为单声道，并在需要时重新采样。"""
    speech, sample_rate = sf.read(wav, always_2d=True, dtype="float32")
    speech = torch.from_numpy(speech.T).contiguous()
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert (
            sample_rate >= min_sr
        ), "wav sample rate {} must be greater than {}".format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr
        )(speech)
    return speech


def contains_chinese(text):
    """判断输入文本是否包含中文字符。"""
    chinese_char_pattern = re.compile(r"[\u4e00-\u9fff]+")
    return bool(chinese_char_pattern.search(text))


def replace_corner_mark(text):
    """将上标符号替换为中文口语表达。"""
    text = text.replace("²", "平方")
    text = text.replace("³", "立方")
    return text


def remove_bracket(text):
    """移除装饰性括号并规范化一些特殊分隔符。"""
    text = text.replace("（", "").replace("）", "")
    text = text.replace("【", "").replace("】", "")
    text = text.replace("`", "").replace("`", "")
    text = text.replace("——", " ")
    return text


def replace_blank(text: str):
    """仅在分隔相邻 ASCII 段时保留空格。"""
    out_str = []
    for i, c in enumerate(text):
        if c == " ":
            if (text[i + 1].isascii() and text[i + 1] != " ") and (
                text[i - 1].isascii() and text[i - 1] != " "
            ):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)


def is_only_punctuation(text):
    """判断字符串是否仅包含标点符号或符号。"""
    # 正则表达式：匹配仅由标点符号组成的字符串或空字符串。
    punctuation_pattern = r"^[\p{P}\p{S}]*$"
    return bool(regex.fullmatch(punctuation_pattern, text))


def spell_out_number(text: str, inflect_parser):
    """
    将文本中的阿拉伯数字转换为口语表达形式。

    Args:
        text: 输入文本，可能包含数字子串。
        inflect_parser: 提供 `number_to_words` 方法的转换对象。

    Returns:
        将数字子串替换为口语形式后的文本。
    """
    new_text = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st:i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return "".join(new_text)


def split_paragraph(
    text: str,
    tokenize,
    lang="zh",
    token_max_n=80,
    token_min_n=60,
    merge_len=20,
    comma_split=False,
):
    """
    将文本拆分为满足令牌长度约束的语句。
    拆分段落逻辑：
    1. 每个句子的最大长度为 token_max_n，最小长度为 token_min_n，如果最后一个句子长度小于 merge_len，则合并。
    2. 根据语言计算句子长度。
    3. 根据标点符号拆分句子。

    Args:
        text: 输入文本，用于拆分。
        tokenize: 用于非中文长度检查的分词函数。
        lang: 用于选择标点符号和长度规则的语言标签。
        token_max_n: 单个语句的最大令牌长度。
        token_min_n: 开始新语句前的最小令牌长度。
        merge_len: 合并短尾段的阈值。
        comma_split: 是否将逗号也作为拆分触发符。

    Returns:
        规范化后的语句列表。
    """

    def calc_utt_length(_text: str):
        """根据语言规则计算语句长度。"""
        if lang == "zh":
            return len(_text)
        else:
            return len(tokenize(_text))

    def should_merge(_text: str):
        """判断尾部语句是否足够短以进行合并。"""
        if lang == "zh":
            return len(_text) < merge_len
        else:
            return len(tokenize(_text)) < merge_len

    if lang == "zh":
        pounc = ["。", "？", "！", "；", "：", "、", ".", "?", "!", ";"]
    else:
        pounc = [".", "?", "!", ";", ":"]
    if comma_split:
        pounc.extend(["，", ","])

    if text[-1] not in pounc:
        if lang == "zh":
            text += "。"
        else:
            text += "."

    st = 0
    utts = []
    for i, c in enumerate(text):
        if c in pounc:
            if len(text[st:i]) > 0:
                utts.append(text[st:i] + c)
            if i + 1 < len(text) and text[i + 1] in ['"', "”"]:
                tmp = utts.pop(-1)
                utts.append(tmp + text[i + 1])
                st = i + 2
            else:
                st = i + 1

    final_utts = []
    cur_utt = ""
    for utt in utts:
        if (
            calc_utt_length(cur_utt + utt) > token_max_n
            and calc_utt_length(cur_utt) > token_min_n
        ):
            final_utts.append(cur_utt)
            cur_utt = ""
        cur_utt = cur_utt + utt
    if len(cur_utt) > 0:
        if should_merge(cur_utt) and len(final_utts) != 0:
            final_utts[-1] = final_utts[-1] + cur_utt
        else:
            final_utts.append(cur_utt)

    return final_utts


def enforce_text_chunk_limit(texts: List[str], max_chars: int = TTS_MAX_INPUT_CHARS) -> List[str]:
    """Ensure each TTS input chunk stays within the model's text-length limit."""
    limited_chunks: List[str] = []
    for text in texts:
        normalized = text.strip()
        if not normalized:
            continue
        if len(normalized) <= max_chars:
            limited_chunks.append(normalized)
            continue
        limited_chunks.extend(_split_text_by_char_limit(normalized, max_chars))
    return limited_chunks


def _split_text_by_char_limit(text: str, max_chars: int) -> List[str]:
    """Split long text into chunks, preferring punctuation boundaries."""
    punctuation = {"。", "？", "！", "；", "：", "，", "、", ".", "?", "!", ";", ":", ","}
    chunks: List[str] = []
    remaining = text.strip()

    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        split_idx = -1
        for idx in range(len(window) - 1, -1, -1):
            if window[idx] in punctuation:
                split_idx = idx + 1
                break

        if split_idx <= 0:
            split_idx = max_chars

        chunk = remaining[:split_idx].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_idx:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def random_sampling(weighted_scores, decoded_tokens, sampling):
    """Sample one token directly from the full probability distribution."""
    top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True).item()
    return top_ids


def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    """Sample one token from the truncated top-p and top-k candidate set."""
    prob, indices = [], []
    cum_prob = 0.0
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(
        descending=True, stable=True
    )
    for i in range(len(sorted_idx)):
        # sampling both top-p and numbers.
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob = torch.tensor(prob).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    top_ids = indices[prob.multinomial(1, replacement=True)].item()
    return top_ids

def get_module_input_names(module):
    """Return all input names declared by a TCIM module."""
    ipt_names = []
    input_num = module.get_num_inputs()
    for idx in range(input_num):
        input_name = module.get_input_name(idx)
        ipt_names.append(input_name)
    return ipt_names

def get_module_output_names(module):
    """Return all output names declared by a TCIM module."""
    opt_names = []
    output_num = module.get_num_outputs()
    for idx in range(output_num):
        output_name = module.get_output_name(idx)
        opt_names.append(output_name)
    return opt_names

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """Apply log compression to stabilize magnitude ranges."""
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    """Normalize spectrogram magnitudes with log-domain compression."""
    output = dynamic_range_compression_torch(magnitudes)
    return output

def load_special_tokens(file_path):
    """从 JSON 文件中加载分词器的特殊标记定义。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Special tokens file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        tokens = json.load(f)
    return tokens

def ras_sampling(
    weighted_scores,
    decoded_tokens,
    sampling,
    top_p=0.8,
    top_k=25,
    win_size=10,
    tau_r=0.1,
):
    """Apply repetition-aware sampling on top of nucleus sampling."""
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    rep_num = (
        (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids)
        .sum()
        .item()
    )
    if rep_num >= win_size * tau_r:
        top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
    return top_ids

def mel_spectrogram(
    y,
    n_fft=1920,
    num_mels=80,
    sampling_rate=24000,
    hop_size=480,
    win_size=1920,
    fmin=0,
    fmax=None,
    center=False,
):
    """Compute a log-mel spectrogram and cache reusable filter kernels."""
    if torch.min(y) < -1.0:
        logger.info("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        logger.info("max value is ", torch.max(y))

    global mel_basis, hann_window  # pylint: disable=global-statement
    mel_key = f"{sampling_rate}_{n_fft}_{num_mels}_{fmin}_{fmax}_{win_size}_{y.device}"
    window_key = f"{win_size}_{y.device}"
    if mel_key not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[mel_key] = torch.from_numpy(mel).float().to(y.device)
    if window_key not in hann_window:
        hann_window[window_key] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[window_key],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    spec = torch.matmul(mel_basis[mel_key], spec)
    spec = spectral_normalize_torch(spec)

    return spec


class CosyVoice3LM:
    def __init__(
        self,
        tts_config: Dict[str, Any],
    ):
        """Initialize the speech-token LLM and its runtime modules."""
        self.llm_input_size = tts_config["llm_input_size"]
        self.llm_output_size = tts_config["llm_output_size"]
        self.speech_token_size = tts_config["speech_token_size"]

        # token_embedding_state_dict = torch.load(
        #     os.path.join(f"{tts_config['model_path']}", "hmquant/quant_embedding.pt"),
        #     map_location="cpu",
        #     weights_only=True
        # )       # torch.Size([151936, 896]), dtype: torch.float16
        with open(tts_config['gguf_path'], "rb") as f:
            f.seek(tts_config["quant_embedding_pt"]["offset"])
            data = f.read(tts_config["quant_embedding_pt"]["size"])
        buffer = io.BytesIO(data)
        token_embedding_state_dict = torch.load(buffer, map_location="cpu", weights_only=True)
        logger.info(f"Loaded token embedding with shape: {token_embedding_state_dict['weight'].shape}, dtype: {token_embedding_state_dict['weight'].dtype}")
        self.token_embedding = nn.Embedding(
            token_embedding_state_dict["weight"].shape[0],
            token_embedding_state_dict["weight"].shape[1],
        )
        self.token_embedding.load_state_dict(token_embedding_state_dict)
        self.token_embedding = self.token_embedding.to(torch.float16)       # torch.Size([151936, 896]), dtype: torch.float16
        logger.info(f"Token embedding loaded with shape: {self.token_embedding.weight.shape}, dtype: {self.token_embedding.weight.dtype}")


        # speech_embedding_param = torch.load(
        #     os.path.join(f"{tts_config['model_path']}", "hmquant/llm_speech_embedding.pt"),
        #     map_location="cpu"
        # )
        with open(tts_config['gguf_path'], "rb") as f:
            f.seek(tts_config["llm_speech_embedding_pt"]["offset"])
            data = f.read(tts_config["llm_speech_embedding_pt"]["size"])
        buffer = io.BytesIO(data)
        speech_embedding_param = torch.load(buffer, map_location="cpu", weights_only=True)
        logger.info(f"Loaded token embedding with shape: {token_embedding_state_dict['weight'].shape}, dtype: {token_embedding_state_dict['weight'].dtype}")
        speech_embedding_state_dict = {"weight": speech_embedding_param}
        self.speech_embedding = nn.Embedding(6761, 896)
        self.speech_embedding.load_state_dict(speech_embedding_state_dict)
        self.speech_embedding = self.speech_embedding.to(torch.float16)     # torch.Size([6761, 896]), dtype: torch.float16
        logger.info(f"Speech embedding loaded with shape: {self.speech_embedding.weight.shape}, dtype: {self.speech_embedding.weight.dtype}")

        dev_manager_0 = tcim.runtime.DevManager(tts_config["device_ids"], "Xh2HalBackend")
        weight_manager_0 = tcim.runtime.WeightManager(dev_manager_0)
        llm_decoder_option = tcim.runtime.Option(weight_manager_0)

        llm_decoder_option.set_model_offset(
            tts_config["cosyvoice3_llm_decoder_hmm"]["offset"],
            tts_config["cosyvoice3_llm_decoder_hmm"]["size"],
        )

        self.llm_decoder = tcim.runtime.load(
            tts_config["gguf_path"],
            # f"{tts_config['model_path']}/cosyvoice3_llm_decoder.hmm",
            option=llm_decoder_option
        )
        self.llm_decoder_input_names = get_module_input_names(self.llm_decoder)
        self.llm_decoder_output_names = get_module_output_names(self.llm_decoder)


        dev_manager = tcim.runtime.DevManager(tts_config["device_ids"], "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)

        prefill_option = tcim.runtime.Option(weight_manager)
        decode_option = tcim.runtime.Option(weight_manager)

        prefill_option.set_model_offset(
            tts_config["cosyvoice3_llm_qwen2_prefill_hmm"]["offset"],
            tts_config["cosyvoice3_llm_qwen2_prefill_hmm"]["size"],
        )

        self.prefill = tcim.runtime.load(
            tts_config["gguf_path"],
            # f"{tts_config['model_path']}/cosyvoice3_llm_qwen2_prefill.hmm",
            option=prefill_option
        )
        self.prefill_input_names = get_module_input_names(self.prefill)

        dummy_tensor_names = []
        for input_name in self.prefill_input_names:
            if "model_layers" in input_name:
                dummy_tensor_names.append(input_name)
        decode_option.set_dummy_tensors(dummy_tensor_names)

        decode_option.set_model_offset(
            tts_config["cosyvoice3_llm_qwen2_decode_hmm"]["offset"],
            tts_config["cosyvoice3_llm_qwen2_decode_hmm"]["size"],
        )

        self.decode = tcim.runtime.load(
            tts_config["gguf_path"],
            # f"{tts_config['model_path']}/cosyvoice3_llm_qwen2_decode.hmm",
            option=decode_option
        )
        self.decode_input_names = get_module_input_names(self.decode)

        # Initialize cache inputs for decode model from prefill model
        for input_name in dummy_tensor_names:
            cache = self.prefill.get_dev_input(input_name)
            self.decode.set_input(input_name, cache)

        # Get model dimension information from input metadata
        self.prefill_length = self.prefill.get_input_info(
            self.prefill_input_names[0]
        ).shape[1]
        self.embedding_len = self.prefill.get_input_info(
            self.prefill_input_names[0]
        ).shape[2]
        self.context_max_length = self.decode.get_input_info(
            self.decode_input_names[3]
        ).shape[2]
        self.batch = self.decode.get_input_info(self.decode_input_names[0]).shape[0]

        # Set initial decode current length input
        current_length_input_1 = np.array([1]).astype("int32")
        decode_current_length_name = self.decode_input_names[2]
        self.decode.set_input(decode_current_length_name, current_length_input_1)

        self.input_sequence_length = 256
        self.context_length = 0

        # Load the start-of-sequence embedding.hmquant/
        # self.sos_emb = torch.load(
        #     os.path.join(f"{tts_config['model_path']}", "hmquant/llm_sos_embedding.pt"),
        #     map_location="cpu"
        # )       # torch.Size([1, 1, 896]), dtype: torch.float32
        with open(tts_config['gguf_path'], "rb") as f:
            f.seek(tts_config["llm_sos_embedding_pt"]["offset"])
            data = f.read(tts_config["llm_sos_embedding_pt"]["size"])
        buffer = io.BytesIO(data)
        self.sos_emb = torch.load(buffer, map_location="cpu", weights_only=True)
        logger.info(f"sos_emb shape: {self.sos_emb.shape}, dtype: {self.sos_emb.dtype}")

        # Load the task-id embedding.hmquant/
        with open(tts_config['gguf_path'], "rb") as f:
            f.seek(tts_config["llm_task_id_embedding_pt"]["offset"])
            data = f.read(tts_config["llm_task_id_embedding_pt"]["size"])
        buffer = io.BytesIO(data)
        self.task_id_emb = torch.load(buffer, map_location="cpu", weights_only=True)       # torch.Size([1, 1, 896]), dtype: torch.float32
        logger.info(f"task_id_emb shape: {self.task_id_emb.shape}, dtype: {self.task_id_emb.dtype}")

        # sampling method
        self.sampling = ras_sampling
        self.pad_token_id = 151645

        self.stop_token_ids = [tts_config["speech_token_size"] + i for i in range(200)]

    def sampling_ids(
        self,
        weighted_scores: torch.Tensor,
        decoded_tokens: List,
        sampling: int,
        ignore_eos: bool = True,
    ):
        """Sample the next token while optionally avoiding EOS for a while."""
        num_trials, max_trials = 0, 100
        speech_token_size = 6561
        while True:
            top_ids = self.sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (
                speech_token_size != top_ids
            ):  # (top_ids < self.speech_token_size):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError(
                    "sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!".format(
                        max_trials
                    )
                )
        return top_ids

    @torch.inference_mode()
    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        perf: Optional[Dict[str, Any]] = None,
    ) -> Generator[torch.Tensor, None, None]:
        """
        Generate speech tokens from text and prompt conditions.

        Args:
            text: Target text token sequence.
            text_len: Length of the target text sequence.
            prompt_text: Prompt text token sequence.
            prompt_text_len: Length of the prompt text sequence.
            prompt_speech_token: Prompt speech-token sequence.
            prompt_speech_token_len: Length of the prompt speech-token sequence.
            sampling: Sampling configuration value.
            max_token_text_ratio: Upper bound on generated speech-token length.
            min_token_text_ratio: Lower bound before EOS is allowed.
            uuid: Request identifier reserved for tracing.

        Returns:
            A generated speech-token sequence.
        """

        t_start = time.perf_counter()

        text = torch.concat([prompt_text, text], dim=1)
        text_len += prompt_text_len

        text = self.token_embedding(text)

        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(
                1, 0, self.llm_input_size, dtype=text.dtype
            )
        lm_input = torch.concat(
            [self.sos_emb, text, self.task_id_emb, prompt_speech_token_emb], dim=1
        )

        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        lm_input = lm_input.to(torch.float16)

        self.context_length = 0
        seq_length = lm_input.shape[1]
        out_tokens = []

        if perf is not None:
            perf["prefill_tokens"] = int(seq_length)

        # Validate input length against maximum context length
        if seq_length >= self.context_max_length:
            logger.error(
                f"Input sequence length ({seq_length}) exceeds maximum context length ({self.context_max_length}), please shorten your question!"
            )
            sys.exit(1)

        # Process prefill in chunks if input length exceeds prefill length
        input_name = self.prefill_input_names[0]
        valid_length_name = self.prefill_input_names[1]
        current_length_name = self.prefill_input_names[2]
        prefill_loop_round = math.ceil(seq_length / self.prefill_length)
        t_prefill0 = time.perf_counter()
        for round_idx in range(prefill_loop_round):
            valid_length = round_idx * self.prefill_length + self.context_length
            if round_idx == prefill_loop_round - 1:
                current_length = seq_length - round_idx * self.prefill_length
                inputs_embeds = lm_input[
                    :, round_idx * self.prefill_length : seq_length, :
                ]
            else:
                current_length = self.prefill_length
                inputs_embeds = lm_input[
                    :,
                    round_idx
                    * self.prefill_length : (round_idx + 1)
                    * self.prefill_length,
                    :,
                ]

            effective_length = inputs_embeds.shape[1]
            pad_input_ids = torch.zeros(
                (self.prefill_length - effective_length), dtype=torch.long
            )
            pad_input_ids.fill_(self.pad_token_id)
            pad_input_ids = pad_input_ids.unsqueeze(0)
            pad_embeds = self.token_embedding(pad_input_ids)
            input_data = torch.cat([inputs_embeds, pad_embeds], dim=1).detach()

            # Prepare length parameters for prefill input
            valid_length_data = np.array([valid_length]).astype("int32")
            current_length_data = np.array([current_length]).astype("int32")

            self.prefill.set_input(input_name, input_data.numpy())
            self.prefill.set_input(valid_length_name, valid_length_data)
            self.prefill.set_input(current_length_name, current_length_data)

            self.prefill.run()
            self.prefill.sync()
        t_prefill1 = time.perf_counter()

        prefill_opt = (
            self.prefill.get_dev_output(self.prefill.get_output_name(0))
            .to_host()
            .numpy()
            .squeeze(0)
        )

        self.llm_decoder.set_input(self.llm_decoder_input_names[0], prefill_opt)
        self.llm_decoder.run()
        self.llm_decoder.sync()
        prefill_logp = (
            self.llm_decoder.get_dev_output(self.llm_decoder_output_names[0])
            .to_host()
            .numpy()
        )
        prefill_logp = torch.from_numpy(prefill_logp)

        top_ids = self.sampling_ids(
            prefill_logp.squeeze(dim=0),
            out_tokens,
            sampling,
            ignore_eos=True if 0 < min_len else False,
        )
        t_first_token = time.perf_counter()
        if top_ids in self.stop_token_ids:
            if perf is not None:
                perf["prefill_s"] = float(t_prefill1 - t_prefill0)
                perf["ttft_s"] = float(t_first_token - t_start)
                perf["decode_s"] = 0.0
                perf["decode_tokens"] = 0
                perf["llm_total_s"] = float(t_first_token - t_start)
            return out_tokens
        out_tokens.append(top_ids)
        self.context_length += seq_length

        input_name = self.decode_input_names[0]
        valid_length_name = self.decode_input_names[1]

        t_decode0 = time.perf_counter()
        decode_token_cnt = 0
        for i in range(1, max_len):
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1).detach()

            self.decode.set_input(input_name, lm_input.numpy())
            self.decode.set_input(
                valid_length_name, np.array([self.context_length]).astype("int32")
            )
            self.decode.run()
            self.decode.sync()
            decode_dev_opt = self.decode.get_dev_output(self.decode.get_output_name(0))
            llm_decoder_dev_ipt = decode_dev_opt.select_batch([0])

            self.llm_decoder.set_input(
                self.llm_decoder_input_names[0], llm_decoder_dev_ipt
            )
            self.llm_decoder.run()
            self.llm_decoder.sync()
            logp = (
                self.llm_decoder.get_dev_output(self.llm_decoder_output_names[0])
                .to_host()
                .numpy()
            )

            logp = torch.from_numpy(logp)
            top_ids = self.sampling_ids(
                logp.squeeze(dim=0),
                out_tokens,
                sampling,
                ignore_eos=True if i < min_len else False,
            )
            self.context_length += 1
            if top_ids in self.stop_token_ids:
                break
            out_tokens.append(top_ids)
            decode_token_cnt += 1
        t_decode1 = time.perf_counter()

        if perf is not None:
            perf["prefill_s"] = float(t_prefill1 - t_prefill0)
            perf["ttft_s"] = float(t_first_token - t_start)
            perf["decode_s"] = float(t_decode1 - t_decode0)
            perf["decode_tokens"] = int(decode_token_cnt)
            perf["llm_total_s"] = float(t_decode1 - t_start)

        return out_tokens



class CausalMaskedDiffWithDiT:
    def __init__(
        self,
        tts_config,
        input_size: int = 512,
        output_size: int = 80,
        output_type: str = "mel",
        vocab_size: int = 4096,
        input_frame_rate: int = 50,
        only_mask_loss: bool = True,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
    ):
        """Initialize the flow-based acoustic decoder runtime modules."""
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate

        # input_embedding_state_dict = torch.load(
        #     os.path.join(f"{tts_config['model_path']}", "hmquant/flow_input_embedding.pt"), map_location="cpu", weights_only=True
        # )       # torch.Size([6561, 80]), dtype: torch.float32
        with open(tts_config['gguf_path'], "rb") as f:
            f.seek(tts_config["flow_input_embedding_pt"]["offset"])
            data = f.read(tts_config["flow_input_embedding_pt"]["size"])
        buffer = io.BytesIO(data)
        input_embedding_state_dict = torch.load(buffer, map_location="cpu", weights_only=True)       # torch.Size([1, 1, 896]), dtype: torch.float32
        logger.info(f"input_embedding_state_dict shape: {input_embedding_state_dict.shape}, dtype: {input_embedding_state_dict.dtype}")
        input_embedding = nn.Embedding(
            input_embedding_state_dict.shape[0],
            input_embedding_state_dict.shape[1],
        )
        input_embedding_state_dict_ = {"weight": input_embedding_state_dict}
        input_embedding.load_state_dict(input_embedding_state_dict_)
        self.input_embedding = input_embedding.to("cpu")        # torch.Size([6561, 80]), dtype: torch.float32
        logger.info(f"Input embedding loaded with shape: {self.input_embedding.weight.shape}, dtype: {self.input_embedding.weight.dtype}")

        dev_manager = tcim.runtime.DevManager(tts_config["device_ids"], "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)

        spk_embed_affine_layer_option = tcim.runtime.Option(weight_manager)
        pre_lookahead_layer_option = tcim.runtime.Option(weight_manager)
        flow_decoder_option = tcim.runtime.Option(weight_manager)

        spk_embed_affine_layer_option.set_model_offset(
            tts_config["cosyvoice3_flow_spk_hmm"]["offset"],
            tts_config["cosyvoice3_flow_spk_hmm"]["size"],
        )
        pre_lookahead_layer_option.set_model_offset(
            tts_config["cosyvoice3_flow_encoder_hmm"]["offset"],
            tts_config["cosyvoice3_flow_encoder_hmm"]["size"],
        )
        flow_decoder_option.set_model_offset(
            tts_config["cosyvoice3_flow_decoder_hmm"]["offset"],
            tts_config["cosyvoice3_flow_decoder_hmm"]["size"],
        )

        self.spk_embed_affine_layer = tcim.runtime.load(
            tts_config["gguf_path"],
            # f"{tts_config['model_path']}/cosyvoice3_flow_spk.hmm",
            option=spk_embed_affine_layer_option
        )
        self.spk_embed_affine_layer_input_names = get_module_input_names(
            self.spk_embed_affine_layer
        )
        self.spk_embed_affine_layer_output_names = get_module_output_names(
            self.spk_embed_affine_layer
        )

        self.pre_lookahead_layer = tcim.runtime.load(
            tts_config["gguf_path"],
            # f"{tts_config['model_path']}/cosyvoice3_flow_encoder.hmm",
            option=pre_lookahead_layer_option
        )
        self.pre_lookahead_layer_input_names = get_module_input_names(
            self.pre_lookahead_layer
        )
        self.pre_lookahead_layer_output_names = get_module_output_names(
            self.pre_lookahead_layer
        )

        self.flow_decoder = tcim.runtime.load(
            tts_config["gguf_path"],
            # f"{tts_config['model_path']}/cosyvoice3_flow_decoder.hmm",
            option=flow_decoder_option
        )
        self.flow_decoder_input_names = get_module_input_names(self.flow_decoder)
        self.flow_decoder_output_names = get_module_output_names(self.flow_decoder)

        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

    @torch.inference_mode()
    def inference(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
    ):
        """
        Convert speech tokens into mel spectrogram features.

        Args:
            token: Target speech-token sequence.
            token_len: Length of the target speech-token sequence.
            prompt_token: Prompt speech-token sequence.
            prompt_token_len: Length of the prompt speech-token sequence.
            prompt_feat: Prompt acoustic feature sequence.
            prompt_feat_len: Length of the prompt acoustic feature sequence.
            embedding: Prompt speech embedding vector.

        Returns:
            A tuple of generated mel features and their valid length.
        """
        # This implementation only supports batch size 1.
        assert token.shape[0] == 1

        token, token_len = (
            torch.concat([prompt_token, token], dim=1),
            prompt_token_len + token_len,
        )
        token = self.input_embedding(token)
        token = F.pad(token, (0, 0, 0, 1024 - token.shape[1]), value=0)
        token = token.to(torch.float16).numpy()

        self.pre_lookahead_layer.set_input(
            self.pre_lookahead_layer_input_names[0], token
        )
        self.pre_lookahead_layer.run()
        self.pre_lookahead_layer.sync()
        h = (
            self.pre_lookahead_layer.get_dev_output(
                self.pre_lookahead_layer_output_names[0]
            )
            .to_host()
            .numpy()
        )
        h = torch.from_numpy(h)
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)

        embedding = F.normalize(embedding, dim=1)
        self.spk_embed_affine_layer.set_input(
            self.spk_embed_affine_layer_input_names[0], embedding.numpy()
        )
        self.spk_embed_affine_layer.run()
        self.spk_embed_affine_layer.sync()
        embedding = (
            self.spk_embed_affine_layer.get_dev_output(
                self.spk_embed_affine_layer_output_names[0]
            )
            .to_host()
            .numpy()
        )
        embedding = torch.from_numpy(embedding)

        # `mel_len1` is the prompt length and `mel_len2` is the newly synthesized part.
        mel_len1, mel_len2 = prompt_feat.shape[1], token_len * 2 - prompt_feat.shape[1]
        # Build conditioning tensors for the flow decoder.
        conds = torch.zeros([1, 2048, 80]).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]), 2048)).to(h)
        mask = mask.unsqueeze(1)

        mu = h.transpose(1, 2).contiguous()
        rand_noise = torch.randn([1, 80, 50 * 300])
        x = rand_noise[:, :, : mu.size(2)].to(mu.dtype) * 1.0
        t_span = torch.linspace(0, 1, 10 + 1, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)
        x_in = torch.zeros([2, 80, x.size(2)], dtype=x.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], dtype=x.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], dtype=x.dtype)
        t_in = torch.zeros([2], dtype=x.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], dtype=x.dtype)
        spks_in = torch.zeros([2, 80], dtype=x.dtype)

        sol = []
        inference_cfg_rate = 0.7
        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = embedding
            cond_in[0] = conds

            self.flow_decoder.set_input(self.flow_decoder_input_names[0], x_in.numpy())
            self.flow_decoder.set_input(
                self.flow_decoder_input_names[1], mask_in.numpy()
            )
            self.flow_decoder.set_input(self.flow_decoder_input_names[2], mu_in.numpy())
            self.flow_decoder.set_input(self.flow_decoder_input_names[3], t_in.numpy())
            self.flow_decoder.set_input(
                self.flow_decoder_input_names[4], spks_in.numpy()
            )
            self.flow_decoder.set_input(
                self.flow_decoder_input_names[5], cond_in.numpy()
            )
            self.flow_decoder.run()
            self.flow_decoder.sync()
            dphi_dt = (
                self.flow_decoder.get_dev_output(self.flow_decoder_output_names[0])
                .to_host()
                .numpy()
            )
            dphi_dt = torch.from_numpy(dphi_dt)

            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = (
                1.0 + inference_cfg_rate
            ) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        feat = sol[-1].float()
        feat = feat[:, :, mel_len1 : mel_len1 + mel_len2]

        return feat, mel_len2


class CosyVoice3Tokenizer:
    def __init__(self, token_path, skip_special_tokens=True):
        """Build the tokenizer and register project-specific special tokens."""
        # NOTE: non-chat model, all these special tokens keep randomly initialized.
        self.special_tokens = load_special_tokens(TOKENS_FILE_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(token_path)
        self.tokenizer.add_special_tokens(self.special_tokens)
        self.skip_special_tokens = skip_special_tokens

    def encode(self, text, **kwargs):
        """Encode a text string into token ids."""
        tokens = self.tokenizer([text], return_tensors="pt")
        tokens = tokens["input_ids"][0].cpu().tolist()
        return tokens

    def decode(self, tokens):
        """Decode token ids back into text."""
        tokens = torch.tensor(tokens, dtype=torch.int64)
        text = self.tokenizer.batch_decode(
            [tokens], skip_special_tokens=self.skip_special_tokens
        )[0]
        return text

class CosyVoice3Model:
    def __init__(self, tts_config: dict[str, Any]):
        self.device = "cpu"
        self.sampling_rate = tts_config["sampling_rate"]

        self.llm = CosyVoice3LM(tts_config)
        self.flow = CausalMaskedDiffWithDiT(tts_config)

        dev_manager = tcim.runtime.DevManager(tts_config["device_ids"], "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)

        hift_part1_option = tcim.runtime.Option(weight_manager)
        hift_part2_option = tcim.runtime.Option(weight_manager)

        hift_part1_option.set_model_offset(
            tts_config["cosyvoice3_hift_part1_hmm"]["offset"],
            tts_config["cosyvoice3_hift_part1_hmm"]["size"],
        )
        hift_part2_option.set_model_offset(
            tts_config["cosyvoice3_hift_part2_hmm"]["offset"],
            tts_config["cosyvoice3_hift_part2_hmm"]["size"],
        )

        self.hift_part1 = tcim.runtime.load(
            tts_config["gguf_path"],
            # f"{tts_config['model_path']}/cosyvoice3_hift_part1.hmm",
            option=hift_part1_option,
        )
        self.hift_part1_input_names = get_module_input_names(self.hift_part1)
        self.hift_part1_output_names = get_module_output_names(self.hift_part1)

        dummy_input_name = self.hift_part1_input_names[0]
        hift_part2_option.set_dummy_tensors([dummy_input_name])

        self.hift_part2 = tcim.runtime.load(
            tts_config["gguf_path"],
            # f"{tts_config['model_path']}/cosyvoice3_hift_part2.hmm",
            option=hift_part2_option,
        )

        dummy_input = self.hift_part1.get_dev_input(dummy_input_name)
        self.hift_part2.set_input(dummy_input_name, dummy_input)
        self.hift_part2_input_names = get_module_input_names(self.hift_part2)
        self.hift_part2_output_names = get_module_output_names(self.hift_part2)

        self.token_hop_len = 25
        self.llm_context = nullcontext()
        self.lock = threading.Lock()
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}
        self.silent_tokens = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]

        self._perf_by_uuid: Dict[str, Dict[str, Any]] = {}
        self.perf_history: List[Dict[str, Any]] = []


    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        cur_silent_token_num, max_silent_token_num = 0, 5
        llm_perf: Dict[str, Any] = {}
        token_generator = self.llm.inference(
            text=text,
            text_len=torch.tensor([text.shape[1]], dtype=torch.int32),
            prompt_text=prompt_text,
            prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32),
            prompt_speech_token=llm_prompt_speech_token,
            prompt_speech_token_len=torch.tensor(
                [llm_prompt_speech_token.shape[1]], dtype=torch.int32
            ),
            perf=llm_perf,
        )

        for i in token_generator:
            if i in self.silent_tokens:
                cur_silent_token_num += 1
                if cur_silent_token_num > max_silent_token_num:
                    logger.debug(f"skip current silent_token {cur_silent_token_num}")
                    continue
            else:
                cur_silent_token_num = 0
            self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True
        with self.lock:
            self._perf_by_uuid[uuid] = {"llm": llm_perf}

    def token2wav(self, token, prompt_token, prompt_feat, embedding, speed=1.0):
        tts_mel, mel_len2 = self.flow.inference(
            token=token.to(dtype=torch.int32),
            token_len=torch.tensor([token.shape[1]], dtype=torch.int32),
            prompt_token=prompt_token,
            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32),
            prompt_feat=prompt_feat,
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32),
            embedding=embedding,
        )

        tts_mel = tts_mel[:, :, 0:]
        needed = 1024 - tts_mel.size(2)
        if needed > 0:
            tts_mel = F.pad(tts_mel, (0, needed), value=0)
        else:
            tts_mel = tts_mel[:, :, :1024]

        if speed != 1.0:
            # Resample the mel frames to approximate the requested speed.
            tts_mel = F.interpolate(
                tts_mel, size=int(tts_mel.shape[2] / speed), mode="linear"
            )

        speech_feat = tts_mel.to(torch.float16).numpy()
        self.hift_part1.set_input(self.hift_part1_input_names[0], speech_feat)
        self.hift_part1.run()
        self.hift_part1.sync()
        hift_part1_opt = (
            self.hift_part1.get_dev_output(self.hift_part1_output_names[0])
            .to_host()
            .numpy()
        )
        stft_ipt = torch.from_numpy(hift_part1_opt)

        stft_window = torch.from_numpy(
            get_window("hann", 16, fftbins=True).astype(np.float32)
        )
        stft_opt = torch.stft(
            stft_ipt,
            16,
            4,
            16,
            window=stft_window,
            onesided=True,
            center=False,
            return_complex=False,
        ).permute(0, 2, 1, 3)

        self.hift_part2.set_input(
            self.hift_part2_input_names[0], stft_opt.to(torch.float16).to("cpu").numpy()
        )
        # self.hift_part2.set_input(self.hift_part2_input_names[1], speech_feat)
        self.hift_part2.run()
        self.hift_part2.sync()
        hift_part2_opt = (
            self.hift_part2.get_dev_output(self.hift_part2_output_names[0])
            .to_host()
            .numpy()
        )
        tts_speech = torch.from_numpy(hift_part2_opt)
        tts_speech = tts_speech[:, : 480 * mel_len2]

        return tts_speech

    def tts(
        self,
        text=torch.zeros(1, 0, dtype=torch.int32),  # tts tokens
        flow_embedding=torch.zeros(0, 192),  # speech embedding - from prompt wav
        llm_embedding=torch.zeros(0, 192),  # speech embedding - from prompt wav
        prompt_text=torch.zeros(1, 0, dtype=torch.int32),  # prompt text tokens
        llm_prompt_speech_token=torch.zeros(
            1, 0, dtype=torch.int32
        ),  # prompt wav tokens
        flow_prompt_speech_token=torch.zeros(
            1, 0, dtype=torch.int32
        ),  # prompt wav tokens
        prompt_speech_feat=torch.zeros(1, 0, 80),  # speech feature - from prompt wav
        source_speech_token=torch.zeros(
            1, 0, dtype=torch.int32
        ),  # speech feature length - from prompt wav
        speed=1.0,
        **kwargs,
    ):
        """Run the full text-to-speech pipeline for one request."""
        e2e_t0 = time.perf_counter()
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = (
                [],
                False,
            )
            self.hift_cache_dict[this_uuid] = None
            self._perf_by_uuid[this_uuid] = {}

        if source_speech_token.shape[1] == 0:
            llm_t0 = time.perf_counter()
            p = threading.Thread(
                target=self.llm_job,
                args=(
                    text,
                    prompt_text,
                    llm_prompt_speech_token,
                    llm_embedding,
                    this_uuid,
                ),
            )
        else:
            assert False, "source_speech_token.shape[1] != 0"
        p.start()

        # deal with all tokens
        p.join()
        llm_t1 = time.perf_counter()
        this_tts_speech_token = torch.tensor(
            self.tts_speech_token_dict[this_uuid]
        ).unsqueeze(dim=0)
        tts_t0 = time.perf_counter()
        this_tts_speech = self.token2wav(
            token=this_tts_speech_token,
            prompt_token=flow_prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            embedding=flow_embedding,
            speed=speed,
        )
        tts_t1 = time.perf_counter()
        result_speech = this_tts_speech.to(torch.float32).cpu()
        tts_wall_s = float(time.perf_counter() - e2e_t0)

        with self.lock:
            perf = self._perf_by_uuid.get(this_uuid, {}) or {}
            llm_perf = perf.get("llm", {}) or {}
            # wall time for llm thread (includes prefill+decode+host overhead)
            llm_perf["llm_stage_wall_s"] = float(llm_t1 - llm_t0)
            perf["llm"] = llm_perf

            tts_compute_s = float(tts_t1 - tts_t0)
            audio_s = (
                float(this_tts_speech.shape[1]) / float(self.sampling_rate)
                if this_tts_speech.ndim == 2
                else 0.0
            )
            perf["tts"] = {
                "tts_total_s": tts_compute_s,
                "tts_wall_s": tts_wall_s,
                "audio_s": audio_s,
                "rtf": (tts_wall_s / audio_s) if audio_s > 0 else None,
            }
            perf["e2e_total_s"] = tts_wall_s
            self.perf_history.append(perf)

            logger.info(f"tts, uuid: {this_uuid}, release resources.")
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
            self._perf_by_uuid.pop(this_uuid, None)

        return result_speech

    def flush_perf_history(self) -> List[Dict[str, Any]]:
        """Return and clear accumulated perf stats."""
        with self.lock:
            out = list(self.perf_history)
            self.perf_history.clear()
            return out


class CosyVoiceFrontEnd:
    def __init__(self, tts_config: dict[str, Any]):
        # 这里可以添加实际的初始化代码，例如加载分词器等
        self.tokenizer = CosyVoice3Tokenizer(
            token_path=tts_config["model_dir"],
            skip_special_tokens=True,
        )

        self.feat_extractor = mel_spectrogram
        self.device = "cpu"

        logger.info(f"device_ids: {tts_config['device_ids']}]")
        dev_manager = tcim.runtime.DevManager(tts_config["device_ids"], "Xh2HalBackend")
        weight_manager = tcim.runtime.WeightManager(dev_manager)
        campplus_option = tcim.runtime.Option(weight_manager)
        speech_tokenizer_option = tcim.runtime.Option(weight_manager)

        campplus_option.set_model_offset(
            tts_config["cosyvoice3_campplus_hmm"]["offset"],
            tts_config["cosyvoice3_campplus_hmm"]["size"],
        )
        speech_tokenizer_option.set_model_offset(
            tts_config["cosyvoice3_speech_tokenizer_hmm"]["offset"],
            tts_config["cosyvoice3_speech_tokenizer_hmm"]["size"],
        )

        self.campplus = tcim.runtime.load(
            tts_config["gguf_path"],
            # f"{tts_config['model_path']}/cosyvoice3_campplus.hmm",
            option=campplus_option
        )
        self.speech_tokenizer = tcim.runtime.load(
            tts_config["gguf_path"],
            # f"{tts_config['model_path']}/cosyvoice3_speech_tokenizer.hmm",
            option=speech_tokenizer_option
        )

        self.campplus_input_names = get_module_input_names(self.campplus)
        self.campplus_output_names = get_module_output_names(self.campplus)
        self.speech_tokenizer_input_names = get_module_input_names(self.speech_tokenizer)
        self.speech_tokenizer_output_names = get_module_output_names(self.speech_tokenizer)

        if tts_config["spk2info"] and os.path.exists(tts_config["spk2info"]):
            logger.info(f"load spk2info from {tts_config['spk2info']}")
            self.spk2info = torch.load(tts_config["spk2info"], map_location="cpu", weights_only=True)
        else:
            self.spk2info = {}

        self.allowed_special = "all"
        self.inflect_parser = inflect.engine()

        try:
            from wetext import Normalizer as ZhNormalizer
            from wetext import Normalizer as EnNormalizer

            self.zh_tn_model = ZhNormalizer(remove_erhua=False)
            self.en_tn_model = EnNormalizer()
            self.text_frontend = "wetext"
        except Exception as e:
            self.text_frontend = ""
            logger.warning(f"No frontend is avaliable, err: {e}")


    def _extract_text_token(self, text):
        """Tokenize text and return both token ids and sequence length."""
        text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        text_token = torch.tensor([text_token], dtype=torch.int32)
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32)
        return text_token, text_token_len

    # 语音 → token ID
    def _extract_speech_token(self, prompt_wav):
        """Extract prompt speech tokens with the speech tokenizer runtime."""
        # 加载音频prompt audio，并重采样到16 kHz
        speech = load_wav(prompt_wav, 16000)
        assert (
            speech.shape[1] / 16000 <= 30
        ), "do not support extract speech token for audio longer than 30s"

        # 提取梅尔频谱图特征，得到一个形状为[1, 128, 348]的张量，其中128是梅尔频率bin的数量，348是时间帧的数量
        # feat = mel_spectrogram(
        #     speech,
        #     n_fft=400,
        #     num_mels=128,
        #     sampling_rate=16000,
        #     hop_size=160,
        #     win_size=400,
        #     fmin=0,
        #     fmax=8000,
        #     center=True,
        # )  # torch.Size([1, 128, T]).

        feat = whisper.log_mel_spectrogram(
            speech, n_mels=128
        )  # torch.Size([1, 128, 348]).
        feat = feat.half()
        feat_len = feat.shape[2]

        padded_input = torch.zeros((1, 128, 3000), dtype=torch.float16)
        padded_input[:, :, :feat_len] = feat

        mask_shape = (1, 20, 750, 750)
        mask = torch.full(
            mask_shape, torch.finfo(torch.float16).min, dtype=torch.float16
        )
        mask[:, :, :, : feat_len // 4] = 0
        mask1 = torch.zeros((1, 750, 1280), dtype=torch.float16)
        mask1[:, 0 : feat_len // 4, :] = 1.0

        self.speech_tokenizer.set_input(self.speech_tokenizer_input_names[0], padded_input.numpy())
        self.speech_tokenizer.set_input(self.speech_tokenizer_input_names[1], mask.numpy())
        self.speech_tokenizer.set_input(self.speech_tokenizer_input_names[2], mask1.numpy())
        self.speech_tokenizer.run()
        self.speech_tokenizer.sync()
        speech_token = (
            self.speech_tokenizer.get_dev_output(self.speech_tokenizer_output_names[0])
            .to_host()
            .numpy()
        )
        speech_token = torch.from_numpy(speech_token)

        # 截取有效的语音token部分
        speech_token = speech_token[:, : feat_len // 4]
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32)
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, prompt_wav):
        speech = load_wav(prompt_wav, 16000)

        feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)  # torch.Size([346, 80])

        T_fixed = 1000
        T = feat.shape[0]
        if T < T_fixed:
            feat = F.pad(feat, (0, 0, 0, T_fixed - T))
        else:
            feat = feat[:T_fixed]
        feat = feat.half()

        campplus_ipt = feat.unsqueeze(0).numpy()
        self.campplus.set_input(self.campplus_input_names[0], campplus_ipt)
        self.campplus.run()
        self.campplus.sync()

        embedding = (
            self.campplus.get_dev_output(self.campplus_output_names[0])
            .to_host()
            .numpy()
        )
        embedding = torch.from_numpy(embedding)
        return embedding

    # 语音 → 梅尔谱
    def _extract_speech_feat(self, prompt_wav):
        """Extract prompt mel features and return their valid length."""
        # 加载音频prompt audio，并重采样到24 kHz
        speech = load_wav(prompt_wav, 24000)
        # 提取梅尔频谱图特征，得到一个形状为[1, 80, T]的张量，其中80是梅尔频率bin的数量，T是时间帧的数量
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1)
        # 归一化、补齐长度到3000帧，并转换为半精度浮点数，得到一个形状为[3000, 80]的张量，其中前T帧是有效的梅尔特征，后面是全零的无效部分
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32)
        return speech_feat, speech_feat_len


    def text_normalize(self, text, split=True, text_frontend=True):
        """Normalize text and optionally split it into synthesis segments."""
        if isinstance(text, Generator):
            logger.warning("get tts_text generator, will skip text_normalize!")
            return [text]
        # NOTE skip text_frontend when ssml symbol in text
        if "<|" in text and "|>" in text:
            text_frontend = False
        if text_frontend is False or text == "":
            return [text] if split is True else text
        text = text.strip()

        if contains_chinese(text):
            if self.text_frontend == "wetext":
                text = self.zh_tn_model.normalize(text)
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "。")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            text = re.sub(r"[，,、]+$", "。", text)
            texts = list(
                split_paragraph(
                    text,
                    partial(
                        self.tokenizer.encode, allowed_special=self.allowed_special
                    ),
                    "zh",
                    token_max_n=80,
                    token_min_n=60,
                    merge_len=20,
                    comma_split=False,
                )
            )
        else:
            if self.text_frontend == "wetext":
                text = self.en_tn_model.normalize(text)
            text = spell_out_number(text, self.inflect_parser)
            texts = list(
                split_paragraph(
                    text,
                    partial(
                        self.tokenizer.encode, allowed_special=self.allowed_special
                    ),
                    "en",
                    token_max_n=80,
                    token_min_n=60,
                    merge_len=20,
                    comma_split=False,
                )
            )
        texts = [i for i in texts if not is_only_punctuation(i)]
        texts = enforce_text_chunk_limit(texts, max_chars=TTS_MAX_INPUT_CHARS)
        texts = [i for i in texts if i and not is_only_punctuation(i)]
        return texts if split is True else text

    def frontend_instruct2(
        self, tts_text, instruct_text, prompt_wav, resample_rate, zero_shot_spk_id 
    ):
        """Build zero-shot instruction-following frontend inputs."""
        model_input = self.frontend_zero_shot(
            tts_text, instruct_text, prompt_wav, resample_rate, zero_shot_spk_id
        )
        del model_input["llm_prompt_speech_token"]
        del model_input["llm_prompt_speech_token_len"]
        return model_input

    def frontend_zero_shot(
        self, tts_text, prompt_text, prompt_wav, resample_rate, zero_shot_spk_id
    ):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)

        if zero_shot_spk_id == "":
            prompt_text_token, prompt_text_token_len = self._extract_text_token(
                prompt_text
            )

            speech_feat, speech_feat_len = self._extract_speech_feat(prompt_wav)
            speech_token, speech_token_len = self._extract_speech_token(prompt_wav)

            if resample_rate == 24000:
                # cosyvoice2, force speech_feat % speech_token = 2
                token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
                speech_feat, speech_feat_len[:] = (
                    speech_feat[:, : 2 * token_len],
                    2 * token_len,
                )
                speech_token, speech_token_len[:] = (
                    speech_token[:, :token_len],
                    token_len,
                )
            embedding = self._extract_spk_embedding(prompt_wav)
            model_input = {
                "prompt_text": prompt_text_token,
                "prompt_text_len": prompt_text_token_len,
                "llm_prompt_speech_token": speech_token,
                "llm_prompt_speech_token_len": speech_token_len,
                "flow_prompt_speech_token": speech_token,
                "flow_prompt_speech_token_len": speech_token_len,
                "prompt_speech_feat": speech_feat,
                "prompt_speech_feat_len": speech_feat_len,
                "llm_embedding": embedding,
                "flow_embedding": embedding,
            }
        else:
            logger.info(f"Reuse the features of {zero_shot_spk_id}.")
            model_input = {**self.spk2info[zero_shot_spk_id]}

        model_input["text"] = tts_text_token
        model_input["text_len"] = tts_text_token_len
        return model_input
class CosyVoice3:
    def __init__(self, tts_config: dict[str, Any]):
        self.tts_config = tts_config
        self.sample_rate = 24000  # 默认采样率，实际值由模型决定

        self.frontend = CosyVoiceFrontEnd(self.tts_config)
        self.model = CosyVoice3Model(self.tts_config)

        logger.info(f"tts_config: {tts_config}")

    def inference_instruct(
        self,
        tts_text: str,
        instruct_text: str,
        prompt_wav: str,
        zero_shot_spk_id: str = "",
        speed: float = 1.0,
        text_frontend: bool = True,
    ) -> Iterator[Any]:
        instruct_text = "You are a helpful assistant. <|endofprompt|>"

        logger.info(
            f"Running CosyVoice3 inference with instruct2 mode. "
            f"tts_text={tts_text}, instruct_text={instruct_text}, "
            f"prompt_wav={prompt_wav}, zero_shot_spk_id={zero_shot_spk_id}, "
            f"speed={speed}, text_frontend={text_frontend}"
        )

        for i in tqdm(
            self.frontend.text_normalize(
                tts_text, split=True, text_frontend=text_frontend
        )):
            model_input = self.frontend.frontend_instruct2(
                i, instruct_text, prompt_wav, self.sample_rate, zero_shot_spk_id
            )
            logger.info("synthesis text {}".format(i))
            tts_speech = self.model.tts(**model_input, speed=speed)
            logger.success("generate tts speech successfully.")
            yield tts_speech



class TTSEngine(InferenceEngine):
    """文本到语音推理引擎（集成 CosyVoice3 demo）。"""

    def __init__(self, instance_config: Dict[str, Any]):
        self.additional_config = instance_config.get(
            "additional_config",
            instance_config.get("extra_config", {}),
        ) or {}
        self.extra_config = self.additional_config
        self.device_ids = instance_config.get(
            "device_ids",
            self.additional_config.get("device_ids", []),
        )
        self._model_lock = threading.Lock()
        self._demo_module: ModuleType | None = None
        self.tts_model: Any = None

        self.model_path = instance_config.get("model_path", "")
        self.model_config = parse_gguf_to_dict(
            self.model_path,
            save_config_files=True,
        )

        self.tts_model = None

        self.tts_config = {}

        logger.info(f"model_config: {self.model_config}")

        self.tts_config["gguf_path"] = self.model_config["model_path"]
        self.tts_config["model_dir"] = self.model_config.get("model_dir", "")
        self.tts_config["model_path"] = self.model_path
        self.tts_config["device_ids"] = self.device_ids

        self.tts_config["spk2info"] = ""
        self.tts_config["sampling_rate"] = 24000
        self.tts_config["llm_input_size"] = 896
        self.tts_config["llm_output_size"] = 896
        self.tts_config["speech_token_size"] = 6561

        self.weight_tensors = self.model_config.get("weight_tensors", [])
        for tensor in self.weight_tensors:
            tensor_name = tensor["name"]
            size = tensor["n_elements"]
            offset = tensor["data_offset"]
            logger.info(f"tensor_name: {tensor_name}, size: {size}, offset: {offset}")
            if "cosyvoice3_speech_tokenizer.hmm" == tensor["name"]:
                cosyvoice3_speech_tokenizer_hmm = {}
                cosyvoice3_speech_tokenizer_hmm["offset"] = offset
                cosyvoice3_speech_tokenizer_hmm["size"] = size
                self.tts_config["cosyvoice3_speech_tokenizer_hmm"] = cosyvoice3_speech_tokenizer_hmm
            elif "cosyvoice3_llm_qwen2_prefill.hmm" == tensor["name"]:
                cosyvoice3_llm_qwen2_prefill_hmm = {}
                cosyvoice3_llm_qwen2_prefill_hmm["offset"] = offset
                cosyvoice3_llm_qwen2_prefill_hmm["size"] = size
                self.tts_config["cosyvoice3_llm_qwen2_prefill_hmm"] = cosyvoice3_llm_qwen2_prefill_hmm
            elif "cosyvoice3_llm_qwen2_decode.hmm" == tensor["name"]:
                cosyvoice3_llm_qwen2_decode_hmm = {}
                cosyvoice3_llm_qwen2_decode_hmm["offset"] = offset
                cosyvoice3_llm_qwen2_decode_hmm["size"] = size
                self.tts_config["cosyvoice3_llm_qwen2_decode_hmm"] = cosyvoice3_llm_qwen2_decode_hmm
            elif "cosyvoice3_llm_decoder.hmm" == tensor["name"]:
                cosyvoice3_llm_decoder_hmm = {}
                cosyvoice3_llm_decoder_hmm["offset"] = offset
                cosyvoice3_llm_decoder_hmm["size"] = size
                self.tts_config["cosyvoice3_llm_decoder_hmm"] = cosyvoice3_llm_decoder_hmm
            elif "cosyvoice3_hift_part2.hmm" == tensor["name"]:
                cosyvoice3_hift_part2_hmm = {}
                cosyvoice3_hift_part2_hmm["offset"] = offset
                cosyvoice3_hift_part2_hmm["size"] = size
                self.tts_config["cosyvoice3_hift_part2_hmm"] = cosyvoice3_hift_part2_hmm
            elif "cosyvoice3_hift_part1.hmm" == tensor["name"]:
                cosyvoice3_hift_part1_hmm = {}
                cosyvoice3_hift_part1_hmm["offset"] = offset
                cosyvoice3_hift_part1_hmm["size"] = size
                self.tts_config["cosyvoice3_hift_part1_hmm"] = cosyvoice3_hift_part1_hmm
            elif "cosyvoice3_flow_spk.hmm" == tensor["name"]:
                cosyvoice3_flow_spk_hmm = {}
                cosyvoice3_flow_spk_hmm["offset"] = offset
                cosyvoice3_flow_spk_hmm["size"] = size
                self.tts_config["cosyvoice3_flow_spk_hmm"] = cosyvoice3_flow_spk_hmm
            elif "cosyvoice3_flow_encoder.hmm" == tensor["name"]:
                cosyvoice3_flow_encoder_hmm = {}
                cosyvoice3_flow_encoder_hmm["offset"] = offset
                cosyvoice3_flow_encoder_hmm["size"] = size
                self.tts_config["cosyvoice3_flow_encoder_hmm"] = cosyvoice3_flow_encoder_hmm
            elif "cosyvoice3_flow_decoder.hmm" == tensor["name"]:
                cosyvoice3_flow_decoder_hmm = {}
                cosyvoice3_flow_decoder_hmm["offset"] = offset
                cosyvoice3_flow_decoder_hmm["size"] = size
                self.tts_config["cosyvoice3_flow_decoder_hmm"] = cosyvoice3_flow_decoder_hmm
            elif "cosyvoice3_campplus.hmm" == tensor["name"]:
                cosyvoice3_campplus_hmm = {}
                cosyvoice3_campplus_hmm["offset"] = offset
                cosyvoice3_campplus_hmm["size"] = size
                self.tts_config["cosyvoice3_campplus_hmm"] = cosyvoice3_campplus_hmm
            elif "flow_input_embedding.bin" == tensor["name"]:
                flow_input_embedding_bin = {}
                flow_input_embedding_bin["offset"] = offset
                flow_input_embedding_bin["size"] = size
                self.tts_config["flow_input_embedding_bin"] = flow_input_embedding_bin
            elif "flow_input_embedding.pt" == tensor["name"]:
                flow_input_embedding_pt = {}
                flow_input_embedding_pt["offset"] = offset
                flow_input_embedding_pt["size"] = size
                self.tts_config["flow_input_embedding_pt"] = flow_input_embedding_pt
            elif "llm_task_id_embedding.bin" == tensor["name"]:
                llm_task_id_embedding_bin = {}
                llm_task_id_embedding_bin["offset"] = offset
                llm_task_id_embedding_bin["size"] = size
                self.tts_config["llm_task_id_embedding_bin"] = llm_task_id_embedding_bin
            elif "llm_task_id_embedding.pt" == tensor["name"]:
                llm_task_id_embedding_pt = {}
                llm_task_id_embedding_pt["offset"] = offset
                llm_task_id_embedding_pt["size"] = size
                self.tts_config["llm_task_id_embedding_pt"] = llm_task_id_embedding_pt
            elif "llm_sos_embedding.bin" == tensor["name"]:
                llm_sos_embedding_bin = {}
                llm_sos_embedding_bin["offset"] = offset
                llm_sos_embedding_bin["size"] = size
                self.tts_config["llm_sos_embedding_bin"] = llm_sos_embedding_bin
            elif "llm_sos_embedding.pt" == tensor["name"]:
                llm_sos_embedding_pt = {}
                llm_sos_embedding_pt["offset"] = offset
                llm_sos_embedding_pt["size"] = size
                self.tts_config["llm_sos_embedding_pt"] = llm_sos_embedding_pt
            elif "llm_speech_embedding.bin" == tensor["name"]:
                llm_speech_embedding_bin = {}
                llm_speech_embedding_bin["data"] = offset
                llm_speech_embedding_bin["offset"] = offset
                llm_speech_embedding_bin["size"] = size
                self.tts_config["llm_speech_embedding_bin"] = llm_speech_embedding_bin
            elif "llm_speech_embedding.pt" == tensor["name"]:
                llm_speech_embedding_pt = {}
                llm_speech_embedding_pt["offset"] = offset
                llm_speech_embedding_pt["size"] = size
                self.tts_config["llm_speech_embedding_pt"] = llm_speech_embedding_pt
            elif "quant_embedding.bin" == tensor["name"]:
                quant_embedding_bin = {}
                quant_embedding_bin["offset"] = offset
                quant_embedding_bin["size"] = size
                self.tts_config["quant_embedding_bin"] = quant_embedding_bin
            elif "quant_embedding.pt" == tensor["name"]:
                quant_embedding_pt = {}
                quant_embedding_pt["offset"] = offset
                quant_embedding_pt["size"] = size
                self.tts_config["quant_embedding_pt"] = quant_embedding_pt

        super().__init__(instance_config)

        logger.info(f"CosyVoice3 model loaded successfully with config: {self.tts_config}")

    def _load_model(self):
        self.tts_model = CosyVoice3(self.tts_config)

    @staticmethod
    def _to_wav_bytes(waveform, sample_rate: int) -> bytes:
        if waveform.ndim != 2:
            raise ValueError("Expected waveform shape [channel, samples]")

        pcm = waveform.clamp(-1.0, 1.0).mul(32767.0).short().cpu().numpy()
        pcm = np.ascontiguousarray(pcm.T)

        bio = BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(int(waveform.shape[0]))
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return bio.getvalue()

    def _ensure_model_loaded(self):
        if self.tts_model is not None:
            return

        with self._model_lock:
            if self.tts_model is None:
                self._load_model()

    def _synthesize(self, request_dict: Dict[str, Any]) -> bytes:
        self._ensure_model_loaded()

        text = str(request_dict.get("input") or "").strip()
        if not text:
            raise ValueError("`input` is required for /v1/audio/speech")

        prompt_wav_path = str(
            request_dict.get("prompt_wav_path") or DEFAULT_PROMPT_WAV_PATH
        ).strip()

        chunks = self.tts_model.inference_instruct(
            tts_text=text,
            instruct_text=str(
                request_dict.get("instructions")
                or request_dict.get("instruct_text")
                or ""
            ),
            prompt_wav=prompt_wav_path,
            zero_shot_spk_id=str(request_dict.get("zero_shot_spk_id", "")),
            speed=float(request_dict.get("speed", 1.0)),
            text_frontend=bool(request_dict.get("text_frontend", True)),
        )

        chunk_list = [chunk for chunk in chunks if chunk is not None]
        if not chunk_list:
            raise RuntimeError("TTS generation returned no audio chunks")

        waveform = torch.cat(chunk_list, dim=1).to(torch.float32).cpu()
        return self._to_wav_bytes(waveform, int(self.tts_model.sample_rate))

    def create_audio_speech(
        self,
        request: AudioSpeechRequest,
        raw_request: Request | None = None,
    ) -> AudioSpeechResponse:
        request_dict = request.model_dump(exclude_none=True)
        response_format = str(request_dict.get("response_format", "wav")).lower()
        voice = str(request_dict.get("voice", "default"))

        if response_format not in {"wav", "b64_json"}:
            raise ValueError("Only response_format `wav` and `b64_json` are supported")

        wav_bytes = self._synthesize(request_dict)

        if response_format == "b64_json":
            return AudioSpeechBase64Response(
                audio=base64.b64encode(wav_bytes).decode("utf-8"),
                format="wav",
                sample_rate=self.tts_model.sample_rate,
                voice=voice,
                model=request.model,
            )

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Audio-Sample-Rate": str(self.tts_model.sample_rate),
                "X-Voice": voice,
            },
        )