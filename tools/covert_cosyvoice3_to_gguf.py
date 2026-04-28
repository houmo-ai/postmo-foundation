#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 CosyVoice3 模型打包成单个 GGUF 文件
所有文件作为原始字节存储
"""

import os
import sys
import json
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加 gguf-py 路径
if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / "gguf-py"))

import gguf
import numpy as np

# ==================== 配置 ====================
# 所有需要打包的文件
MODEL_FILES = {
    # Embedding 文件（.pt 和 .bin 都保留）
    "quant_embedding.pt": "*.quant_embedding.pt",
    "quant_embedding.bin": "*.quant_embedding.bin",
    "llm_speech_embedding.pt": "*.llm_speech_embedding.pt",
    "llm_speech_embedding.bin": "*.llm_speech_embedding.bin",
    "llm_sos_embedding.pt": "*.llm_sos_embedding.pt",
    "llm_sos_embedding.bin": "*.llm_sos_embedding.bin",
    "llm_task_id_embedding.pt": "*.llm_task_id_embedding.pt",
    "llm_task_id_embedding.bin": "*.llm_task_id_embedding.bin",
    "flow_input_embedding.pt": "*.flow_input_embedding.pt",
    "flow_input_embedding.bin": "*.flow_input_embedding.bin",

    # HMM 模型文件
    "cosyvoice3_campplus.hmm": "*.cosyvoice3_campplus.hmm",
    "cosyvoice3_flow_decoder.hmm": "*.cosyvoice3_flow_decoder.hmm",
    "cosyvoice3_flow_encoder.hmm": "*.cosyvoice3_flow_encoder.hmm",
    "cosyvoice3_flow_spk.hmm": "*.cosyvoice3_flow_spk.hmm",
    "cosyvoice3_hift_part1.hmm": "*.cosyvoice3_hift_part1.hmm",
    "cosyvoice3_hift_part2.hmm": "*.cosyvoice3_hift_part2.hmm",
    "cosyvoice3_llm_decoder.hmm": "*.cosyvoice3_llm_decoder.hmm",
    "cosyvoice3_llm_qwen2_decode.hmm": "*.cosyvoice3_llm_qwen2_decode.hmm",
    "cosyvoice3_llm_qwen2_prefill.hmm": "*.cosyvoice3_llm_qwen2_prefill.hmm",
    "cosyvoice3_speech_tokenizer.hmm": "*.cosyvoice3_speech_tokenizer.hmm",

    # CosyVoice-BlankEN
    "config.json": "*.config.json",
    "vocab.json": "*.vocab.json",
    "tokenizer_config.json": "*.tokenizer_config.json",
    "merges.txt": "*.merges.txt",
    "generation_config.json": "*.generation_config.json",
}


class GGUFPacker:
    """GGUF 打包器 - 文件作为原始字节存储"""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.tensors: List[tuple[str, np.ndarray]] = []
        self.file_info: Dict[str, Dict] = {}
        self.vocab_keys: List[str] = []
        self.vocab_size: int = 0
        self.merges: List[str] = []
        self.merges_size: int = 0

    def add_raw_file(self, key: str, file_path: str):
        """添加文件，存储为原始字节"""
        if not os.path.exists(file_path):
            print(f"⚠️  文件不存在: {file_path}")
            return False

        with open(file_path, 'rb') as f:
            data = f.read()

        # 转为 1D numpy 数组
        data_arr = np.frombuffer(data, dtype=np.uint8)
        self.tensors.append((key, data_arr))

        file_size = len(data)
        ext = os.path.splitext(file_path)[1]

        self.file_info[key] = {
            "original_path": file_path,
            "size": file_size,
            "format": ext,
        }

        print(f"✓ 添加: {key}")
        print(f"   <- {file_path}")
        print(f"   size: {file_size / 1024 / 1024:.2f} MB")
        return True

    def add_vocab(self, vocab_path: str):
        """添加 vocab.json，提取所有 key"""
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"vocab.json 不存在: {vocab_path}")

        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        # 直接提取 key（value 已经按 id 从小到大排序）
        self.vocab_keys = list(vocab.keys())
        self.vocab_size = len(self.vocab_keys)

        print(f"✓ 添加 Vocab: {vocab_path}")
        print(f"   vocab 大小: {self.vocab_size}")
        print(f"   前10个: {self.vocab_keys[:10]}")

    def add_merges_file(self, file_path: str):
        """添加 merges.txt 按行读取 添加到 self.merges : list"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'merges.txt 不存在: {file_path}')
        with open(file_path, 'r', encoding='utf-8') as f:
            self.merges = [line.strip() for line in f if line.strip()]
        self.merges_size = len(self.merges)
        print(f"✓ 添加 Merges: {file_path}")
        print(f"   merges 数量: {self.merges_size}")
        print(f"   前5个: {self.merges[:5]}")


    def set_metadata(self, key: str, value: Any):
        """设置元数据"""
        self.file_info[key] = value

    def write(self):
        """写入 GGUF 文件"""
        print(f"\n📦 打包: {self.output_path}")
        print(f"   文件数量: {len(self.tensors)}")

        total_bytes = sum(t[1].nbytes for t in self.tensors)
        print(f"   总数据量: {total_bytes / 1024 / 1024:.2f} MB")

        # 创建 GGUF Writer
        gguf_writer = gguf.GGUFWriter(
            path=None,
            arch="cosyvoice3",
            endianess=gguf.GGUFEndian.LITTLE,
        )

        # 写入基础元数据
        gguf_writer.add_string("model_name", "CosyVoice3")
        gguf_writer.add_string("model_type", "cosyvoice3_tts")
        gguf_writer.add_int32("total_file_count", len(self.tensors))
        gguf_writer.add_string("file_info", json.dumps(self.file_info))

        if self.vocab_keys:
            gguf_writer.add_int32("vocab_size", self.vocab_size)
            gguf_writer.add_array(f"vocab", self.vocab_keys)

        if self.merges:
            gguf_writer.add_int32(f"merges_size", self.merges_size)
            gguf_writer.add_array("merges", self.merges)

        # 写入文件信息（作为 tensor 元数据）
        # for key, info in self.file_info.items():
        #     gguf_writer.add_string(f"info.{key}.path", info["original_path"])
        #     gguf_writer.add_int32(f"info.{key}.size", info["size"])
        #     gguf_writer.add_string(f"info.{key}.format", info["format"])

        # 写入文件数据（作为 tensor）
        for name, data in self.tensors:
            gguf_writer.add_tensor(
                name=name,
                tensor=data,
                raw_dtype=gguf.GGMLQuantizationType.I8,  # 原始字节存储
            )

        # 分三步写入
        gguf_writer.write_header_to_file(path=self.output_path)
        gguf_writer.write_kv_data_to_file()
        gguf_writer.write_tensors_to_file(progress=True)

        file_size = os.path.getsize(self.output_path)
        print(f"\n✅ 完成! GGUF 大小: {file_size / 1024 / 1024:.2f} MB")


def find_file(directory: str, pattern: str) -> Optional[str]:
    """在目录中查找文件"""
    search_pattern = pattern.replace("*.", "")
    files = glob.glob(os.path.join(directory, "**", search_pattern), recursive=True)
    return files[0] if files else None


def pack_cosyvoice3(input_dir: str, output_path: str):
    """打包 CosyVoice3 模型"""
    packer = GGUFPacker(output_path)

    # 添加所有文件
    for key, pattern in MODEL_FILES.items():
        file_path = find_file(input_dir, pattern)
        if file_path:
            packer.add_raw_file(key, file_path)
        else:
            print(f"⚠️  未找到: {key}")

    # 必须有 vocab.json
    vocab_path = find_file(input_dir, "vocab.json")
    packer.add_vocab(vocab_path)

    # 添加 merges.txt
    merges_path = find_file(input_dir, "merges.txt")
    if merges_path:
        packer.add_merges_file(merges_path)

    # 写入
    packer.write()
    return packer


# ==================== 主函数 ====================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="将 CosyVoice3 打包成 GGUF")
    parser.add_argument("-i", "--input", required=True, help="输入目录")
    parser.add_argument("-o", "--output", required=True, help="输出 GGUF 文件")

    args = parser.parse_args()

    pack_cosyvoice3(
        input_dir=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()