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

from gguf import GGUFReader

reader = GGUFReader("/data/useradmin/houmo/lanyue_release/HiModel/v1.1.0/cosyvoice3/HiModel_cosyvoice3.gguf")

print("=== Metadata ===")
for k, field in reader.fields.items():
    try:
        val = field.parts[0]
    except:
        val = field.value
    print(f"{k:<35} : {val}")

print("\n=== Tensors ===")
for t in reader.tensors:
    shape = "×".join(map(str, t.shape))
    print(f"{t.name:<35} shape={shape:12} dtype={t.tensor_type.name}")


from gguf import GGUFReader

# 读取 GGUF 文件
reader = GGUFReader("/data/useradmin/houmo/lanyue_release/HiModel/v1.1.0/cosyvoice3/HiModel_cosyvoice3.gguf")

print("=== Metadata ===")
for key, field in reader.fields.items():
    # GGUF 字段分两种：带 parts（数组/多值）、直接 value（单值）
    # 严格按字段原生数据读取，不做任何类型强转
    if hasattr(field, 'parts') and field.parts is not None and len(field.parts) > 0:
        value = field.parts
        # 如果是单元素数组，直接提取元素，保持原生类型
        if len(value) == 1:
            value = value[0]
    else:
        value = field.value

    # 打印：字段名 | 类型 | 值
    print(f"{key:<35} type={type(value).__name__:<10} : {value}")

print("\n=== Tensors ===")
for tensor in reader.tensors:
    shape = "×".join(map(str, tensor.shape))
    print(f"{tensor.name:<35} shape={shape:12} dtype={tensor.tensor_type.name}")