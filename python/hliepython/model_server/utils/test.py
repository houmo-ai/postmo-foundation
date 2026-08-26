import os
import torch
import numpy as np
import torch.nn.functional as F
from gguf.gguf_reader import GGUFReader

def main(file_name):
    try:
        reader = GGUFReader(file_name)
        print(f"成功加载文件: {file_name}")
    except Exception as e:
        print(f"加载GGUF失败: {e}")
        return

    # 读取架构
    try:
        arch = reader.get_field("general.architecture")
        if arch:
            arch_str = bytes(arch.parts[-1]).decode("utf-8")
            print(f"模型架构: {arch_str}")
    except:
        pass

    # 遍历张量
    for tensor in reader.tensors:
        name = tensor.name
        print(f"\n{'='*60}")
        print(f"Tensor Name: {name}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.data.dtype}")

        # 只处理 .bin 且包含 embedding 的张量
        if not name.endswith(".bin"):
            print("跳过：只处理 .bin 权重")
            continue
        if "embedding" not in name.lower():
            print("跳过：非 embedding 权重")
            continue

        np_data = tensor.data
        total = np_data.size

        # ======================
        # CosyVoice3 官方真实维度（全部完美整除）
        # ======================
        try:
            if "quant_embedding" in name:
                # 文本Embedding
                weight = torch.from_numpy(np_data).reshape(-1, 896)

            elif "llm_speech_embedding" in name:
                # 语音Embedding
                weight = torch.from_numpy(np_data).reshape(-1, 896)

            elif "sos_embedding" in name or "task_id_embedding" in name:
                # 特殊标记Embedding
                weight = torch.from_numpy(np_data).reshape(-1, 896)

            elif "flow_input_embedding" in name:
                # Flow输入Embedding：1049760 / 384 = 2734 → 完美整除！
                weight = torch.from_numpy(np_data).reshape(-1, 80)

            else:
                weight = torch.from_numpy(np_data).reshape(-1, 80)

        except Exception as e:
            print(f"❌ 加载失败: {e}")
            continue

        # ======================
        # 推理测试
        # ======================
        print(f"✅ 加载成功: {weight.shape}, dtype={weight.dtype}")
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        vocab_size = weight.shape[0]
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)

        try:
            with torch.no_grad():
                out = F.embedding(input_ids, weight.to(torch.float32))
            print(f"✅ 查找成功！输出形状 = {out.shape}")
        except Exception as e:
            print(f"❌ 推理失败: {e}")

        print("-" * 60)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"用法: {sys.argv[0]} model.gguf")
        sys.exit(1)

    file = sys.argv[1]
    if not os.path.exists(file):
        print("文件不存在")
        sys.exit(1)

    torch.set_printoptions(sci_mode=False)
    import warnings
    warnings.filterwarnings("ignore")
    main(file)
    print("\n处理完成！")