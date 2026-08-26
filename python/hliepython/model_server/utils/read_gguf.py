import os
import io
import struct
import numpy as np
import torch
import torch.nn.functional as F
from enum import Enum
from gguf.gguf_reader import GGUFReader

# 定义与C++对应的Embedding数据类型枚举
class EmbeddingDtype(Enum):
    INT16 = 0    # 对应C++的INT16
    FLOAT16 = 1  # 对应C++的FLOAT16

def load_embedding_from_binary(binary_data, offset=0):
    """
    从二进制数据加载Embedding权重（完全对齐C++逻辑，INT16保持原始类型，修复只读警告）
    :param binary_data: 二进制权重数据（bytes类型）
    :param offset: 偏移量（对应C++的offset参数）
    :return: torch.Tensor - 构建好的embedding权重张量
    """
    buffer = io.BytesIO(binary_data)
    
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

def main(file_name):
    """
    使用 gguf 库读取 .hmm 文件中的张量信息和元数据
    """
    # 初始化 GGUFReader
    try:
        reader = GGUFReader(file_name)
        print(f"成功加载文件: {file_name}")
    except Exception as e:
        print(f"加载GGUF文件失败: {e}")
        return
    
    # 读取架构信息
    try:
        arch = reader.get_field("general.architecture")
        if arch is not None:
            string_bytes = bytes(arch.parts[-1])
            arch = string_bytes.decode("utf-8")
            print(f"模型架构: {arch}")
            
            blocks = reader.get_field(arch + ".block_count")
            if blocks is not None:
                blocks = int(blocks.parts[-1][0])
                print("Blocks数量: ", blocks)
    except Exception as e:
        print(f"读取模型元数据失败: {e}")
    
    # 主处理逻辑
    for tensor in reader.tensors:
        # 打印张量基础信息
        print(f"\n{'='*60}")
        print(f"Tensor Name: {tensor.name}")
        print(f"  Original Shape: {tensor.shape}")
        print(f"  Original Dtype: {tensor.data.dtype}")
        
        embedding_weight = None
        
        if tensor.name.endswith(".pt"):
            # 处理JIT序列化的embedding权重
            try:
                buffer = io.BytesIO(tensor.data.tobytes())
                embedding = torch.jit.load(buffer, map_location=torch.device("cpu"))
                
                # 优先从buffers提取，失败则从parameters提取
                try:
                    embedding_weight = embedding._buffers["0"]
                except KeyError:
                    print("Warning: '0' not found in buffers, 尝试从parameters提取")
                    embedding_weight = embedding._parameters["0"]
                
                # 确保张量可写
                if embedding_weight is not None:
                    embedding_weight = embedding_weight.clone()
                    print(f"  加载的权重类型: {embedding_weight.dtype}, 形状: {embedding_weight.shape}")
                else:
                    print("Warning: 未能提取embedding权重")
                    continue
                    
            except Exception as e:
                print(f"加载{tensor.name}失败: {e}")
                continue
            
        elif tensor.name.endswith(".bin"):
            # 处理二进制格式的embedding权重
            try:
                # 从tensor的二进制数据中加载embedding权重
                embedding_weight = load_embedding_from_binary(tensor.data.tobytes(), offset=0)
            except Exception as e:
                print(f"解析{tensor.name}二进制数据失败: {e}")
                import traceback
                traceback.print_exc()  # 打印详细堆栈信息
                continue
        
        else:
            # 跳过其他张量
            print(f"Skipping unsupported tensor: {tensor.name}")
            continue
        
        # ========== 统一的Embedding查找逻辑（适配int16类型） ==========
        if embedding_weight is not None:
            # 准备输入Token ID（必须是LongTensor）
            input_ids = torch.tensor([[10, 25, 5], [40, 99, 1]], dtype=torch.long)
            print(f"\n输入ID信息:")
            print(f"  形状: {input_ids.shape}")
            print(f"  数值范围: [{input_ids.min().item()}, {input_ids.max().item()}]")
            
            # 检查输入ID是否越界
            if input_ids.max().item() >= embedding_weight.shape[0]:
                print(f"Warning: 输入ID {input_ids.max().item()} 超出权重表大小 {embedding_weight.shape[0]}")
                # 截断输入ID到有效范围
                input_ids = torch.clamp(input_ids, 0, embedding_weight.shape[0]-1)
                print(f"  已截断输入ID到有效范围: [{input_ids.min().item()}, {input_ids.max().item()}]")
            
            # 执行Embedding查找
            with torch.no_grad():
                try:
                    # F.embedding原生支持int16类型权重
                    output_embeddings = F.embedding(input_ids, embedding_weight)
                    print("\n✅ Embedding查找成功！")
                    print(f"  输出形状: {output_embeddings.shape}")
                    print(f"  输出数据类型: {output_embeddings.dtype}")
                    print(f"  输出张量（第一个向量的前5维）: {output_embeddings[0, 0, :5].tolist()}")
                    
                    # 验证原始INT16数值
                    if output_embeddings.dtype == torch.int16:
                        print(f"  原始INT16数值验证（第一个向量前3个）: {output_embeddings[0, 0, :3].tolist()}")
                        
                except Exception as e:
                    print(f"\n❌ Embedding查找失败: {e}")
                    # 打印详细调试信息
                    print(f"  权重信息 - 形状: {embedding_weight.shape}, 类型: {embedding_weight.dtype}")
                    print(f"  权重数值范围: [{embedding_weight.min().item()}, {embedding_weight.max().item()}]")
                    import traceback
                    traceback.print_exc()
        
        print("-" * 60)  # 分隔符

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"用法: {sys.argv[0]} <gguf_file>")
        print(f"示例: {sys.argv[0]} model.gguf")
        sys.exit(1)

    file_name = sys.argv[1]
    if not os.path.exists(file_name):
        print(f"文件不存在: {file_name}")
        sys.exit(1)

    # 设置PyTorch不显示冗余警告
    torch.set_printoptions(precision=4, sci_mode=False)
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    main(file_name)
    print("\n处理完成！")