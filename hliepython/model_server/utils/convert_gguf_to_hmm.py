import os
import gc
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from loguru import logger
from gguf import GGUFReader

@dataclass
class TensorMetadata:
    """张量元数据结构体，存储核心字段，与原tensor对象解耦"""
    shape: tuple          # 对应tensor.shape
    n_elements: int       # 对应tensor.n_elements
    quantization: str     # 对应quantization_str（张量量化类型）
    data_offset: int      # 对应tensor.data_offset
    name: str             # 补充tensor.name，用于文件保存和标识（必要字段）

    def to_dict(self) -> dict:
        """将 TensorMetadata 转换为字典，确保 JSON 可序列化"""
        # 将 shape 转换为元组，并确保所有元素都是 Python int 类型
        shape_list = list(self.shape)
        converted_shape = []
        for item in shape_list:
            if hasattr(item, 'item'):
                # numpy 标量转换为 Python int
                converted_shape.append(int(item.item()))
            elif hasattr(item, 'tolist'):
                # numpy 数组转换为 Python 列表
                converted_shape.append(item.tolist())
            else:
                converted_shape.append(int(item))
        return {
            "shape": tuple(converted_shape),
            "n_elements": int(self.n_elements),
            "quantization": str(self.quantization),
            "data_offset": int(self.data_offset),
            "name": str(self.name),
        }

    def __repr__(self) -> str:
        return f"TensorMetadata(name={self.name!r}, shape={self.shape}, n_elements={self.n_elements}, quantization={self.quantization!r}, data_offset={self.data_offset})"


# ------------------------------------------------------------------------------
# 工具函数1：解析GGUF路径（返回标准化绝对路径）
# ------------------------------------------------------------------------------
def resolve_gguf_model_path(model_path: str) -> str:
    """
    标准化解析GGUF模型路径，支持直接传入文件/目录
    目录会自动查找 HiMode*.gguf 文件
    :return: 唯一有效的GGUF文件绝对路径
    :raises FileNotFoundError, ValueError: 路径异常
    """
    if not model_path or not isinstance(model_path, str):
        raise ValueError("模型路径不能为空，必须是字符串类型")

    abs_path = os.path.abspath(model_path.strip())

    # 是文件
    if os.path.isfile(abs_path):
        return abs_path

    # 是目录：查找 HiMode*.gguf
    if os.path.isdir(abs_path):
        gguf_files = sorted(glob.glob(os.path.join(abs_path, "HiMode*.gguf")))
        if len(gguf_files) == 0:
            raise FileNotFoundError(f"目录下未找到 HiMode*.gguf 文件: {abs_path}")
        if len(gguf_files) > 1:
            raise ValueError(f"目录下存在多个GGUF文件，无法确定唯一模型: {gguf_files}")
        return os.path.abspath(gguf_files[0])

    # 路径不存在
    raise FileNotFoundError(f"路径不存在或不是有效文件/目录: {abs_path}")

# ------------------------------------------------------------------------------
# 工具函数2：GGUF解析核心工具（对外提供）
# 功能：传入GGUF路径 → 返回标准字典（无全局变量、无退出、无副作用）
# ------------------------------------------------------------------------------
def parse_gguf_to_dict(
    gguf_path: str,
    additional_config: Optional[Dict] = None,
    save_config_files: bool = True
) -> Dict[str, Any]:
    """
    对外提供的GGUF解析工具函数
    传入GGUF路径 → 返回完整元信息字典（主模型 + VL模型 + 张量信息）
    无全局变量、无sys.exit、可重复调用、线程安全、资源自动释放
    """
    # 最终返回结果容器
    result = {
        "model_path": "",
        "model_dir": "",
        "main_model": {},
        "vl_model": {},
        "configfiles_tensors": [],
        "weight_tensors": []
    }

    main_reader: Optional[GGUFReader] = None
    vl_reader: Optional[GGUFReader] = None

    try:
        # ====================== 路径处理 ======================
        gguf_file_path = resolve_gguf_model_path(gguf_path)
        output_dir = os.path.dirname(gguf_file_path)
        
        result["model_path"] = gguf_file_path
        result["model_dir"] = output_dir
        
        gguf_file = Path(gguf_file_path)

        if not gguf_file.exists():
            raise FileNotFoundError(f"GGUF文件不存在: {gguf_file_path}")

        logger.info(f"开始解析GGUF主模型: {gguf_file_path}")

        # ====================== 读取主模型 ======================
        main_reader = GGUFReader(gguf_file_path)
        main_meta = {}

        # 读取KV元数据
        for key, field in main_reader.fields.items():
            value = field.parts[field.data[0]]
            if hasattr(value, "tolist"):
                value = value.tolist()
            elif hasattr(value, "item"):
                value = value.item()
            main_meta[key] = value

        result["main_model"] = main_meta

        # ====================== 解析张量 ======================
        config_tensors: List[TensorMetadata] = []
        weight_tensors: List[TensorMetadata] = []

        for tensor in main_reader.tensors:
            shape = tuple(
                int(item.item()) if hasattr(item, "item") else int(item)
                for item in tensor.shape
            )

            meta = TensorMetadata(
                shape=shape,
                n_elements=int(tensor.n_elements),
                quantization=str(tensor.tensor_type.name),
                data_offset=int(tensor.data_offset),
                name=str(tensor.name)
            )

            # 分类：配置文件张量 / 权重张量
            if meta.name.endswith((".json", ".txt", ".jinja", ".md")):
                config_tensors.append(meta)
            else:
                weight_tensors.append(meta)

        # 转为字典存入结果
        result["configfiles_tensors"] = [m.to_dict() for m in config_tensors]
        result["weight_tensors"] = [m.to_dict() for m in weight_tensors]

        # ====================== 导出配置文件（可选） ======================
        if save_config_files and config_tensors:
            os.makedirs(output_dir, exist_ok=True)
            with open(gguf_file_path, "rb") as f:
                for meta in config_tensors:
                    f.seek(meta.data_offset)
                    data = f.read(meta.n_elements)
                    out_path = os.path.join(output_dir, meta.name)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, "wb") as out_f:
                        out_f.write(data)
                    logger.info(f"已导出配置文件: {out_path}")

        # ====================== 处理VL模型 ======================
        vl_config = additional_config or {}
        vl_path = vl_config.get("visual_model_path")

        if vl_path and isinstance(vl_path, str):
            vl_abs_path = os.path.abspath(vl_path.strip())
            if os.path.isfile(vl_abs_path):
                logger.info(f"解析VL模型: {vl_abs_path}")
                vl_reader = GGUFReader(vl_abs_path)
                vl_meta = {}

                for key, field in vl_reader.fields.items():
                    value = field.parts[field.data[0]]
                    if hasattr(value, "tolist"):
                        value = value.tolist()
                    elif hasattr(value, "item"):
                        value = value.item()
                    vl_meta[key] = value

                result["vl_model"] = vl_meta

                # 读取VL权重张量
                for tensor in vl_reader.tensors:
                    if tensor.name.endswith((".json", ".txt", ".jinja", ".md")):
                        continue

                    shape = tuple(
                        int(item.item()) if hasattr(item, "item") else int(item)
                        for item in tensor.shape
                    )

                    meta = TensorMetadata(
                        shape=shape,
                        n_elements=int(tensor.n_elements),
                        quantization=str(tensor.tensor_type.name),
                        data_offset=int(tensor.data_offset),
                        name=str(tensor.name)
                    )
                    weight_tensors.append(meta)

                # 更新权重列表
                result["weight_tensors"] = [m.to_dict() for m in weight_tensors]

        logger.info("✅ GGUF 解析完成")
        return result

    finally:
        # 安全释放资源
        for reader in [main_reader, vl_reader]:
            if reader:
                try:
                    if hasattr(reader, "close"):
                        reader.close()
                except Exception as e:
                    logger.warning(f"关闭GGUFReader失败: {e}")
        gc.collect()


# def main():
#     print("=" * 60)
#     print("          GGUF 解析工具 - 使用示例")
#     print("=" * 60)

#     # ========== 示例1：直接传入 GGUF 文件路径 ==========
#     print("\n【示例1】解析单个 GGUF 模型文件")
#     # 请替换成你的真实 .gguf 文件路径
#     model_path = "/data/useradmin/houmo/lanyue_release/HiModel/v1.1.0/cosyvoice3/HiModel_cosyvoice3.gguf"
#     try:
#         model_info = parse_gguf_to_dict(model_path)
#         print(f"✅ 主模型架构: {model_info['main_model'].get('general.architecture', 'unknown')}")
#         print(f"✅ 配置文件张量数量: {len(model_info['configfiles_tensors'])}")
#         print(f"✅ 权重张量数量: {len(model_info['weight_tensors'])}")
#     except Exception as e:
#         print(f"❌ 示例1失败: {e}")

#     # # ========== 示例2：传入模型目录（自动查找 HiMode*.gguf） ==========
#     # print("\n【示例2】传入目录，自动查找 HiMode 模型")
#     # # 请替换成你的模型目录
#     # model_dir = "/data/useradmin/houmo/lanyue_release/HiModel/v1.1.0/cosyvoice3"
#     # try:
#     #     model_info = parse_gguf_to_dict(model_dir)
#     #     print(f"✅ 解析成功，模型路径: {resolve_gguf_model_path(model_dir)}")
#     # except Exception as e:
#     #     print(f"❌ 示例2失败: {e}")

#     # # ========== 示例3：主模型 + VL 视觉模型 联合解析 ==========
#     # print("\n【示例3】解析主模型 + VL 视觉模型")
#     # # 请替换成你的真实路径
#     # main_model_path = "/data/useradmin/houmo/lanyue_release/HiModel/v1.1.0/cosyvoice3/HiModel_cosyvoice3.gguf"
#     # vl_model_path = "/data/useradmin/houmo/lanyue_release/HiModel/v1.1.0/cosyvoice3/clip-vit.gguf"

#     # config = {
#     #     "visual_model_path": vl_model_path
#     # }

#     # try:
#     #     model_info = parse_gguf_to_dict(main_model_path, additional_config=config)
#     #     print(f"✅ 主模型: {model_info['main_model'].get('general.architecture', 'N/A')}")
#     #     print(f"✅ VL模型: {model_info['vl_model'].get('general.architecture', 'N/A')}")
#     #     print(f"✅ 总权重张量数量: {len(model_info['weight_tensors'])}")
#     # except Exception as e:
#     #     print(f"❌ 示例3失败: {e}")

#     # # ========== 示例4：只解析，不导出配置文件 ==========
#     # print("\n【示例4】仅解析，不导出文件")
#     # try:
#     #     model_info = parse_gguf_to_dict(model_path, save_config_files=False)
#     #     print("✅ 纯解析模式完成")
#     # except Exception as e:
#     #     print(f"❌ 示例4失败: {e}")

#     # print("\n" + "=" * 60)
#     # print("                   全部示例运行完成")
#     # print("=" * 60)
    
def main():
    print("=" * 60)
    print("          GGUF 解析工具 - 张量元数据查看")
    print("=" * 60)

    # ========== 请在这里替换成你的真实 GGUF 路径 ==========
    model_path = "/data/useradmin/houmo/lanyue_release/HiModel/v1.1.0/cosyvoice3/HiModel_cosyvoice3.gguf"  # 可改为文件或目录

    try:
        model_info = parse_gguf_to_dict(model_path, save_config_files=True)

        print("\n📊 模型基础信息")
        print(f"模型架构: {model_info['main_model'].get('general.architecture', 'unknown')}")
        print(f"配置文件张量数量: {len(model_info['configfiles_tensors'])}")
        print(f"权重张量数量: {len(model_info['weight_tensors'])}")

        print(f"\n{'-' * 94}")
        print("🔍 权重张量元数据信息")
        print("-" * 94)
        print(f"{'No.':<5} {'tensor name':<40} {'shape':<20} {'quantization':<15} {'n_elements':<10}")
        print("-" * 94)

        for idx, tensor in enumerate(model_info["weight_tensors"], 1):
            name = tensor["name"]
            shape = str(tensor["shape"])
            quant = tensor["quantization"]
            n_ele = tensor["n_elements"]
            print(f"{idx:<5} {name:<40} {shape:<20} {quant:<15} {n_ele:<10}")

        print("-" * 94)
        print("✅ 权重信息打印完成")

    except Exception as e:
        print(f"\n❌ 运行失败: {str(e)}")

if __name__ == "__main__":
    main()