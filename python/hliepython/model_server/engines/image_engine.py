import os
import json
import asyncio
import base64
from io import BytesIO
from hliepython.utils.logging import async_logger as logger
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image

from typing import Dict, Any
from hliepython.model_server.engines.base_engine import InferenceEngine
from hliepython.model_server.engines.utils import (
    merge_additional_config,
    move_vllm_engine_args_from_additional_config,
    normalize_vllm_engine_args,
)

try:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
except ImportError:
    logger.error("vLLM 库未安装，图像引擎将无法使用本地推理功能！请安装 vLLM 或检查环境配置。")
    raise ImportError("vLLM 库未安装，图像引擎将无法使用本地推理功能！请安装 vLLM 或检查环境配置。")


class ImageEngine(InferenceEngine):
    """图像生成推理引擎"""

    def _load_model(self):
        """加载图像生成模型（使用 vLLM 或其他图像生成库）"""
        if AsyncLLMEngine is None:
            raise RuntimeError("vLLM not available")

        try:
            # vLLM 参数从 vllm_config 透传到底层 AsyncEngineArgs。
            vllm_config = self.config.get(
                "vllm_config",
                self.config.get("vllm_extra_config", {}),
            )
            vllm_engine_args = normalize_vllm_engine_args(vllm_config)
            user_additional_config = vllm_engine_args.pop("additional_config", None)
            additional_config = merge_additional_config(
                dict(self.additional_config),
                user_additional_config,
            )
            additional_config.setdefault("device_ids", self.device_ids)
            vllm_engine_args, additional_config = move_vllm_engine_args_from_additional_config(
                vllm_engine_args,
                additional_config,
            )

            # 配置 vLLM 引擎参数（适用于多模态模型）
            engine_arg_kwargs = {
                **vllm_engine_args,
                "model": self.model_path,
                "additional_config": additional_config,
            }
            engine_args = AsyncEngineArgs(**engine_arg_kwargs)

            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info(f"ImageEngine: vLLM 引擎加载完成 {self.model_path} 到设备 {self.device_ids}")

        except Exception as e:
            logger.error(f"ImageEngine: vLLM 引擎加载失败 {e}")
            raise RuntimeError(f"Failed to load vLLM engine: {e}")

    async def create_images_generations(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理图像生成请求"""
        prompt = request.get("prompt", "")
        n = request.get("n", 1)
        size = request.get("size", "512x512")

        # 使用 vLLM 生成图像描述，然后使用 diffusers 生成图像
        try:
            # 生成详细的图像描述
            enhanced_prompt = f"Generate a detailed description for creating an image with: {prompt}"

            sampling_params = SamplingParams(
                max_tokens=512,
                temperature=0.7,
                stop=["</s>", "<|endoftext|>"],
            )

            # 生成描述
            async def generate_description():
                results = self.engine.generate(enhanced_prompt, sampling_params, request_id="image_gen")
                final_output = None
                async for request_output in results:
                    final_output = request_output

                if final_output and final_output.outputs:
                    description = final_output.outputs[0].text
                    # 使用描述生成图像
                    return self._generate_image_from_description(description, n, size)
                else:
                    raise Exception("No description generated")

            # 运行异步任务
            return await generate_description()

        except Exception as e:
            logger.error(f"ImageEngine: 图像生成失败 {e}")
            return {
                "error": {
                    "message": f"Image generation failed: {str(e)}",
                    "type": "generation_error"
                }
            }

    def _generate_image_from_description(self, description: str, n: int, size: str) -> Dict[str, Any]:
        """根据描述生成图像"""
        try:
            # 尝试使用 diffusers 生成图像
            from diffusers import StableDiffusionPipeline
            import torch

            if not hasattr(self, 'pipeline') or self.pipeline is None:
                # 加载 Stable Diffusion 模型
                model_id = "runwayml/stable-diffusion-v1-5"  # 或其他模型
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )

                if torch.cuda.is_available() and self.device_ids:
                    self.pipeline = self.pipeline.to(f"cuda:{self.device_ids[0]}")
                else:
                    self.pipeline = self.pipeline.to("cpu")

            # 解析尺寸
            width, height = map(int, size.split('x'))

            # 生成图像
            images = self.pipeline(
                description,
                num_images_per_prompt=n,
                height=height,
                width=width,
                num_inference_steps=20,  # 可以从配置中获取
            ).images

            # 转换为 base64
            image_data = []
            for img in images:
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                image_data.append({
                    "url": f"data:image/png;base64,{img_base64}",
                    "revised_prompt": description
                })

            return {
                "created": int(asyncio.get_event_loop().time()),
                "data": image_data
            }

        except ImportError:
            logger.warning("diffusers 库未安装，使用占位符图像数据")
            # 回退到占位符实现
            return {
                "created": 1234567890,
                "data": [
                    {
                        "url": f"data:image/png;base64,{description[:20]}...",  # 占位符
                        "revised_prompt": description
                    } for _ in range(n)
                ]
            }
        except Exception as e:
            logger.error(f"图像生成失败: {e}")
            return {
                "error": {
                    "message": f"Image generation failed: {str(e)}",
                    "type": "generation_error"
                }
            }