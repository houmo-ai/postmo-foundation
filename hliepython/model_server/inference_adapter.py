from typing import Dict, Any

from hliepython.model_server.engines.base_engine import InferenceEngine

class InferenceEngineFactory:
    """推理引擎工厂"""

    @staticmethod
    def get_engine(instance_config: Dict[str, Any]) -> InferenceEngine:
        model_type = instance_config.get("model_type", "text")

        if model_type == "text":
            from hliepython.model_server.engines.text_engine import TextEngine

            return TextEngine(instance_config)
        elif model_type == "image":
            from hliepython.model_server.engines.image_engine import ImageEngine

            return ImageEngine(instance_config)
        elif model_type == "asr":
            from hliepython.model_server.engines.asr_engine import ASREngine

            return ASREngine(instance_config)
        elif model_type == "tts":
            from hliepython.model_server.engines.tts_engine import TTSEngine

            return TTSEngine(instance_config)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")