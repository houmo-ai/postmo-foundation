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

from typing import Dict, Any

from hliepython.model_server.engines.base_engine import InferenceEngine

class InferenceEngineFactory:
    """推理引擎工厂"""

    @staticmethod
    def get_engine(instance_config: Dict[str, Any]) -> InferenceEngine:
        model_type = instance_config.get("model_type", "llm")

        if model_type == "llm":
            from hliepython.model_server.engines.text_engine import TextEngine

            return TextEngine(instance_config)
        elif model_type == "image":
            from hliepython.model_server.engines.image_engine import ImageEngine

            return ImageEngine(instance_config)
        elif model_type == "asr":
            from hliepython.model_server.engines.asr_engine import ASREngine

            return ASREngine(instance_config)
        elif model_type == "forced_aligner":
            from hliepython.model_server.engines.forced_aligner_engine import ForcedAlignerEngine

            return ForcedAlignerEngine(instance_config)
        elif model_type == "tts":
            from hliepython.model_server.engines.tts_engine import TTSEngine

            return TTSEngine(instance_config)
        elif model_type == "embedding":
            from hliepython.model_server.engines.embedding_engine import EmbeddingEngine

            return EmbeddingEngine(instance_config)
        elif model_type == "reranker":
            from hliepython.model_server.engines.reranker_engine import RerankerEngine

            return RerankerEngine(instance_config)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")