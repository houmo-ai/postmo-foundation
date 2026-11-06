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

# -*- coding: UTF-8 -*-
import base64
import struct
from typing import List, Literal, Optional, Union

from pydantic import model_validator

from hliepython.model_server.engines.protocol.chat_completions import (
    OpenAIBaseModel,
    UsageInfo,
)

EncodingFormat = Literal["float", "base64"]
EmbeddingInput = Union[str, List[str], List[int], List[List[int]]]


class EmbeddingRequest(OpenAIBaseModel):
    model: Optional[str] = None
    input: EmbeddingInput
    encoding_format: EncodingFormat = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_input(cls, values):
        if not isinstance(values, dict):
            return values

        val = values.get("input")
        if isinstance(val, str):
            return values

        if not isinstance(val, list) or len(val) == 0:
            raise ValueError("input must be a non-empty str, list[str], list[int] or list[list[int]]")

        if all(isinstance(item, str) for item in val):
            return values

        if all(isinstance(item, int) for item in val):
            return values

        if all(
            isinstance(item, list) and item and all(isinstance(token, int) for token in item)
            for item in val
        ):
            return values

        raise ValueError("input must be str, list[str], list[int] or list[list[int]]")

    @model_validator(mode="after")
    def validate_dimensions(self):
        if self.dimensions is not None and self.dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        return self


class EmbeddingResponseData(OpenAIBaseModel):
    index: int
    object: str = "embedding"
    embedding: Union[list[float], str]


class EmbeddingResponse(OpenAIBaseModel):
    object: str = "list"
    model: str
    data: list[EmbeddingResponseData]
    usage: UsageInfo


def encode_embedding_base64(vector: list[float]) -> str:
    """Encode an embedding vector as OpenAI-compatible base64 float32 bytes."""
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("utf-8")
