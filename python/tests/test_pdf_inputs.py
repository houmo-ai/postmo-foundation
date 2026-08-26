import base64

from hliepython.model_server.engines.protocol.chat_completions import (
    ChatCompletionRequest,
)
from hliepython.model_server.engines.text_engine import TextEngine
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


PDF_DATA = "data:application/pdf;base64," + base64.b64encode(b"%PDF-test").decode()


def test_chat_pdf_file_data_survives_vllm_request_conversion():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "glm-ocr",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "识别文档"},
                        {
                            "type": "file",
                            "file": {
                                "filename": "test.pdf",
                                "file_data": PDF_DATA,
                            },
                        },
                    ],
                }
            ],
        }
    )

    engine = TextEngine.__new__(TextEngine)
    engine.model_name = "glm-ocr"
    engine.return_tokens_as_token_ids = False
    converted = engine._to_vllm_chat_request(request)
    part = converted.messages[0]["content"][1]

    assert part["type"] == "file"
    assert part["file"]["file_data"] == PDF_DATA
    assert part["file"]["filename"] == "test.pdf"


def test_responses_pdf_input_file_survives_vllm_request_conversion():
    request = ResponsesRequest.model_validate(
        {
            "model": "glm-ocr",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "识别文档"},
                        {
                            "type": "input_file",
                            "filename": "test.pdf",
                            "file_data": PDF_DATA,
                        },
                    ],
                }
            ],
        }
    )

    engine = TextEngine.__new__(TextEngine)
    engine.model_name = "glm-ocr"
    converted = engine._to_vllm_responses_request(request)
    part = converted.input[0]["content"][1]

    assert part["type"] == "input_file"
    assert part["file_data"] == PDF_DATA
    assert part["filename"] == "test.pdf"