import os
import json
import httpx
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

# 初始化子进程服务
app = FastAPI()

# 子进程绑定的模型和端口
MODEL_NAME = ""
PORT = 0

# ===================== OpenAI 配置 =====================
# OPENAI_API_KEY = "sk-cp-Uc4FLnA4I4-5HQobU3W8lrDUmeXeZUn8J60uzqpGf23YFFfdMrs_jT4l-vuWBO5kZsbRISGWJw2fSQsxQoCuhOcXqqnK61QUPUBAd4yzGz9Q9jYDg2LxfSk"  # 你的 OpenAI Key
# OPENAI_BASE_URL = "https://api.minimaxi.com/v1"  # 可替换为代理地址
# OPENAI_MODEL = "MiniMax-M2.7"  # 实际后台调用的 OpenAI 模型

OPENAI_API_KEY = "sk-sPj9HwI9LlrNFhbm07Ff81AdCfF748E4B7086f7448F5F468"  # 你的 OpenAI Key
OPENAI_BASE_URL = "https://aihubmix.com/v1"  # 可替换为代理地址
# OPENAI_MODEL = "gpt-4o-free"  # 实际后台调用的 OpenAI 模型
OPENAI_MODEL = "gpt-4.1-free"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    对外伪装成本地模型推理
    对内：真实调用 OpenAI API 获取结果
    """
    global MODEL_NAME, PORT

    # 1. 获取用户请求
    req_data = await request.json()
    user_messages = req_data.get("messages", "")
    stream = req_data.get("stream", False)
    request_model = req_data.get("model", "")

    # 2. 安全校验：端口必须匹配模型
    if request_model != MODEL_NAME:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"端口{PORT}只允许模型 {MODEL_NAME}，不允许 {request_model}"
            }
        )

    # 3. 构造 OpenAI 格式请求
    openai_messages = [
        {"role": "user", "content": str(user_messages)}
    ]

    # ===================== 非流式响应 =====================
    if not stream:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url=f"{OPENAI_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": OPENAI_MODEL,
                        "messages": openai_messages,
                        "stream": False
                    },
                    timeout=60.0
                )

            # 解析 OpenAI 返回
            openai_resp = resp.json()
            content = openai_resp["choices"][0]["message"]["content"]

            # 伪装成本地模型返回给用户
            return JSONResponse({
                "model": MODEL_NAME,
                "port": PORT,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop"
                    }
                ],
                "source": "local_model_infer"  # 假装是本地推理
            })
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    # ===================== 流式响应 =====================
    async def stream_generator():
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    url=f"{OPENAI_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": OPENAI_MODEL,
                        "messages": openai_messages,
                        "stream": True
                    },
                    timeout=120.0
                ) as resp:
                    async for line in resp.aiter_lines():
                        print(f"line: {line}")
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                yield "data: [DONE]\n\n"
                                break

                            # 解析 OpenAI 流式片段
                            try:
                                json_data = json.loads(data)
                                delta = json_data["choices"][0]["delta"].get("content", "")
                                if delta:
                                    # 伪装成本地模型流式输出
                                    wrapped = {
                                        "model": MODEL_NAME,
                                        "port": PORT,
                                        "choices": [{"delta": {"content": delta}}]
                                    }
                                    yield f"data: {json.dumps(wrapped, ensure_ascii=False)}\n\n"
                            except:
                                continue
        except Exception as e:
            yield f'data: {{"error": "{str(e)}"}}\n\n'

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )


@app.post("/v1/images/generations")
async def images_generations(request: Request):
    global MODEL_NAME, PORT
    req_data = await request.json()
    request_model = req_data.get("model", "")

    if request_model != MODEL_NAME:
        return JSONResponse(
            status_code=400,
            content={"error": f"端口{PORT}只允许模型 {MODEL_NAME}，不允许 {request_model}"}
        )

    # 目前 OpenAI 图片接口一般无需 stream；如需流式可按相似模式扩展
    stream = req_data.get("stream", False)

    if stream:
        return JSONResponse(status_code=400, content={"error": "图片接口暂不支持stream=true"})

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url=f"{OPENAI_BASE_URL}/images/generations",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENAI_MODEL,
                    "prompt": req_data.get("prompt"),
                    "n": req_data.get("n", 1),
                    "size": req_data.get("size", "512x512"),
                    "response_format": req_data.get("response_format", "b64_json")
                },
                timeout=120.0
            )

        openai_resp = resp.json()

        return JSONResponse(
            {
                "model": MODEL_NAME,
                "port": PORT,
                "data": openai_resp.get("data"),
                "source": "local_model_infer"
            },
            status_code=resp.status_code
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# 子进程实例配置
INSTANCE_CONFIG = {}

def run_worker(model_name: str, port: int, device_ids=None, model_path=None, extra_config=None):
    import uvicorn
    global MODEL_NAME, PORT, INSTANCE_CONFIG
    MODEL_NAME = model_name
    PORT = port
    INSTANCE_CONFIG = {
        "device_ids": device_ids or [],
        "model_path": model_path,
        "extra_config": extra_config or {}
    }

    # 为该子进程设置 GPU 设备
    if device_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_ids)
        os.environ["NVIDIA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_ids)

    print(f"🚀 启动子进程, 模型={MODEL_NAME} 端口={PORT} 设备={device_ids} 路径={model_path}")

    # 通用配置可根据 model_path/extra_config 动态设置
    # 这里可扩展为加载本地模型路径或专用调参

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")