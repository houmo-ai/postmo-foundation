import json

from hliepython.config.app_config import AppConfig, ModelConfig, ModelInstanceConfig
from hliepython import main


MAIN_MODEL = (
    "HiModel_xh2_qwen3.5-0.8b_w8a16_256_256k_b1_1chip_2cores_"
    "v1.4.0_20260708.gguf"
)
MMPROJ_MODEL = (
    "mmproj_xh2_qwen3.5-0.8b_w8a16_2cores_vit_448x448_"
    "v1.4.0_20260708.gguf"
)
MULTIMODAL_MODEL = (
    "HiModel_xh2_qwen3.5-0.8b_w8a16_256_256k_b1_1chip_2cores_"
    "v1.4.0_20260708_vit_448_448.gguf"
)
FORCED_ALIGNER_MODEL = (
    "HiModel_xh2_qwen3-forcealigner-0.6b_2k_1chip_2cores_"
    "v1.4.0_20260721.gguf"
)
FORCED_ALIGNER_MMPROJ = (
    "mmproj_xh2_qwen3-forcealigner-0.6b_2cores_v1.4.0_20260721.gguf"
)


def test_scan_pairs_forcealigner_model_with_mmproj(tmp_path, monkeypatch):
    main_model_path = tmp_path / FORCED_ALIGNER_MODEL
    mmproj_path = tmp_path / FORCED_ALIGNER_MMPROJ
    main_model_path.touch()
    mmproj_path.touch()
    runtime_dir = tmp_path / ".gguf_model_dirs" / main_model_path.stem
    monkeypatch.setattr(main, "_ensure_gguf_runtime_dir", lambda _: runtime_dir)

    model_paths = main.scan_available_models(str(tmp_path))

    assert model_paths[FORCED_ALIGNER_MODEL] == str(runtime_dir)
    assert main.AVAILABLE_MODEL_ADDITIONAL_CONFIG[FORCED_ALIGNER_MODEL] == {
        "vit_model_name": str(mmproj_path)
    }
    assert main.AVAILABLE_MODEL_BASE_NAMES[FORCED_ALIGNER_MODEL] == FORCED_ALIGNER_MODEL


def test_loaded_mmproj_model_replaces_main_model_in_model_list(monkeypatch):
    instance = ModelInstanceConfig(
        port=7801,
        model_path=f"/models/.gguf_model_dirs/{MAIN_MODEL.removesuffix('.gguf')}",
        additional_config={"vit_model_name": f"/models/{MMPROJ_MODEL}"},
    )
    app_config = AppConfig(
        models=[ModelConfig(name=MAIN_MODEL, model_type="llm", instances=[instance])]
    )

    monkeypatch.setattr(main, "APP_CONFIG", app_config)
    monkeypatch.setattr(main, "MODEL_PORT_MAP", {MAIN_MODEL: [7801]})
    monkeypatch.setattr(
        main,
        "AVAILABLE_MODEL_PATHS",
        {
            MAIN_MODEL: instance.model_path,
            MULTIMODAL_MODEL: instance.model_path,
        },
    )
    monkeypatch.setattr(
        main,
        "AVAILABLE_MODEL_BASE_NAMES",
        {MULTIMODAL_MODEL: MAIN_MODEL},
    )

    records = main._build_model_records()

    assert [(record["id"], record["port"], record["status"]) for record in records] == [
        (MULTIMODAL_MODEL, 7801, 1)
    ]


def test_non_multimodal_model_name_is_unchanged(monkeypatch):
    model_name = "text-model"
    instance = ModelInstanceConfig(port=7802, model_path="/models/text-model")
    app_config = AppConfig(
        models=[ModelConfig(name=model_name, model_type="llm", instances=[instance])]
    )

    monkeypatch.setattr(main, "APP_CONFIG", app_config)
    monkeypatch.setattr(main, "MODEL_PORT_MAP", {model_name: [7802]})
    monkeypatch.setattr(main, "AVAILABLE_MODEL_PATHS", {model_name: instance.model_path})
    monkeypatch.setattr(main, "AVAILABLE_MODEL_BASE_NAMES", {})

    records = main._build_model_records()

    assert [(record["id"], record["port"], record["status"]) for record in records] == [
        (model_name, 7802, 1)
    ]


def test_multimodal_model_list_alias_is_an_instance_request_name():
    instance_cfg = {
        "vllm_config": {},
        "additional_config": {"vit_model_name": f"/models/{MMPROJ_MODEL}"},
    }

    request_model_names = main._get_instance_request_model_names(
        MAIN_MODEL,
        instance_cfg,
    )

    assert MAIN_MODEL in request_model_names
    assert MULTIMODAL_MODEL in request_model_names


def test_all_served_model_names_are_instance_request_names():
    instance_cfg = {
        "vllm_config": {
            "serving": {"served_model_name": ["public-model-a", "public-model-b"]}
        },
        "additional_config": {},
    }

    request_model_names = main._get_instance_request_model_names(MAIN_MODEL, instance_cfg)

    assert request_model_names == [MAIN_MODEL, "public-model-a", "public-model-b"]


def test_non_stream_response_uses_requested_public_model_name():
    response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": MAIN_MODEL,
        "choices": [],
    }

    rewritten = main._rewrite_response_model(response, MULTIMODAL_MODEL)

    assert rewritten["model"] == MULTIMODAL_MODEL


def test_stream_response_uses_requested_public_model_name():
    upstream_chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": MAIN_MODEL,
        "choices": [],
    }
    line = f"data: {json.dumps(upstream_chunk)}\r\n".encode()

    rewritten_line = main._rewrite_sse_model_line(line, MULTIMODAL_MODEL)
    rewritten_chunk = json.loads(rewritten_line[len(b"data: "):])

    assert rewritten_chunk["model"] == MULTIMODAL_MODEL
    assert rewritten_line.endswith(b"\r\n")


def test_stream_done_marker_is_unchanged():
    done = b"data: [DONE]\n"

    assert main._rewrite_sse_model_line(done, MULTIMODAL_MODEL) == done


def test_unload_worker_removes_port_from_all_aliases(monkeypatch):
    monkeypatch.setattr(
        main,
        "MODEL_PORT_MAP",
        {
            MAIN_MODEL: [7801, 7802],
            MULTIMODAL_MODEL: [7801, 7802],
            "public-model": [7801, 7802],
        },
    )
    monkeypatch.setattr(main, "PORT_MODEL_MAP", {7801: MAIN_MODEL, 7802: MAIN_MODEL})
    monkeypatch.setattr(main, "WORKER_PROCESS_MAP", {})
    monkeypatch.setattr(main, "DYNAMIC_MODEL_INSTANCES", {})
    monkeypatch.setattr(main, "_unregister_worker_runtime", lambda port: None)
    monkeypatch.setattr(
        main.MODEL_LOAD_MANAGER,
        "unregister_instance",
        lambda model_name, port: None,
    )

    unloaded = main.unload_worker(port=7801)

    assert unloaded == [{"model": MAIN_MODEL, "port": 7801}]
    assert main.MODEL_PORT_MAP == {
        MAIN_MODEL: [7802],
        MULTIMODAL_MODEL: [7802],
        "public-model": [7802],
    }
