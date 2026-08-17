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
