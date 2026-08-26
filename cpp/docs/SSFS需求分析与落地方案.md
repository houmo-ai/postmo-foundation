# SSFS 需求分析与落地方案

## 1. 文档信息

- 文档状态：实施中
- 当前版本：v1.1
- 更新日期：2026-07-28
- 适用工程：hi.cpp / HLIECpp
- 适用范围：本地模型发现、模型身份识别、llama.cpp 个性化加载参数、子进程启动参数传递与运行状态观测

## 2. 背景与目标

hi.cpp 当前能够扫描模型目录、识别 GGUF 模型、注册模型实例，并通过 `LLModelService` 将参数转换为 llama.cpp 的 `common_params`。随着模型种类、模型版本、上下文长度、设备数量和多模态组合持续增加，单一的全局 `[llm_params]` 已无法为所有模型提供稳定且高效的加载参数。

本方案的目标是建立一套可持续维护的模型参数预设体系，使每个模型在加载时可以根据自身身份、用途、版本和硬件条件获得个性化参数，同时满足以下约束：

1. 非 SD 模型不再依赖模型包 `info.json` 进行识别。
2. SD 继续使用 `info.json` 描述多文件组件。
3. 模型预设不进入正常推理请求路径，不影响主流程性能。
4. 主进程和子进程必须使用完全一致的最终参数。
5. 未配置预设时保持现有行为兼容。
6. 已加载模型不因配置文件变化而被动态修改。

## 3. 当前实现分析

### 3.1 当前模型识别

当前模型来源包含两类：

- 模型目录中的 GGUF 文件，通过文件名和 GGUF metadata 生成 `HLIEModelInfo`。
- 模型目录中的 JSON 文件，通过 `HLIEModelInfo::parse_from_file()` 解析。

后续应收紧为：

- LLM、VL、Embedding、Rerank、ASR、ForcedAligner、Whisper、CosyVoice、Qwen3-TTS 等非 SD 模型只允许通过 GGUF 识别。
- `info.json` 仅允许描述 SD 模型。
- 内部生成的 `.hlif` 是子进程启动清单，不属于模型识别文件。

### 3.2 当前 llama.cpp 参数来源

`LLModelService::init()` 当前按照以下方式合并参数：

1. 程序内生成 `--model`、`--parallel`、`--ctx-size` 等基础参数。
2. 合并进程命令行参数。
3. 合并 `sys.cfg` 的 `[llm_params]`。
4. 调用 `common_params_parse()`。
5. 对 Embedding、Rerank、MTP 等模型做额外修正。

现有问题：

- `[llm_params]` 是全局参数，无法针对单个模型调优。
- 当前使用 `unordered_map<string, string>`，参数输出顺序不稳定。
- 无法表达参数删除、布尔关闭和可重复参数。
- `HLIEModelInfo::Params` 当前表示 `0.8B/8B/27B` 等模型规模，不能复用为 llama.cpp 参数。
- 主进程生成子进程时复制了命令行参数，但缺少独立、明确的最终参数快照。

### 3.3 当前模型目录自动发现

模型目录需要支持运行期间发现新下载模型。性能约束如下：

- HTTP 请求线程只允许唤醒后台探测线程，不执行 GGUF 解析。
- 无目录变化时仅读取文件名、文件大小和修改时间。
- 只有目录内容发生变化并经过稳定窗口后，才读取 GGUF metadata。
- `.part`、`.tmp` 等非 `.gguf` 文件天然忽略。
- 对直接以 `.gguf` 写入的文件，需要校验 GGUF metadata 和全部张量数据边界。
- 扫描结果只允许增量注册，不得覆盖已加载模型的状态、端口和请求计数。

## 4. 本机实际模型现状

2026-07-27 通过本机 `http://127.0.0.1:7901/v1/models` 获取到 17 个模型，其中 1 个处于 loaded 状态。

### 4.1 模型分类

| 类别 | 实际模型 | service_kind | model_family / architecture |
| --- | --- | --- | --- |
| 文本 LLM | DeepSeek 8B | llm | qwen3 / qwen3 |
| 文本/多模态 LLM | Qwen3.5 0.8B、2B、9B | llm | qwen3.5 / qwen35 |
| 大模型 | Qwen3.6 27B | llm | qwen3.6 / qwen35 |
| 视觉 | Qwen3-VL 8B | llm | qwen3-vl / qwen3vl |
| OCR | GLM-OCR 0.8B、0.9B | llm | glm-ocr / glm4 |
| 音频理解 | Qwen3-ASR 0.6B | llm | qwen3-asr / qwen3 |
| 音频对齐 | Qwen3-ForcedAligner 0.6B | llm | qwen3-forcealigner / qwen3 |
| 检索 | Qwen3-Reranker 8B | llm | qwen3 / qwen3 |
| ASR 服务 | Whisper medium | whisper | whisper / whisper |
| TTS 服务 | CosyVoice3 0.5B | cosyvoice | cosyvoice3 / cosyvoice3 |
| TTS 服务 | Qwen3-TTS 1.7B | qwen3_tts | qwen3-tts / qwen3_tts |
| 图像生成 | SD3 | sd | sd3 |

当前已加载模型：

```text
HiModel_xh2_qwen3.5_9b_256_32k_b1_1chip_2cores_v1.2.0_20260401
```

### 4.2 当前命名可提取字段

典型名称：

```text
HiModel_xh2_qwen3.5_9b_256_32k_b1_1chip_2cores_v1.2.0_20260401_vit_448_448
```

可提取：

- `xh_version = 2`
- `model_family = qwen3.5`
- `params = 9b`
- `context_size = 32k`
- `batch_num = 1`
- `device_count = 1`
- `cores = 2`
- `model_version = v1.2.0`
- `build_time = 20260401`
- `vl_spec = vit_448_448`

### 4.3 已识别的数据问题

1. DeepSeek 显示名称与底层 `model_family=qwen3` 不一致，不能仅按完整名称判断底层加载参数。
2. Qwen3-Reranker 当前仍表现为普通 qwen3，需要增加 `role=rerank`。
3. Qwen3-ASR 和 ForcedAligner 的 `service_kind=llm`，需要通过 `role` 和音频能力区分。
4. Qwen3.6 的产品模型族为 qwen3.6，但底层 architecture 为 qwen35，匹配时必须同时保留两个维度。
5. Qwen3.5 和 Qwen3-VL 存在不同 `vl_spec`，视觉参数不能只匹配主模型名。
6. Whisper、CosyVoice、Qwen3-TTS 当前返回的 `capabilities.audio=false`，能力字段需要补齐后才能作为可靠匹配条件。
7. 模型名称分隔符存在 `-`、`_` 混用，匹配器不能依赖固定字符串切割。

## 5. 需求定义

### 5.1 功能需求

#### FR-001 非 SD 模型识别

- 非 SD 模型必须通过 GGUF 文件名与 GGUF metadata 识别。
- 模型包 JSON 不再作为非 SD 模型注册入口。
- GGUF metadata 优先于文件名推断；文件名用于补充产品模型族、构建版本和部署规格。

#### FR-002 SD 模型识别

- SD 保留 `info.json`。
- JSON 必须明确包含 `ServiceKind=sd` 或等价的 SD 判定字段。
- 非 SD JSON 应记录警告并忽略，不得注册为模型。

#### FR-003 模型目录自动发现

- 支持周期探测。
- 支持 `GET /models` 和未知模型请求异步唤醒后台探测。
- 支持稳定窗口。
- 支持 GGUF 完整性校验。
- 只增量注册新模型。

#### FR-004 官方模型参数预设

- 提供 `config/model_presets.json`。
- 文件随 hi.cpp 版本发布。
- 支持运行时版本兼容范围。
- 支持按模型身份和硬件条件自动匹配。

#### FR-005 本地模型参数覆盖

- 提供 `config/model_overrides.json`。
- 文件属于部署环境配置，升级软件时不得覆盖。
- 支持按精确 `model_key` 覆盖。
- 支持按模型条件规则覆盖。

#### FR-006 子进程参数一致性

- 主进程负责完成预设匹配和参数合并。
- 主进程生成带最终参数的 `.hlif` 启动清单。
- 子进程只消费最终参数，不重新匹配预设。
- 主子进程最终参数应具有相同 hash。

#### FR-007 参数可观测性

- 日志输出匹配的 preset ID。
- 日志输出参数来源和最终参数摘要。
- `/models` 输出当前 preset ID 和非敏感参数摘要。
- 禁止通过接口暴露密钥、完整文件路径等敏感参数。

### 5.2 非功能需求

#### NFR-001 性能

- 推理请求路径不得读取预设文件。
- 模型列表请求不得同步解析 GGUF。
- 预设目录只在程序初始化时加载一次。
- 预设只在模型加载时解析。
- 已加载模型请求不得因预设解析新增锁竞争。

#### NFR-002 兼容性

- 没有预设文件时维持现有加载行为。
- 现有 `[llm_params]` 继续生效。
- 旧版 `--json_info` 可作为子进程启动参数的兼容别名。
- 旧版非 SD `info.json` 不再参与发现，但应给出明确迁移日志。

#### NFR-003 确定性

- 相同模型身份、相同硬件条件和相同配置必须得到相同参数集合。
- 参数合并顺序必须稳定。
- 规则冲突必须在配置校验阶段发现，不能依赖 JSON 顺序或哈希表顺序。

#### NFR-004 安全性

- 预设不能修改框架拥有的模型路径、端口和进程控制参数。
- 配置中的未知参数必须明确报错。
- 官方预设必须在 CI 中针对当前 vendored llama.cpp 参数解析器完成验证。

## 6. 总体架构

```text
模型目录
  ├─ *.gguf ───────────────→ 非 SD 模型身份识别
  └─ info.json(ServiceKind=sd) → SD 模型描述

模型注册表（只保存模型身份和运行状态）
  ↓ 模型加载
模型身份 + 设备条件
  ↓
HLIELlmPresetCatalog + HLIELlmOverrideCatalog
  ↓
HLIELlmPresetResolver
  ↓
HLIELlmParamSet（最终参数 + 来源追踪）
  ├─ 本进程 LLModelService
  └─ HLIEModelLaunchManifest(.hlif) → 子进程 LLModelService
```

### 6.1 职责边界

#### `HLIEModelInfo`

只保存稳定的模型身份和模型包信息：

- 模型路径与模型名
- service kind
- role
- model family
- architecture
- 参数规模
- 上下文规格
- 模型版本和构建时间
- XH 版本、设备数量
- 多模态能力和 `vl_spec`

不保存官方预设库，也不承担全局参数合并。

#### `HLIEModelRegistry`

- 保存已发现模型和运行状态。
- 不读取预设文件。
- 不解析 llama.cpp 参数。
- 注册表中的模型状态不能被目录重扫覆盖。

#### `HLIELlmPresetCatalog`

- 加载官方预设。
- 校验 schema、版本范围、参数合法性和规则冲突。
- 加载后保持只读。

#### `HLIELlmOverrideCatalog`

- 加载用户和部署环境覆盖。
- 与官方预设分离，避免软件升级覆盖用户配置。

#### `HLIELlmPresetResolver`

- 输入模型身份、硬件上下文、全局配置。
- 输出最终参数集合、匹配规则、覆盖来源和参数 hash。
- 不持有模型服务生命周期。

#### `LLModelService`

- 消费最终参数。
- 构建稳定 argv。
- 调用 `common_params_parse()`。
- 执行 Embedding、Rerank、MTP 等最终安全修正。
- 不在内部重复读取预设文件。

#### `HLIEModelLaunchManifest`

- 表达一次确定的模型启动。
- 用于主进程到子进程的内部协议。
- 与 SD `info.json`、模型发现 JSON 完全区分。

## 7. 模型身份结构

建议补充统一身份结构：

```cpp
enum class HLIEModelRole {
    Chat,
    VisionLanguage,
    Ocr,
    Asr,
    ForcedAligner,
    Embedding,
    Rerank,
    Tts,
    ImageGeneration,
    Unknown,
};

struct HLIEModelIdentity {
    std::string model_key;
    std::string model_name;
    std::string service_kind;
    HLIEModelRole role;
    std::string model_family;
    std::string architecture;
    std::string param_scale;
    std::string quantization;
    std::string model_version;
    std::string build_time;
    std::string vl_spec;
    int context_size = 0;
    int batch_num = 1;
    int xh_version = 0;
    int device_count = 1;
    int cores = 0;
    bool multimodal = false;
    HLIEModelCapabilities capabilities;
};
```

身份来源优先级：

1. GGUF metadata。
2. 已注册的模型族规则。
3. 文件名部署规格。
4. 保守默认值。

产品模型族和底层 architecture 必须分别保存，不能互相覆盖。

## 8. 参数数据模型

### 8.1 参数类型

```cpp
using HLIELlmParamValue = std::variant<
    std::monostate,
    bool,
    int64_t,
    double,
    std::string,
    std::vector<std::string>>;

using HLIELlmParamSet = std::map<std::string, HLIELlmParamValue>;
```

JSON 语义：

| JSON 类型 | 语义 |
| --- | --- |
| string / number | 带值参数 |
| `true` | 启用无值开关 |
| `false` / `null` | 删除或禁用继承参数 |
| array | 可重复参数 |

示例：

```json
{
  "--ctx-size": 32768,
  "--parallel": 1,
  "--no-warmup": true,
  "--flash-attn": false,
  "--lora": ["adapter-a.gguf", "adapter-b.gguf"]
}
```

### 8.2 框架拥有参数

以下参数禁止出现在 preset 和 override 中：

- `--model`
- `--mmproj`
- `--host`
- `--port`
- `--device_id`
- `--instance_id`
- `--json_info`
- `--launch_manifest`
- API Key、日志路径和其他进程级安全参数

路径型可选参数，如 LoRA 文件，需要独立的路径校验策略。

## 9. 官方预设文件

文件：`config/model_presets.json`

```json
{
  "schema_version": 1,
  "catalog_version": "2.1.1",
  "runtime": {
    "hlie_min_version": "2.1.0",
    "hlie_max_version": "2.1.x"
  },
  "presets": [
    {
      "id": "llm-xh2-default",
      "priority": 10,
      "match": {
        "service_kind": "llm",
        "xh_version": 2
      },
      "params": {
        "--parallel": 1,
        "--no-context-shift": true,
        "--no-warmup": true
      }
    },
    {
      "id": "qwen35-9b-xh2-1chip",
      "priority": 200,
      "match": {
        "role": "chat",
        "model_family": "qwen3.5",
        "architecture": "qwen35",
        "params": "9b",
        "xh_version": 2,
        "device_count": 1
      },
      "params": {
        "--ctx-size": 32768,
        "--batch-size": 1024,
        "--ubatch-size": 256
      }
    }
  ]
}
```

第一版匹配字段：

- `model_key`
- `model_name`
- `service_kind`
- `role`
- `model_family`
- `architecture`
- `params`
- `xh_version`
- `device_count`
- `context_size`
- `batch_num`
- `multimodal`
- `vl_spec`
- `model_version_min`
- `model_version_max`

量化类型和设备显存档位在身份字段稳定后加入。

## 10. 本地覆盖文件

文件：`config/model_overrides.json`

```json
{
  "schema_version": 1,
  "models": {
    "HiModel_xh2_qwen3.5_9b_256_32k_b1_1chip_2cores_v1.2.0_20260401": {
      "params": {
        "--ctx-size": 65536
      }
    }
  },
  "rules": [
    {
      "id": "local-qwen35-two-chip",
      "priority": 1000,
      "match": {
        "model_family": "qwen3.5",
        "device_count": 2
      },
      "params": {
        "--parallel": 2
      }
    }
  ]
}
```

要求：

- 官方安装包可提供空模板。
- 升级流程不得覆盖用户已有文件。
- 配置错误应指出规则 ID 和具体参数。

## 11. 匹配与合并算法

### 11.1 规则匹配

1. 校验 preset 与当前 HLIECpp 版本兼容。
2. 规则中声明的所有 match 字段必须同时满足。
3. 未声明字段不参与判断。
4. 字符串采用规范化后的精确匹配；仅 `model_name` 可选支持 glob。
5. 不在第一版支持任意正则表达式，避免配置复杂度和性能风险。

### 11.2 稳定排序

匹配成功的规则按照以下顺序合并：

1. `priority` 从低到高。
2. match 字段数量从少到多。
3. preset ID 字典序。

同一优先级、相同匹配精度的规则若对同一参数给出不同值，应在 catalog 校验阶段报冲突。

### 11.3 参数覆盖顺序

最终覆盖顺序：

1. llama.cpp 默认值。
2. hi.cpp 安全默认值。
3. 官方通用 preset。
4. 官方具体模型 preset。
5. 本地规则 override。
6. 精确 `model_key` override。
7. 进程命令行参数。
8. `sys.cfg [llm_params]` 管理员强制覆盖。
9. hi.cpp 最终安全修正。

保留 `sys.cfg` 的最高管理员覆盖能力，以维持当前行为兼容。

## 12. 子进程启动清单

### 12.1 协议定位

`.hlif` 是内部 Launch Manifest，不是模型包 `info.json`。

建议新增：

```cpp
struct HLIEModelLaunchManifest {
    int schema_version = 1;
    HLIEModelInfo model_info;
    HLIELlmParamSet effective_llama_params;
    std::vector<std::string> matched_presets;
    std::string effective_params_hash;
    std::string resolved_by_version;
};
```

### 12.2 JSON 示例

```json
{
  "Schema": "hlie.model-launch",
  "SchemaVersion": 1,
  "Model": {
    "ModelPath": "HiModel_xh2_qwen3.5_9b_256_32k_b1_1chip_2cores_v1.2.0_20260401.gguf",
    "ServiceKind": "llm",
    "Devices": "0"
  },
  "Runtime": {
    "ResolvedBy": "HLIECpp-2.1.1",
    "MatchedPresets": ["llm-xh2-default", "qwen35-9b-xh2-1chip"],
    "EffectiveParamsHash": "sha256:...",
    "LlamaParams": {
      "--ctx-size": 32768,
      "--parallel": 1,
      "--batch-size": 1024,
      "--ubatch-size": 256
    }
  }
}
```

### 12.3 命令行迁移

目标参数：

```text
--launch_manifest <path>
```

兼容期保留：

```text
--json_info <path>
```

当文件 schema 为 `hlie.model-launch` 时按新协议解析；旧格式继续按当前 `HLIEModelInfo` 解析。

## 13. 加载生命周期

```text
程序初始化
  → 加载并校验 model_presets.json
  → 加载并校验 model_overrides.json
  → 扫描 GGUF 和 SD info.json
  → 注册原始模型身份

模型加载
  → 解析模型身份和 role
  → 确定设备数量与部署条件
  → 匹配官方 preset
  → 合并本地 override、CLI、sys.cfg
  → 生成最终参数与 hash
  → 本进程直接加载，或写入 .hlif 启动子进程
  → common_params_parse
  → 安全修正
  → 模型加载完成
```

预设文件修改后：

- 不修改已加载模型。
- 仅在下一次模型加载时生效。
- 第一版不做热加载。
- 后续若增加热加载，也只能刷新 catalog 快照，不能修改运行实例。

## 14. 可观测性设计

### 14.1 日志

模型加载时输出：

```text
model_key=...
model_role=chat
matched_presets=[llm-xh2-default,qwen35-9b-xh2-1chip]
override_sources=[official,model_override,sys_cfg]
effective_params_hash=...
```

Debug 日志可输出完整非敏感参数。Info 日志只输出摘要和 hash。

### 14.2 `/models`

建议在 `loaded_info` 中增加：

```json
{
  "role": "chat",
  "llama_presets": ["llm-xh2-default", "qwen35-9b-xh2-1chip"],
  "llama_params_hash": "sha256:...",
  "llama_params_summary": {
    "ctx_size": 32768,
    "parallel": 1,
    "batch_size": 1024,
    "ubatch_size": 256
  }
}
```

未加载模型可以显示“预计匹配 preset”，但不得在每次 `/models` 请求中重新解析配置；应使用注册或 catalog 加载时生成的缓存结果。

## 15. 代码落点

### 15.1 新增文件

```text
include/hlie_app/hlie_llm_param_set.h
src/hlie_app/hlie_llm_param_set.cpp

include/hlie_app/hlie_llm_preset_catalog.h
src/hlie_app/hlie_llm_preset_catalog.cpp

include/hlie_app/hlie_llm_override_catalog.h
src/hlie_app/hlie_llm_override_catalog.cpp

include/hlie_app/hlie_llm_preset_resolver.h
src/hlie_app/hlie_llm_preset_resolver.cpp

include/hlie_app/hlie_model_launch_manifest.h
src/hlie_app/hlie_model_launch_manifest.cpp

config/model_presets.json
config/model_overrides.json
```

新增 `.cpp` 后必须重新执行 CMake configure，因为 `src/hlie_base/CMakeLists.txt` 等位置存在 `file(GLOB ...)`。

### 15.2 修改文件

#### `hlie_model_info.h/.cpp` 与 `hlie_model_info_core.cpp`

- 增加 `role`。
- 明确 `param_scale` 与 llama params 的语义区别。
- 补充 quantization、上下文、能力字段读取。
- 产品模型族与 architecture 分开保存。

#### `hlie_workflow.cpp`

- 将 JSON 目录扫描改为只识别 SD manifest。
- 接入 preset catalog 生命周期。
- 模型加载前调用 resolver。
- 生成新 `.hlif` Launch Manifest。
- 子进程入口同时支持 `--launch_manifest` 和旧 `--json_info`。

#### `hlie_llm_model_service.cpp`

- 移除内部无序参数合并。
- 接收或读取已解析的 `HLIELlmParamSet`。
- 使用稳定 argv 顺序调用 `common_params_parse()`。
- 保留最终安全修正。

#### `hlie_model_registry.cpp`

- 在 `loaded_info` 中记录 role、preset 和参数 hash。
- 不在注册表内执行 preset 文件 IO。

#### `config_schema.json`

- 增加 preset 和 override 文件位置、strict 模式等配置。

建议配置：

```ini
[model]
llm_preset_catalog = ./config/model_presets.json
llm_override_catalog = ./config/model_overrides.json
llm_preset_strict = true
```

## 16. 分阶段实施计划

### 阶段 0：行为基线

- 固化当前 7901 `/v1/models` 模型快照测试。
- 固化当前无 preset 时的有效 llama 参数。
- 为主进程和子进程加载建立回归用例。

交付物：基线测试和模型样本清单。

### 阶段 1：模型识别边界收敛

- JSON 扫描仅接受 SD。
- 为非 SD JSON 输出迁移警告。
- 增加 `HLIEModelRole`。
- 修正 Rerank、ASR、ForcedAligner、OCR、VL、TTS role。
- 修正音频模型 capabilities。

交付物：稳定的模型身份结构和 `/models` 输出。

### 阶段 2：参数基础设施

- 实现 `HLIELlmParamSet`。
- 实现参数覆盖、删除、重复值和稳定 argv。
- 实现框架参数 denylist。
- 建立参数序列化和 hash。

交付物：参数数据模型与单元测试。

### 阶段 3：Preset Resolver

- 实现官方 catalog。
- 实现本地 override catalog。
- 实现版本校验、匹配、排序、冲突检测和来源追踪。
- 接入配置 schema。

交付物：可独立测试的 resolver。

### 阶段 4：模型加载接入

- 在模型加载前解析最终参数。
- 重构 `LLModelService` 参数合并。
- 保持 CLI 和 `[llm_params]` 兼容。
- 确保现有 Embedding、Rerank、MTP 修正逻辑不回退。

交付物：本进程个性化加载。

### 阶段 5：子进程协议升级

- 引入 `HLIEModelLaunchManifest`。
- 主进程写入最终参数和 hash。
- 子进程只消费最终参数。
- 增加 `--launch_manifest`，兼容 `--json_info`。

交付物：主子进程参数一致性。

### 阶段 6：首批预设与性能验证

首批建议覆盖：

- Qwen3.5 0.8B / 2B / 9B
- Qwen3.6 27B 两卡
- Qwen3-VL 8B
- GLM-OCR 0.8B / 0.9B
- Qwen3-ASR / ForcedAligner
- Qwen3-Reranker 8B

具体参数值必须通过目标硬件基准测试确定，不应仅根据名称推测。

交付物：经验证的 `model_presets.json` 初始版本。

## 17. 测试方案

### 17.1 单元测试

- 参数 JSON 类型解析。
- 参数覆盖和删除。
- 重复参数 argv 生成。
- denylist 拒绝。
- 模型身份匹配。
- 版本范围匹配。
- priority 与 specificity 排序。
- 同优先级冲突检测。
- 参数 hash 稳定性。
- Launch Manifest 新旧格式兼容。

### 17.2 集成测试

- 无 preset 文件时保持旧行为。
- Qwen3.5 9B 命中专属 preset。
- Reranker 不命中 Chat preset。
- Qwen3-ASR 命中音频 LLM preset。
- `vit_448_448` 与 `vit_896_896` 命中不同视觉规则。
- 精确 model override 覆盖官方 preset。
- `sys.cfg [llm_params]` 覆盖本地 override。
- 主进程和子进程参数 hash 一致。
- 非 SD `info.json` 不再注册模型。
- SD `info.json` 正常加载。

### 17.3 性能测试

- `/v1/models` 请求不发生 preset 文件 IO。
- 正常推理路径不增加 preset resolver 调用。
- 空闲模型发现只读取目录 metadata。
- 目录无变化时 CPU 与磁盘开销接近零。
- 预设解析只发生在模型加载阶段。

### 17.4 回归测试

- Release 构建通过。
- 全部现有单测通过。
- 默认模型启动成功。
- 自动加载和自动卸载正常。
- 多模型子进程启动、状态回报和卸载正常。
- PDF 文档管线随 LLM 生命周期加载和卸载正常。

## 18. 验收标准

1. 非 SD 模型不再通过模型目录 JSON 注册。
2. SD `info.json` 行为保持正常。
3. 7901 当前模型可全部通过 GGUF 或 SD manifest 识别。
4. 模型 role、family、architecture、vl_spec 等身份字段正确。
5. 不同模型可以命中不同 preset。
6. 未配置 preset 时有效参数与当前版本一致。
7. 本地 override 和 `sys.cfg` 覆盖顺序符合设计。
8. 主子进程最终参数 hash 完全一致。
9. 非法、冲突或当前 llama.cpp 不支持的参数可明确定位。
10. 已加载模型不受 catalog 文件变化影响。
11. 推理请求路径不读取 catalog 文件。
12. `/v1/models` 不同步解析 GGUF 和 preset 文件。
13. Release 构建和全部测试通过。

## 19. 风险与应对

### 19.1 模型名称不规范

风险：分隔符、产品名和底层 architecture 不一致。

应对：metadata 优先、模型族规则补充、禁止仅靠完整名称通配完成通用匹配。

### 19.2 预设与 llama.cpp 版本不兼容

风险：vendored llama.cpp 参数发生变化导致加载失败。

应对：catalog 声明运行时版本范围，CI 对全部官方 preset 执行参数解析验证。

### 19.3 预设导致显存不足

风险：上下文、batch 或并行度过大。

应对：按设备数量分层预设，保留现有内存估算和自动驱逐；后续加入设备显存档位匹配。

### 19.4 主子进程参数漂移

风险：子进程重新读取配置后产生不同结果。

应对：主进程生成不可变 Launch Manifest，子进程不重新解析 preset。

### 19.5 用户配置被升级覆盖

风险：本地调优丢失。

应对：官方 preset 与用户 override 分文件，打包和升级流程只更新官方文件。

## 20. 回滚策略

- 增加 `model.llm_presets_enabled=false` 总开关。
- 关闭后恢复当前程序默认值、CLI 和 `[llm_params]` 合并方式。
- Launch Manifest 中保留兼容字段，旧子进程入口仍可使用 `--json_info`。
- preset 解析失败时，非 strict 模式允许回退旧行为；生产发布的官方 catalog 建议使用 strict 模式。

## 21. 实施结论

本方案可以开始落地，推荐按“身份边界收敛 → 参数基础设施 → Resolver → LLModelService 接入 → 子进程协议 → 性能预设”顺序实施。

在首批代码提交中不应直接填写未经基准测试的所谓最佳参数。第一阶段应先保证模型身份可靠、匹配逻辑确定、覆盖顺序可解释、主子进程参数一致。随后基于实际硬件测试结果逐步完善官方 `model_presets.json`。

## 22. 实施进展记录

### 22.1 2026-07-28 第一批实现

已完成：

- 非 SD JSON 不再参与模型目录识别，SD `info.json` 保持有效。
- 增加 Chat、VisionLanguage、OCR、ASR、ForcedAligner、Embedding、Rerank、TTS、ImageGeneration role。
- 修正 Whisper、CosyVoice、Qwen3-TTS、Qwen3-ASR、ForcedAligner 等音频能力推断。
- 增加稳定、有序的 `HLIELlmParamSet`。
- 支持字符串、数字、布尔开关、继承删除和重复参数。
- 增加框架拥有参数 denylist。
- 增加官方 preset catalog、本地 override catalog、版本约束、字段校验和冲突检测。
- 增加确定性规则排序、参数来源追踪和稳定 hash。
- Workflow 在模型加载阶段合并 preset、override、CLI 和 `[llm_params]`。
- `LLModelService` 优先消费已解析参数，并保留直接调用兼容回退。
- `.hlif` 增加 `hlie.model-launch` schema、最终参数、preset 和 hash。
- 新子进程入口使用 `--launch_manifest`，保留 `--json_info` 兼容入口。
- `/v1/models` 增加 role、preset、参数 hash 和非敏感摘要。
- Windows/Linux 打包脚本增加官方 `model_presets.json`。

验证结果：

- Release 主程序构建通过。
- 单元测试 40 个用例、250 个断言通过。
- 使用 7911 隔离实例扫描 `E:\models`，17 个模型全部正常注册。
- role 输出覆盖 `asr/chat/forced_aligner/image_generation/ocr/rerank/tts/vision_language`。
- 使用本地 smoke override 加载 GLM-OCR 0.8B 成功。
- `/v1/models` 正确返回 `runtime-ocr-smoke`、参数 hash、`ctx_size=2048`、`batch_size=256`、`parallel=1`。
- 模型健康检查与卸载成功。

尚未完成：

- 根据正式基准测试填写首批生产级最佳参数。
- 增加 quantization 和设备显存档位身份字段。
- 对主子进程实际双实例参数 hash 一致性进行专项集成测试。
- 完成 Linux Release 和安装升级不覆盖本地 override 的打包验证。
