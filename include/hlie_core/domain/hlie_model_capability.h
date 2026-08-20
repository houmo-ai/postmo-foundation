#pragma once

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

struct HLIEModelCapabilities {
    bool text = true;
    bool vision = false;
    bool audio = false;
};

enum class HLIEModelRole {
    Unknown,
    Chat,
    VisionLanguage,
    Ocr,
    Asr,
    ForcedAligner,
    Embedding,
    Rerank,
    Tts,
    ImageGeneration,
};

enum class HLIERuntimeKind {
    Embedded,
    Process,
};

HLIE_NAMESPACE_END
