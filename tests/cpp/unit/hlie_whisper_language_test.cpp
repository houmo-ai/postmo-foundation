#include "hlie_app/hlie_ws_service.h"

#include <catch_amalgamated.hpp>

using hm_HLIE::normalize_whisper_language;

TEST_CASE("Whisper language defaults to automatic detection", "[asr][whisper][language]") {
    CHECK(normalize_whisper_language("") == "auto");
    CHECK(normalize_whisper_language(" Auto ") == "auto");
    CHECK(normalize_whisper_language("AUTOMATIC") == "auto");
}

TEST_CASE("Whisper language normalizes common client aliases", "[asr][whisper][language]") {
    CHECK(normalize_whisper_language("zh-CN") == "zh");
    CHECK(normalize_whisper_language("zh_TW") == "zh");
    CHECK(normalize_whisper_language("Mandarin") == "zh");
    CHECK(normalize_whisper_language("EN-us") == "en");
    CHECK(normalize_whisper_language("English") == "en");
    CHECK(normalize_whisper_language("ja") == "ja");
}
