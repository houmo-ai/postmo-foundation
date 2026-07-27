#include "hlie_app/hlie_document_pipeline_service.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <utility>

#include "gguf.h"
#include "hlie_app/hlie_document_core.h"
#include "hlie_app/hlie_document_image_encoder.h"
#include "hlie_app/hlie_document_input_adapter.h"
#include "hlie_app/hlie_document_layout_engine.h"
#include "hlie_app/hlie_document_region_cropper.h"
#include "hlie_app/hlie_document_types.h"
#include "hlie_app/hlie_llm_model_service.h"
#include "hlie_app/hlie_model_info.h"
#include "hlie_app/hlie_pdf_renderer.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_singleton.h"
#include "hlie_base/hlie_util.h"
#include "json.hpp"

HLIE_NAMESPACE_BEGIN

struct DocumentPipelineService::PreprocessContext {
    PdfRenderOptions render_options;
    DocumentPreprocessResult result;
    int total_page_count = 0;
    int crop_count = 0;
    bool include_page_margins = false;
};

namespace {
    std::string trim_ascii(std::string value) {
        const auto first = std::find_if_not(value.begin(), value.end(), [](const unsigned char ch) {
            return std::isspace(ch);
        });
        const auto last = std::find_if_not(value.rbegin(), value.rend(), [](const unsigned char ch) {
            return std::isspace(ch);
        }).base();
        if (first >= last) {
            return {};
        }
        return {first, last};
    }

    std::string extract_chat_text(const server_http_res_ptr& response) {
        if (!response) {
            return {};
        }

        std::string data = response->data;
        if (response->is_stream()) {
            std::string chunk;
            while (response->next(chunk)) {
                data += chunk;
                chunk.clear();
            }
        }

        const auto body = json::parse(data, nullptr, false);
        if (body.is_discarded()) {
            return trim_ascii(data);
        }
        if (body.contains("choices") && body["choices"].is_array() && !body["choices"].empty()) {
            const auto& choice = body["choices"][0];
            if (choice.contains("message") && choice["message"].contains("content") &&
                choice["message"]["content"].is_string()) {
                return trim_ascii(choice["message"]["content"].get<std::string>());
            }
            if (choice.contains("text") && choice["text"].is_string()) {
                return trim_ascii(choice["text"].get<std::string>());
            }
        }
        if (body.contains("output_text") && body["output_text"].is_string()) {
            return trim_ascii(body["output_text"].get<std::string>());
        }
        if (body.contains("text") && body["text"].is_string()) {
            return trim_ascii(body["text"].get<std::string>());
        }
        if (body.contains("error")) {
            LOG_WARN("Document OCR region recognition returned error: {}", body.dump());
            return {};
        }
        return trim_ascii(body.dump());
    }

    bool json_boolish_value(const json& value) {
        if (value.is_boolean()) {
            return value.get<bool>();
        }
        if (value.is_number_integer()) {
            return value.get<int>() != 0;
        }
        if (value.is_string()) {
            auto text = value.get<std::string>();
            std::transform(text.begin(), text.end(), text.begin(), [](const unsigned char ch) {
                return static_cast<char>(std::tolower(ch));
            });
            return text == "1" || text == "true" || text == "yes" || text == "on";
        }
        return false;
    }

    bool request_boolish(const server_http_req& req, const std::string& key, const bool fallback = false) {
        const auto param = req.get_param(key.c_str());
        if (!param.empty()) {
            return json_boolish_value(param);
        }
        if (!req.body.empty()) {
            const auto body = json::parse(req.body, nullptr, false);
            if (body.is_object() && body.contains(key)) {
                return json_boolish_value(body.at(key));
            }
        }
        return fallback;
    }

    std::string prompt_for_region(const DocumentLayoutRegion& region) {
        if (region.task_type == "table" || region.label == "table") {
            return "Recognize the table in this image. Return only the table content in Markdown, without explanations.";
        }
        if (region.task_type == "formula") {
            return "Recognize the formula in this image. Return only the formula text, without explanations.";
        }
        if (region.class_id < 0 || region.score <= 0.21f) {
            return "Transcribe only clearly readable text in this image. If there is no clearly readable text, return an empty string. Do not describe the image.";
        }
        return "Recognize all visible text in this image. Return only the transcribed text, without explanations.";
    }

    json build_region_chat_body(const std::string& model, const DocumentLayoutRegion& region) {
        return {
            {"model", model},
            {"stream", false},
            {"temperature", 0},
            {"max_tokens", region.task_type == "table" ? 1024 : 512},
            {"messages", json::array({
                {
                    {"role", "user"},
                    {"content", json::array({
                        {
                            {"type", "text"},
                            {"text", prompt_for_region(region)},
                        },
                        {
                            {"type", "image_url"},
                            {"image_url", {
                                {"url", DocumentImageEncoder::crop_to_png_data_url(region.crop)},
                            }},
                        },
                    })},
                },
            })},
        };
    }

} // namespace

bool DocumentPipelineService::gguf_has_tensor(const std::string& path, const std::string& tensor_name) {
    if (path.empty()) {
        return false;
    }

    ggml_context* ctx_data = nullptr;
    gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_data,
    };

    gguf_context* ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) {
        LOG_WARN("Document pipeline failed to open mmproj gguf: {}", path);
        return false;
    }

    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    bool found = false;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(ctx, i);
        if (name != nullptr && tensor_name == name) {
            found = true;
            break;
        }
    }

    gguf_free(ctx);
    return found;
}

std::string DocumentPipelineService::resolve_mmproj_path(const std::string& model_dir, const HLIEModelInfo& model_info) {
    auto mmproj_path = model_info.get_mm_proj_path();
    if (!mmproj_path.empty() && HlieUtil::is_relative_path(mmproj_path)) {
        mmproj_path = HlieUtil::join_paths(model_dir, mmproj_path);
    }
    return mmproj_path;
}

bool DocumentPipelineService::supports_document_layout(const std::string& model_dir, const HLIEModelInfo& model_info) {
    return gguf_has_tensor(resolve_mmproj_path(model_dir, model_info), "ppdoclayoutv3.hmm");
}

int DocumentPipelineService::init(const std::string& model_dir, const HLIEModelInfo& model_info,
                                  const std::shared_ptr<HLIEDeviceService>& device_service) {
    device_service_ = device_service;
    model_key_ = model_info.model_key_.empty() ? HLIEModelInfo::build_model_key(model_info) : model_info.model_key_;
    mmproj_path_ = resolve_mmproj_path(model_dir, model_info);

    if (mmproj_path_.empty()) {
        LOG_INFO("Document pipeline disabled for {}, no mmproj configured", model_key_);
        return -1;
    }

    if (!gguf_has_tensor(mmproj_path_, "ppdoclayoutv3.hmm")) {
        LOG_INFO("Document pipeline disabled for {}, ppdoclayoutv3.hmm not found in {}", model_key_, mmproj_path_);
        return -1;
    }

    layout_engine_ = std::make_unique<DocumentLayoutEngine>();
    if (!layout_engine_->init(mmproj_path_, {0})) {
        LOG_ERROR("Document pipeline failed to initialize layout engine for {}", model_key_);
        layout_engine_.reset();
        return -1;
    }

    state_.store(SERVER_STATE_LOADING_MODEL);
    LOG_INFO("Document pipeline initialized for {}, mmproj={}", model_key_, mmproj_path_);
    return 0;
}

void DocumentPipelineService::run() {
    state_.store(SERVER_STATE_READY);
    is_ready_.store(true);
    LOG_INFO("Document pipeline state: {}", state_.load());
}

void DocumentPipelineService::process() {
}

void DocumentPipelineService::stop() {
    is_ready_.store(false);
    if (layout_engine_) {
        layout_engine_->stop();
        layout_engine_.reset();
    }
    state_.store(SERVER_STATE_LOADING_MODEL);
    LOG_INFO("Document pipeline stopped for {}", model_key_);
}

int DocumentPipelineService::get_state() {
    return state_.load();
}

DocumentPipelineService::PreprocessContext DocumentPipelineService::preprocess_document(const server_http_req& req) const {
    const auto input = DocumentInputAdapter::extract_pdf(req);
    if (!input.has_pdf || input.data.empty()) {
        throw std::runtime_error("PDF input was detected but payload extraction failed");
    }

    PreprocessContext context;
    const auto& config = Singleton<HLIEConfigService>::instance();
    context.render_options.dpi = config->get_int("document", "pdf_dpi", context.render_options.dpi, true);
    context.render_options.max_long_side = config->get_int("document", "pdf_max_long_side", context.render_options.max_long_side, true);
    context.render_options.max_pages = config->get_int("document", "pdf_max_pages", context.render_options.max_pages, true);
    context.include_page_margins = request_boolish(req, "document_include_page_margins", false);
    context.result.source = input.source;
    context.result.filename = input.filename;
    context.result.content_type = input.content_type;

    for (auto& page_image : PdfiumPdfRenderer::render(input.data, context.render_options, &context.total_page_count)) {
        DocumentPage page;
        page.image = std::move(page_image);
        context.result.pages.push_back(std::move(page));
    }

    if (layout_engine_ && layout_engine_->is_ready()) {
        layout_engine_->detect_layout(context.result);
    }
    DocumentCore::apply_page_margin_policy(context.result, context.include_page_margins);
    context.crop_count = DocumentRegionCropper::crop_recognizable_regions(context.result);
    return context;
}

nlohmann::json DocumentPipelineService::build_document_response(const server_http_req& req, const PreprocessContext& context,
                                                                const std::string& object, const std::string& status) const {
    auto document_json = document_preprocess_result_to_json(context.result);
    const auto merged_text = DocumentCore::merge_document_text(context.result);
    document_json["text"] = merged_text;
    document_json["markdown"] = merged_text;
    json layout_module_json = json::object();
    if (layout_engine_) {
        layout_module_json = layout_engine_->module_info();
    }
    return {
        {"object", object},
        {"model", model_key_},
        {"mmproj", mmproj_path_},
        {"path", req.path},
        {"render_options", {
            {"dpi", context.render_options.dpi},
            {"max_long_side", context.render_options.max_long_side},
            {"max_pages", context.render_options.max_pages},
            {"total_pages", context.total_page_count},
            {"rendered_pages", context.result.pages.size()},
            {"truncated", context.total_page_count > static_cast<int>(context.result.pages.size())},
        }},
        {"document", document_json},
        {"layout", {
            {"status", layout_engine_ && layout_engine_->is_ready() ? "loaded" : "unavailable"},
            {"engine", "ppdoclayoutv3"},
            {"module", layout_module_json},
        }},
        {"document_pipeline", {
            {"region_crop_count", context.crop_count},
            {"region_crop_mode", "bbox"},
            {"include_page_margins", context.include_page_margins},
            {"layout_metrics", DocumentCore::build_layout_metrics(context.result, context.include_page_margins)},
            {"recognition", status == "recognized" ? "completed" : "not_requested"},
        }},
        {"status", status},
    };
}

server_http_res_ptr DocumentPipelineService::handle_document_request(const server_http_req& req) const {
    auto res = std::make_unique<server_http_res>();
    res->content_type = MIMETYPE_JSON;

    try {
        auto context = preprocess_document(req);
        res->status = 200;
        res->data = safe_json_to_str(build_document_response(req, context, "document.preprocess", "rendered"));
    } catch (const std::exception& ex) {
        LOG_ERROR("Document pipeline failed to preprocess PDF: {}", ex.what());
#ifdef HLIE_ENABLE_PDFIUM
        const auto error_type = ERROR_TYPE_SERVER;
        const int ret = 80007;
#else
        const auto error_type = ERROR_TYPE_NOT_SUPPORTED;
        const int ret = 80008;
#endif
        const auto error = format_error_response_ex(ex.what(), error_type, ret);
        res->status = error.value("code", 500);
        res->data = safe_json_to_str({{"error", error}});
    }

    return res;
}

server_http_res_ptr DocumentPipelineService::handle_document_ocr_request(const server_http_req& req,
                                                                         const LLModelService& llm_service) const {
    auto res = std::make_unique<server_http_res>();
    res->content_type = MIMETYPE_JSON;

    try {
        auto context = preprocess_document(req);

        const auto& config = Singleton<HLIEConfigService>::instance();
        const int max_regions = config->get_int("document", "ocr_max_regions", 0, true);
        int recognized_count = 0;
        int attempted_count = 0;
        int skipped_by_limit_count = 0;
        for (auto& page : context.result.pages) {
            for (auto& region : page.regions) {
                if (max_regions > 0 && attempted_count >= max_regions) {
                    if (!region.crop.pixels.empty()) {
                        ++skipped_by_limit_count;
                    }
                    continue;
                }
                if (region.crop.pixels.empty()) {
                    continue;
                }
                ++attempted_count;
                json body = build_region_chat_body(model_key_, region);
                server_http_req region_req{
                    req.params,
                    req.headers,
                    "/v1/chat/completions",
                    "",
                    body.dump(),
                    {},
                    req.should_stop,
                };
                const auto region_response = llm_service.handle_chat_completions(region_req);
                region.text = extract_chat_text(region_response);
                if (DocumentCore::is_low_confidence_fallback_region(region) &&
                    DocumentCore::looks_like_non_transcription_response(region.text)) {
                    LOG_INFO("Document OCR discarded non-transcription fallback output page={}, region={}, text_len={}",
                             page.image.page_index, region.id, region.text.size());
                    region.text.clear();
                }
                if (!region.text.empty()) {
                    ++recognized_count;
                }
                LOG_INFO("Document OCR recognized page={}, region={}, label={}, text_len={}",
                         page.image.page_index, region.id, region.label, region.text.size());
            }
        }

        auto response = build_document_response(req, context, "document.ocr", "recognized");
        const bool limited = skipped_by_limit_count > 0;
        response["document_pipeline"]["recognition"] = {
            {"status", limited ? "limited" : "completed"},
            {"attempted_regions", attempted_count},
            {"recognized_regions", recognized_count},
            {"skipped_by_limit_regions", skipped_by_limit_count},
            {"max_regions", max_regions},
        };
        res->status = 200;
        res->data = safe_json_to_str(response);
    } catch (const std::exception& ex) {
        LOG_ERROR("Document pipeline failed to OCR PDF: {}", ex.what());
#ifdef HLIE_ENABLE_PDFIUM
        const auto error_type = ERROR_TYPE_SERVER;
        const int ret = 80009;
#else
        const auto error_type = ERROR_TYPE_NOT_SUPPORTED;
        const int ret = 80008;
#endif
        const auto error = format_error_response_ex(ex.what(), error_type, ret);
        res->status = error.value("code", 500);
        res->data = safe_json_to_str({{"error", error}});
    }

    return res;
}

HLIE_NAMESPACE_END
