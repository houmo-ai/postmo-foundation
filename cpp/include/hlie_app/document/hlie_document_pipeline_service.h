#ifndef HLIE_DOCUMENT_PIPELINE_SERVICE_H
#define HLIE_DOCUMENT_PIPELINE_SERVICE_H

#include <atomic>
#include <memory>
#include <string>

#include <server/server-http.h>

#include "hlie_app/document/hlie_document_layout_engine.h"
#include "hlie_app/hlie_model_service.h"

HLIE_NAMESPACE_BEGIN

class LLModelService;

class DocumentPipelineService final : public IHLIEModelService {
public:
    DocumentPipelineService() = default;
    ~DocumentPipelineService() override = default;

    static bool supports_document_layout(const std::string& model_dir, const HLIEModelInfo& model_info);

    int init(const std::string& model_dir, const HLIEModelInfo& model_info,
             const std::shared_ptr<HLIEDeviceService>& device_service) override;
    void run() override;
    void process() override;
    void stop() override;
    int get_state() override;

    [[nodiscard]] const std::string& get_model_key() const { return model_key_; }
    [[nodiscard]] const std::string& get_mmproj_path() const { return mmproj_path_; }

    server_http_res_ptr handle_document_request(const server_http_req& req) const;
    server_http_res_ptr handle_document_ocr_request(const server_http_req& req, const LLModelService& llm_service) const;

private:
    struct PreprocessContext;

    static bool gguf_has_tensor(const std::string& path, const std::string& tensor_name);
    static std::string resolve_mmproj_path(const std::string& model_dir, const HLIEModelInfo& model_info);
    PreprocessContext preprocess_document(const server_http_req& req) const;
    nlohmann::json build_document_response(const server_http_req& req, const PreprocessContext& context,
                                           const std::string& object, const std::string& status) const;

    std::atomic<bool> is_ready_{false};
    std::string model_key_;
    std::string mmproj_path_;
    std::unique_ptr<DocumentLayoutEngine> layout_engine_;
};

HLIE_NAMESPACE_END

#endif // HLIE_DOCUMENT_PIPELINE_SERVICE_H
