#ifndef HLIE_DOCUMENT_LAYOUT_ENGINE_H
#define HLIE_DOCUMENT_LAYOUT_ENGINE_H

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "hlie_app/hlie_document_types.h"
#include "hlie_app/models/hlie_model_base.h"
#include "json.hpp"

HLIE_NAMESPACE_BEGIN

class DocumentLayoutEngine final : public HLIEModelBase {
public:
    DocumentLayoutEngine() = default;
    ~DocumentLayoutEngine() = default;

    bool init(const std::string& mmproj_path, const std::vector<int>& devices = {0});
    void stop();

    [[nodiscard]] bool is_ready() const { return layout_module_ != nullptr; }
    [[nodiscard]] const nlohmann::json& module_info() const { return module_info_; }

    void detect_layout(DocumentPreprocessResult& document) const;

private:
    static constexpr const char* layout_tensor_name = "ppdoclayoutv3.hmm";

    bool load_layout_module(const std::string& mmproj_path);
    void capture_module_info();
    std::vector<float> build_pixel_values(const PdfPageImage& image) const;
    std::vector<DocumentLayoutRegion> decode_layout_outputs(const std::vector<float>& logits, const std::vector<float>& boxes,
                                                            const std::vector<float>& order_logits,
                                                            const std::vector<float>& masks,
                                                            int query_count, int class_count,
                                                            int mask_height, int mask_width,
                                                            int image_width, int image_height) const;

    std::shared_ptr<tcim::Module> layout_module_;
    nlohmann::json module_info_ = nlohmann::json::object();
    mutable std::mutex infer_mutex_;
};

HLIE_NAMESPACE_END

#endif // HLIE_DOCUMENT_LAYOUT_ENGINE_H
