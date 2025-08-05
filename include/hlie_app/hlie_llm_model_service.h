#ifndef HLIE_LLM_MODEL_SERVICE_H
#define HLIE_LLM_MODEL_SERVICE_H


#include <thread>


#include "hlie_base/hlie_common_define.h"
#include "hlie_model_service.h"
#include "json.hpp"
#include "utils.hpp"

HLIE_NAMESPACE_BEGIN

struct LlamaServer;

class LLModelService final : public IHLIEModelService {
public:
    LLModelService() = default;
    ~LLModelService() override = default;

    int init(const std::string& model_path, const HLIEModelInfo& model_info) override;

    void run() override;

    void process() override;

    void stop() override;

    int get_state() override;

    int get_batch_num() const;

    int handle_health(json& resp) const;

    void handle_models(json& resp) const;

    void handle_chat_completions(const httplib::Request& req, httplib::Response& res) const;

    void handle_completions(const httplib::Request& req, httplib::Response& res) const;

    void handle_completions_oai(const httplib::Request & req, httplib::Response & res) const;

private:
    /**
     * handle completion-like requests (completion, chat, infill)
     * we can optionally provide a custom format for partial results and final results
     *
     * @param type
     * @param data
     * @param files
     * @param is_connection_closed
     * @param res
     * @param oaicompat
     */
    void handle_completions_impl(server_task_type type, json& data, const std::vector<raw_buffer>& files,
                                 const std::function<bool()>& is_connection_closed, httplib::Response& res, oaicompat_type oaicompat) const;


    std::shared_ptr<LlamaServer> llama_server_;
    std::unique_ptr<std::thread> worker_thread_;  // 工作线程
    std::atomic<bool> should_stop_{false};  // 线程停止标志
    int batch_num_{1};
};

HLIE_NAMESPACE_END


#endif // HLIE_LLM_MODEL_SERVICE_H
