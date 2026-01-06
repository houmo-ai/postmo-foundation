#ifndef HLIE_LLM_MODEL_SERVICE_H
#define HLIE_LLM_MODEL_SERVICE_H


#include <thread>


#include "hlie_base/hlie_common_define.h"
#include "hlie_model_service.h"
#include <httplib.h>
#include <server/server-common.h>
#include <server/server-task.h>

class InfoPrinter;
struct server_context;
struct server_routes;
struct server_http_res;
struct server_http_req;
using server_http_res_ptr = std::unique_ptr<server_http_res>;

HLIE_NAMESPACE_BEGIN

class LLModelService final : public IHLIEModelService {
public:
    LLModelService() = default;
    ~LLModelService() override = default;

    int init(const std::string& model_dir, const HLIEModelInfo& model_info) override;

    void run() override;

    void process() override;

    void stop() override;

    int get_state() override;

    int get_batch_num() const;

    server_http_res_ptr handle_health(const server_http_req& req) const;

    server_http_res_ptr handle_models(const server_http_req& req) const;

    server_http_res_ptr handle_chat_completions(const server_http_req& req) const;

    server_http_res_ptr handle_completions(const server_http_req& req) const;

    server_http_res_ptr handle_completions_oai(const server_http_req& req) const;

    server_http_res_ptr handle_embeddings(const server_http_req& req, task_response_type oaicompat) const;

    server_http_res_ptr handle_props(const server_http_req& req) const;

    [[nodiscard]] std::string get_model_key() const {
        return model_key_;
    }

private:
    std::shared_ptr<InfoPrinter> info_printer_;
    std::shared_ptr<server_context> llama_server_;
    std::shared_ptr<server_routes> llama_server_routes_;
    std::unique_ptr<std::thread> worker_thread_;  // 工作线程
    std::atomic<bool> should_stop_{false};  // 线程停止标志
    std::atomic<bool> is_ready = false;
    std::string model_key_;
    int batch_num_{1};
};

HLIE_NAMESPACE_END


#endif // HLIE_LLM_MODEL_SERVICE_H
