#include "gguf.h"
#include "tcim/tcim_dev_ctrl.h"
#include "tcim/tcim_runtime.h"

#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct HmmSection {
    std::string name;
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
};

struct ModuleDeleter {
    void operator()(tcim::Module * module) const { delete module; }
};

using ModulePtr = std::unique_ptr<tcim::Module, ModuleDeleter>;

std::uint32_t used_mb(const std::shared_ptr<tcim::dev_ctrl::HalDevice> & device, int device_id) {
    tcim::dev_ctrl::MemInfo info{};
    const auto status = device->GetMemInfo(device_id, &info);
    if (status != tcim::Status::OK) {
        std::cerr << "GetMemInfo failed, status=" << static_cast<int>(status) << '\n';
        return UINT32_MAX;
    }
    return info.mem_used;
}

std::vector<HmmSection> find_hmm_sections(const std::string & path) {
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(
        gguf_init_from_file(path.c_str(), params), gguf_free);
    if (!ctx) {
        throw std::runtime_error("gguf_init_from_file failed: " + path);
    }

    const std::regex hmm_pattern("^.+\\.hmm(s)?$");
    const std::uint64_t meta_size = static_cast<std::uint64_t>(gguf_get_meta_size(ctx.get()));
    std::vector<HmmSection> result;
    const auto count = gguf_get_n_tensors(ctx.get());
    for (std::int64_t i = 0; i < count; ++i) {
        const char * raw_name = gguf_get_tensor_name(ctx.get(), i);
        if (raw_name == nullptr || !std::regex_match(raw_name, hmm_pattern)) {
            continue;
        }
        result.push_back({
            raw_name,
            meta_size + static_cast<std::uint64_t>(gguf_get_tensor_offset(ctx.get(), i)),
            static_cast<std::uint64_t>(gguf_get_tensor_size(ctx.get(), i)),
        });
    }
    return result;
}

void print_sample(const std::string & phase, int iteration, std::uint32_t memory_mb) {
    std::cout << "SAMPLE phase=" << phase
              << " iteration=" << iteration
              << " mem_used_mb=" << memory_mb << '\n';
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "usage: tcim-module-lifecycle-test <model.gguf> [iterations] [device_id] [module-only|touch-cache-inputs|run-default|llama-asr-init]\n";
        return 2;
    }

    const std::string model_path = argv[1];
    const int iterations = argc >= 3 ? std::max(1, std::atoi(argv[2])) : 2;
    const int device_id = argc >= 4 ? std::max(0, std::atoi(argv[3])) : 0;
    const std::string mode = argc >= 5 ? argv[4] : "module-only";
    const bool touch_cache_inputs = mode == "touch-cache-inputs";
    const bool run_default = mode == "run-default";
    const bool llama_asr_init = mode == "llama-asr-init";

    auto hal = tcim::dev_ctrl::HalDeviceFactory::Create("Xh2HalBackend");
    if (!hal) {
        std::cerr << "HalDeviceFactory::Create failed\n";
        return 3;
    }

    const auto sections = find_hmm_sections(model_path);
    if (sections.empty()) {
        std::cerr << "No prefill/decode HMM tensors found in " << model_path << '\n';
        return 4;
    }

    std::cout << "DIRECT_TCIM_TEST model=" << model_path
              << " sections=" << sections.size()
              << " iterations=" << iterations
              << " device=" << device_id
              << " mode=" << mode << '\n';
    for (const auto & section : sections) {
        std::cout << "SECTION name=" << section.name
                  << " offset=" << section.offset
                  << " size=" << section.size << '\n';
    }

    print_sample("baseline", 0, used_mb(hal, device_id));

    for (int iteration = 1; iteration <= iterations; ++iteration) {
        tcim::DevManager dev_manager =
            tcim::DevManager::Create(std::vector<int>{device_id}, "Xh2HalBackend");
        if (dev_manager.GetInitStatus() != tcim::Status::OK) {
            std::cerr << "DevManager creation failed\n";
            return 5;
        }

        tcim::Module::WeightManager weight_manager =
            tcim::Module::WeightManager::CreateWeightManager(dev_manager);
        std::vector<ModulePtr> modules;
        std::vector<tcim::Tensor> cache_tensors;
        std::vector<std::string> asr_cache_names;
        if (llama_asr_init) {
            for (int layer = 0; layer < 28; ++layer) {
                asr_cache_names.push_back("model_layers_" + std::to_string(layer) + "_self_attn_kcache_input");
                asr_cache_names.push_back("model_layers_" + std::to_string(layer) + "_self_attn_vcache_input");
            }
        }
        for (const auto & section : sections) {
            auto module = ModulePtr(new tcim::Module());
            tcim::Module::Option option(weight_manager);
            if (llama_asr_init && section.name.rfind("prefill", 0) == 0) {
                option.SetDummyTensors(asr_cache_names);
            }
            option.SetModelOffset(section.offset, section.size);
            const auto status = module->LoadModel(model_path, option);
            if (status != tcim::Status::OK) {
                std::cerr << "LoadModel failed section=" << section.name
                          << " status=" << static_cast<int>(status) << '\n';
                return 6;
            }
            if (touch_cache_inputs || (llama_asr_init && section.name.rfind("decoder", 0) == 0)) {
                for (std::size_t input_id = 0; input_id < module->GetInputNum(); ++input_id) {
                    const std::string input_name = module->GetInputName(input_id);
                    if (input_name.find("cache") == std::string::npos &&
                        input_name.find("past_") == std::string::npos &&
                        input_name.find("recurrent_state") == std::string::npos) {
                        continue;
                    }
                    auto tensor = module->GetDevInput(input_name);
                    std::cout << "CACHE_INPUT section=" << section.name
                              << " name=" << input_name
                              << " size=" << tensor.MemSize() << '\n';
                    cache_tensors.push_back(std::move(tensor));
                }
            }
            modules.push_back(std::move(module));
        }

        if (run_default) {
            for (std::size_t module_id = 0; module_id < modules.size(); ++module_id) {
                const auto run_status = modules[module_id]->Run();
                if (run_status != tcim::Status::OK) {
                    std::cerr << "Run failed module=" << module_id
                              << " status=" << static_cast<int>(run_status) << '\n';
                    return 7;
                }
                const auto sync_status = modules[module_id]->Sync();
                if (sync_status != tcim::Status::OK) {
                    std::cerr << "Sync failed module=" << module_id
                              << " status=" << static_cast<int>(sync_status) << '\n';
                    return 8;
                }
            }
        }

        print_sample("loaded", iteration, used_mb(hal, device_id));

        cache_tensors.clear();
        modules.clear();
        weight_manager = tcim::Module::WeightManager();
        std::this_thread::sleep_for(std::chrono::seconds(3));
        print_sample("destroyed", iteration, used_mb(hal, device_id));
    }

    return 0;
}
