#include "hlie_app/models/hlie_model_base.h"

#include <regex>

#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_tcim_util.h"

HLIE_NAMESPACE_BEGIN


void HLIEModelBase::set_devices(const std::vector<int>& devices) {
    device_ids_.clear();
    device_ids_ = devices;
}

std::shared_ptr<tcim::Module> HLIEModelBase::create_infer_engine(const std::string& file_name, const size_t offset, const size_t size,
                                                                 std::vector<std::string> dummy_tensor_names) {
    if (wm_.GetInitStatus() != tcim::Status::OK) {
        LOG_INFO("Start to set  wm_ with dev {} of all {} ", device_ids_[0], device_ids_.size());
        const tcim::DevManager dev_manager = tcim::DevManager::Create(device_ids_, "Xh2HalBackend");
        wm_ = tcim::Module::WeightManager::CreateWeightManager(dev_manager);
    }

    auto infer_model = std::make_shared<tcim::Module>();
    auto option = tcim::Module::Option(wm_);
    option.SetDummyTensors(dummy_tensor_names);
    option.EnableHostLazyLoading(false);
    option.SetModelOffset(offset, size);
    LOG_INFO("Start to load LoadModel offset: {}, size: {}", offset, size);
    if (const tcim::Status status = infer_model->LoadModel(file_name, option); status != tcim::Status::OK) {
        LOG_ERROR("LoadModel failed: status == {}", static_cast<int>(status));
        return nullptr;
    }

    return infer_model;
}


HLIE_NAMESPACE_END
