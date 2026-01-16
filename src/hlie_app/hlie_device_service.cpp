#include "hlie_app/hlie_device_service.h"


#include "hlie_app/hlie_model_info.h"
#include "hlie_base/hlie_common_define.h"
#include "hlie_base/hlie_config_service.h"
#include "hlie_base/hlie_device_helper.h"
#include "hlie_base/hlie_exception.h"
#include "hlie_base/hlie_logger.h"
#include "hlie_base/hlie_util.h"

HLIE_NAMESPACE_BEGIN


int HLIEDeviceService::acquire_mem(int device, int size) {
    hm_device_info info = {};
    const uint32_t ret = hm_sys_get_device_info(&info);
    /* prefer info.num_devices if filled, otherwise use return value */
    if (const uint32_t dev_count = info.num_devices ? info.num_devices : ret; device >= dev_count) {
        LOG_ERROR("device {} not found", device);
        return -1;
    }

    hm_mem_info mem_info{};
    if (hm_sys_get_mem_info(device, &mem_info) < 0) {
        LOG_ERROR("failed to get mem info of device {}", device);
        return -1;
    }

    if (auto safe_available_mem = mem_info.mem_avail - 500; safe_available_mem < size) {
        LOG_ERROR("device {} memory is not enough, available {}MB, required {}MB", device, safe_available_mem, size);
        return -1;
    }

    return 0;
}

int HLIEDeviceService::get_device_infos(std::vector<HLIEDeviceInfo>& device_infos) {
    hm_device_info info = {};
    const uint32_t ret_code = hm_sys_get_device_info(&info);
    /* prefer info.num_devices if filled, otherwise use return value */
    const uint32_t dev_count = info.num_devices ? info.num_devices : ret_code;
    if (dev_count <= 0) {
        LOG_ERROR("no device found!");
        return -1;
    }

    LOG_INFO("found {} devices", dev_count);

    float temp = 0.0f;

    for (int i = 0; i < dev_count; i++) {
        int dev = static_cast<int>(info.device_ids[i]);
        LOG_INFO("try to get mem info of device {}", dev);
        hm_mem_info mem_info{};
        auto ret = hm_sys_get_mem_info(dev, &mem_info);
        if (ret) {
            LOG_ERROR("failed to get mem info of device {}", dev);
            //return -1;
        }

        LOG_INFO("get mem info of device {}: total {}MB, used {}MB, avail {}MB", dev, mem_info.mem_total, mem_info.mem_used,
                 mem_info.mem_avail);

        LOG_INFO("try to get temperature of device {}", dev);

        ret = hm_sys_get_temperature(dev, &temp);
        if (ret < 0) {
            LOG_ERROR("failed to get temperature of device{}", dev);
           // return -1;
        }

        LOG_INFO("get temperature of device {}: {}", dev, temp);

        HLIEDeviceInfo device_info{device_info.id = dev, device_info.mem_total = mem_info.mem_total, device_info.mem_used = mem_info.mem_used,
                                   device_info.mem_avail = mem_info.mem_avail, device_info.temperature = temp};
        device_infos.emplace_back(device_info);
    }

    return static_cast<int>(dev_count);
}

server_http_res_ptr HLIEDeviceService::handle_devices(const server_http_req& req) {
    std::vector<HLIEDeviceInfo> device_infos;
    get_device_infos(device_infos);

    json devices = json::array();
    for (size_t i = 0; i < device_infos.size(); ++i) {
        const auto& [id, mem_total, mem_used, mem_avail, temperature] = device_infos[i];
        devices.push_back(
            {{"id", i}, {"mem_total", mem_total}, {"mem_used", mem_used}, {"mem_avail", mem_avail}, {"temperature", temperature}});
    }

    const json resp = {{"code", 0}, {"devices", devices}};
    auto res = std::make_unique<server_res_generator_light>();
    res->ok(resp);
    return res;
}


HLIE_NAMESPACE_END
