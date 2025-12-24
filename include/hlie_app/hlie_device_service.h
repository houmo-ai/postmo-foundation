#ifndef HLIE_DEVICE_SERVICE_H
#define HLIE_DEVICE_SERVICE_H

#include <future>
#include <vector>
#include <server/server-http.h>

#include "hlie_base/hlie_base.h"
#include "hm_sys.h"

HLIE_NAMESPACE_BEGIN

struct HLIEDeviceInfo {
    int id;
    uint32_t mem_total;
    uint32_t mem_used;
    uint32_t mem_avail;
    float temperature;
};

class HLIEDeviceService {
public:
    HLIEDeviceService() = default;
    ~HLIEDeviceService() = default;

    /**
     * acquire hm ai device memory
     *
     * @param device device index
     * @param size the acquiring memory size * MB
     * @return
     */
    int acquire_mem(int device, int size);

    int get_device_infos(std::vector<HLIEDeviceInfo>& device_infos);

    server_http_res_ptr handle_devices(const server_http_req& req);

private:
    int total_mem_;
    std::mutex mutex_;
};

HLIE_NAMESPACE_END

#endif
