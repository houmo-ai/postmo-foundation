#include "hlie_app/hlie_app_controller.h"
#include "hlie_base/hlie_config_service.h"

#include <iostream>

int main(const int argc, char** argv) {
    std::setlocale(LC_NUMERIC, "C");

    const auto& config_service = hm_HLIE::Singleton<hm_HLIE::HLIEConfigService>::instance();
    config_service->set_args(argc, argv);

    hm_HLIE::HLIELinuxAppController appController;
    int ret = appController.initialize();
    if (ret < 0) {
        std::cout << "HLIECpp appController failed to initialize:" << ret << std::endl;
        return ret;
    }

    ret = appController.run();
    std::cout << "HLIECpp appController run, exit code:" << ret << std::endl;
    return ret;
}
