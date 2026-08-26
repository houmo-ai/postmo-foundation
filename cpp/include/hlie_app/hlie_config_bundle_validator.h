#ifndef HLIE_CONFIG_BUNDLE_VALIDATOR_H
#define HLIE_CONFIG_BUNDLE_VALIDATOR_H

#include <string>
#include <vector>

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

class HLIEConfigService;

struct HLIE_API HLIEConfigBundleValidationResult {
    bool valid = true;
    std::string error;
    std::vector<std::string> warnings;
    std::vector<std::string> notes;
};

class HLIE_API HLIEConfigBundleValidator {
public:
    /** Validate only explicit --config bundles; platform-default deployments retain legacy behavior. */
    [[nodiscard]] static HLIEConfigBundleValidationResult validate(const HLIEConfigService& config_service);
};

HLIE_NAMESPACE_END

#endif // HLIE_CONFIG_BUNDLE_VALIDATOR_H
