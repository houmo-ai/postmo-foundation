//
// Created by tuo.wang on 2025/7/28.
//

#ifndef HLIE_WORKFLOW_H
#define HLIE_WORKFLOW_H

#include <memory>


#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

class HLIEModelService;
class HLIEInferService;
class HLIEWorkflow {
public:
    HLIEWorkflow() = default;
    ~HLIEWorkflow() = default;

    void run();

    std::shared_ptr<HLIEInferService> get_infer_service() const;
    std::shared_ptr<HLIEModelService> get_model_service() const;

private:
    std::shared_ptr<HLIEInferService> infer_service_;
    std::shared_ptr<HLIEModelService> model_service_;
};

HLIE_NAMESPACE_END

#endif //HLIE_WORKFLOW_H
