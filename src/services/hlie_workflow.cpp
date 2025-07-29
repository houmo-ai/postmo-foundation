#include "hlie_workflow.h"
#include "hlie_model_service.h"
#include "hlie_infer_service.h"

HLIE_NAMESPACE_BEGIN

void HLIEWorkflow::run() {

}

std::shared_ptr<HLIEInferService> HLIEWorkflow::get_infer_service() const {
    return infer_service_;
}

std::shared_ptr<HLIEModelService> HLIEWorkflow::get_model_service() const {
    return model_service_;
}

HLIE_NAMESPACE_END
