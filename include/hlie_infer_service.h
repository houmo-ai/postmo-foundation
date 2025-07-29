//
// Created by tuo.wang on 2025/7/28.
//

#ifndef HLIE_INFER_SERVICE_H
#define HLIE_INFER_SERVICE_H

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

class HLIEModelInfo;

class HLIEInferService {
public:
    HLIEInferService() = default;
    ~HLIEInferService() = default;
    int init(const HLIEModelInfo & model_desc);


};

HLIE_NAMESPACE_END

#endif //HLIE_INFER_SERVICE_H
