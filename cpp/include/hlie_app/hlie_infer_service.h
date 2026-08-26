//
// Created by tuo.wang on 2025/7/28.
//

#ifndef HLIE_INFER_SERVICE_H
#define HLIE_INFER_SERVICE_H

#include "hlie_base/hlie_base.h"
#include "hlie_base/hlie_blocking_queue.h"

HLIE_NAMESPACE_BEGIN

class HLIEModelInfo;

class HLIE_API HLIEInferService {
public:
    HLIEInferService() = default;
    ~HLIEInferService() = default;


    void generate_images();
};

HLIE_NAMESPACE_END

#endif //HLIE_INFER_SERVICE_H
