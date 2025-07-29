#ifndef HLIE_MODEL_SERVICE_H
#define HLIE_MODEL_SERVICE_H

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

class IHLIEModelService {
public:
    IHLIEModelService() = default;
    virtual ~IHLIEModelService() = default;

    virtual void init() = 0;
    virtual void run() = 0;
};

class SDModelService final : public IHLIEModelService {
public:
    SDModelService() = default;
    ~SDModelService() override = default;

    /**
     * 初始化并传递参数给SDCtx
     */
    void init() override;

    /**
     * 执行模型推理进行生图
     */
    void run() override;
};

class LLModelService final : public IHLIEModelService {
public:
    LLModelService() = default;
    ~LLModelService() override = default;

    void init() override;
    void run() override;
};

HLIE_NAMESPACE_END


#endif //HLIE_MODEL_SERVICE_H
