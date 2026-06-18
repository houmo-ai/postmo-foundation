#ifndef HLIE_RK_PERFORMANCE_H
#define HLIE_RK_PERFORMANCE_H

#include "hlie_base/hlie_base.h"

HLIE_NAMESPACE_BEGIN

class HLIE_API HLIERkPerformance {
public:
    static bool is_enabled();
    static void apply_for_process();
    static void apply_for_current_thread();
};

HLIE_NAMESPACE_END

#endif // HLIE_RK_PERFORMANCE_H
