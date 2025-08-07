#ifndef HLIE_SINGLETON_H
#define HLIE_SINGLETON_H

#include <iostream>
#include <memory>
#include <mutex>

#include "hlie_base.h"

HLIE_NAMESPACE_BEGIN

template <typename T>
class HLIE_API Singleton {
public:
    template <typename... ArgTypes>
    static T* instance(ArgTypes&&... args) {
        std::call_once(_of, [&args...] { _instance.reset(new T(std::forward<ArgTypes>(args)...)); });
        return _instance.get();
    }

private:
    Singleton() = default;
    ~Singleton() = default;
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;

    inline static std::once_flag _of;
    inline static std::unique_ptr<T> _instance;
};

HLIE_NAMESPACE_END

#endif // HLIE_SINGLETON_H
