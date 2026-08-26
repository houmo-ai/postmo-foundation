#ifndef HLIE_BASE_H
#define HLIE_BASE_H

#ifndef HLIE_API
#if defined HLIE_STATIC
#   define HLIE_API
#elif defined _WIN32 || defined __CYGWIN__
#   ifdef HLIE_SHARED_LIB_BUILDING
#		define HLIE_API __declspec(dllexport)
#	else
#		define HLIE_API __declspec(dllimport)
#   endif
#elif defined(__GNUC__) && __GNUC__ >= 4
    #define HLIE_API __attribute__((visibility("default")))
#else
    #define HLIE_API
#endif
#endif

#define HLIE_NAMESPACE_BEGIN namespace hm_HLIE {
#define HLIE_NAMESPACE_END }

#endif
