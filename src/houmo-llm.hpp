

#ifndef HOUMO_LLM_HPP
#define HOUMO_LLM_HPP
#include <string>

#ifdef _MSC_VER
#define HTTP_SERVER_API __declspec(dllexport)
#else
#define HTTP_SERVER_API
#endif

typedef void* HANDLE;
int HTTP_SERVER_API generate_stream_next_token_string(HANDLE h, std::string & result_json_str);

#endif // HOUMO_LLM_HPP