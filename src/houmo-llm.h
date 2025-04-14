

#ifndef HOUMO_LLM_H
#define HOUMO_LLM_H

#include <stddef.h>
#include <string>

#ifdef _MSC_VER
#define HTTP_SERVER_API __declspec(dllexport)
#else
#define HTTP_SERVER_API
#endif

typedef void* HANDLE;
HANDLE HTTP_SERVER_API create_instance(std::string model_path);
int HTTP_SERVER_API destroy_instance(HANDLE h);
int HTTP_SERVER_API generate(HANDLE h);
int HTTP_SERVER_API generate_stream(HANDLE h);
int HTTP_SERVER_API generate_stream_next_token_all(HANDLE h);
int HTTP_SERVER_API generate_stream_next_token(HANDLE h, char ** result, size_t* result_size);
void HTTP_SERVER_API api_get_info(HANDLE handle);

#endif // HOUMO_LLM_H