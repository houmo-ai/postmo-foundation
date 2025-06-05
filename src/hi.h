// Copyright (C) 2025, Houmo. All Rights Reserved.
// Confidential and Proprietary.

#ifndef HM_INFER_INTERFACE_H
#define HM_INFER_INTERFACE_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

typedef int32_t int32;
typedef float_t float32;
typedef double_t double64;
typedef void* HANDLE;

enum ERROR_CODE {
	E_OK,
	E_BUFFER_NOT_ENOUGH,
};

/////////////////////////////////////////////////////////////////////////////
/// Class api_interface  declaration.
/////////////////////////////////////////////////////////////////////////////
struct params
{
    int32 n_ctx = 2048;
    int32 n_predict = 2048;
    int32 top_k = 30;
    float32 top_p = 0.95;
    float32 temperature = 1.0f;
    float32 repeat_penalty = 1.1;
    int32 repeat_last_n = 64;
    int32 seed = 42;
    double64 rope_theta = 10000.0;
	int model_type; //1:llama3,2:qianwen2,3....
};

enum generater_status {
    completed       = 0,    // finished generate
    generating      = 1,    // Running generation
    stopped         = 2,    // generation stopped
    exceed          = 3,    // exceed context size
    failed          = -1    // generation failed
};

enum status
{
    init           = 0,    // Init parameters
    loaded         = 1,    // Model loaded or reset
    unloaded       = 2,    // Unload model -> Release model and tokenizer
    inference      = 3,    // Running generation
    error          = -1    // General error
};


#ifdef _MSC_VER
#define HMINFER_API __declspec(dllexport)
#else
#define HMINFER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// if fail, return nullptr
//HANDLE HMINFER_API create_api_instance(params& params);
HANDLE HMINFER_API create_infer_instance(void* params, size_t params_bytes);

void HMINFER_API destroy_infer_instance(HANDLE handle);

//load model with model name ,layer:-1 load all layer.
bool HMINFER_API infer_loadmodel_by_path(HANDLE handle, const char* model_path, const char** model_names, size_t model_name_count);

//load model with buffer stream,layer:-1 load all layer.
bool HMINFER_API infer_loadmodel_by_buffer(HANDLE handle, uint8_t** buffers, size_t* bufferSize, size_t buffers_count);

//load token model not must
bool HMINFER_API infer_loadtoken(HANDLE handle, const char* token_fullpath);

//unload model
bool HMINFER_API infer_unloadmodel(HANDLE handle);

// model status see the enum status
//return enum status
int32 HMINFER_API infer_status(HANDLE handle);

//Generate
//if result == nullptr or *result_size is not enough, return E_BUFFER_NOT_ENOUGH and result_size return the needed size.
int32 HMINFER_API infer_Generate(HANDLE handle, const char* prompt, const void* params, size_t params_bytes, char** result, size_t* result_size);

//get prompt token number
int32 HMINFER_API infer_token_num(HANDLE handle, const char* prompt);

//stream generate
bool HMINFER_API infer_Generate_stream(HANDLE handle, const char* prompt, const void* params, size_t params_size);

//the generate token ,use for stream mode
//if result == nullptr or *result_size is not enough, return E_BUFFER_NOT_ENOUGH and result_size return the needed size.
int32 HMINFER_API infer_Get_Generate_token(HANDLE handle, char** result, size_t* result_size);

//generate status see enum generater_status defined ahead
//return enum generater_status
int32 HMINFER_API infer_Generate_status(HANDLE handle);

//stop generate
bool HMINFER_API infer_stop(HANDLE handle);

//warmup not must
bool HMINFER_API infer_warmup(HANDLE handle);

//reset model not must
void HMINFER_API infer_Reset(HANDLE handle);

#ifdef __cplusplus
}
#endif

#endif // HM_INFER_INTERFACE_H
