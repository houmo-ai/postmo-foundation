// Copyright (C) 2024, Lenovo. All Rights Reserved.
// Confidential and Proprietary.

#ifndef HTTP_SERVER_C_INTERFACE_3_H
#define HTTP_SERVER_C_INTERFACE_3_H

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

enum hardware_type
{
	HW_GPU = 0,
	HW_IGPU = 1,
	HW_AGPU = 2,
	HW_ANPU = 3,
	HW_INPU = 4,
	HW_DNPU = 5
};

struct hardware_info
{
	int32 layer;
	int32 type;
};

#ifdef _MSC_VER
#define HTTP_SERVER_API __declspec(dllexport)
#else
#define HTTP_SERVER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// if fail, return nullptr
//HANDLE HTTP_SERVER_API create_api_instance(params& params);
HANDLE HTTP_SERVER_API create_api_instance(void* params, size_t params_bytes);

void HTTP_SERVER_API destroy_api_instance(HANDLE handle);

//load model with model name ,layer:-1 load all layer.
//bool HTTP_SERVER_API api_loadmodel_by_path(HANDLE handle, const std::string& model_path, std::vector<std::string>& model_name, hardware_info hwinfo);
bool HTTP_SERVER_API api_loadmodel_by_path(HANDLE handle, const char* model_path, const char** model_names, size_t model_name_count, hardware_info hwinfo);

//load model with buffer stream,layer:-1 load all layer.
//bool HTTP_SERVER_API api_loadmodel_by_buffer(HANDLE handle, std::vector<uint8_t*>& buffers, std::vector<size_t>& buffersSize, hardware_info hwinfo);
bool HTTP_SERVER_API api_loadmodel_by_buffer(HANDLE handle, uint8_t** buffers, size_t* bufferSize, size_t buffers_count, hardware_info hwinfo);

//load token model not must
//bool HTTP_SERVER_API api_loadtoken(HANDLE handle, const std::string& token_fullpath);
bool HTTP_SERVER_API api_loadtoken(HANDLE handle, const char* token_fullpath);

//unload model
bool HTTP_SERVER_API api_unloadmodel(HANDLE handle);

// model status see the enum status
//return enum status
int32 HTTP_SERVER_API api_status(HANDLE handle);

//Generate
//std::string HTTP_SERVER_API api_Generate(HANDLE handle, const std::string& prompt, const params& params);
//if result == nullptr or *result_size is not enough, return E_BUFFER_NOT_ENOUGH and result_size return the needed size.
int32 HTTP_SERVER_API api_Generate(HANDLE handle, const char* prompt, const void* params, size_t params_bytes, char** result, size_t* result_size);
/*
{
"content": "我是联想基于通义千问打造的天禧大模型。",
"id_slot": 0,
"stop": true,
"model": "model0_v2.0.2.0-confused.gguf",
"tokens_predicted": 15,
"tokens_evaluated": 23,
"generation_settings": {
	"n_ctx": 4096,
	"n_predict": 1024,
	"model": "model0_v2.0.2.0-confused.gguf",
	"seed": 4294967295,
	"temperature": 0.699999988079071,
	"dynatemp_range": 0,
	"dynatemp_exponent": 1,
	"top_k": 20,
	"top_p": 0.800000011920929,
	"min_p": 0.05000000074505806,
	"tfs_z": 1,
	"typical_p": 1,
	"repeat_last_n": 64,
	"repeat_penalty": 1.0499999523162842,
	"presence_penalty": 0,
	"frequency_penalty": 0,
	"penalty_prompt_tokens": [],
	"use_penalty_prompt_tokens": false,
	"mirostat": 0,
	"mirostat_tau": 5,
	"mirostat_eta": 0.10000000149011612,
	"penalize_nl": false,
	"stop": [],
	"n_keep": 0,
	"ignore_eos": false,
	"stream": false,
	"logit_bias": [],
	"n_probs": 0,
	"min_keep": 0,
	"grammar": "",
	"samplers": [
		"top_k",
		"tfs_z",
		"typical_p",
		"top_p",
		"min_p",
		"temperature"
	]
},
"prompt": "",
"truncated": false,
"stopped_eos": true,
"stopped_word": false,
"stopped_limit": false,
"stopping_word": "",
"tokens_cached": 37,
"timings": {
	"prompt_n": 23,
	"prompt_ms": 178.442,
	"prompt_per_token_ms": 7.758347826086957,
	"prompt_per_second": 128.89342195223097,
	"predicted_n": 15,
	"predicted_ms": 1012.613,
	"predicted_per_token_ms": 67.50753333333334,
	"predicted_per_second": 14.813161592829639
}
}
*/

//get prompt token number
//int32 HTTP_SERVER_API api_token_num(HANDLE handle, const std::string& prompt);
int32 HTTP_SERVER_API api_token_num(HANDLE handle, const char* prompt);

//stream generate
//bool HTTP_SERVER_API api_Generate_stream(HANDLE handle, const std::string& prompt, const params& params);
bool HTTP_SERVER_API api_Generate_stream(HANDLE handle, const char* prompt, const void* params, size_t params_size);

//the generate token ,use for stream mode
//if result == nullptr or *result_size is not enough, return E_BUFFER_NOT_ENOUGH and result_size return the needed size.
int32 HTTP_SERVER_API api_Get_Generate_token(HANDLE handle, char** result, size_t* result_size);
//{"content":" ","stop":false,"id_slot":0,"multimodal":false}
//{"content":"","id_slot":0,"stop":true,"model":"model0_v2.0.2.0-confused.gguf","tokens_predicted":79,"tokens_evaluated":23,"generation_settings":{"n_ctx":4096,"n_predict":1024,"model":"model0_v2.0.2.0-confused.gguf","seed":4294967295,"temperature":0.699999988079071,"dynatemp_range":0.0,"dynatemp_exponent":1.0,"top_k":20,"top_p":0.800000011920929,"min_p":0.05000000074505806,"tfs_z":1.0,"typical_p":1.0,"repeat_last_n":64,"repeat_penalty":1.0499999523162842,"presence_penalty":0.0,"frequency_penalty":0.0,"penalty_prompt_tokens":[],"use_penalty_prompt_tokens":false,"mirostat":0,"mirostat_tau":5.0,"mirostat_eta":0.10000000149011612,"penalize_nl":false,"stop":[],"n_keep":0,"ignore_eos":false,"stream":true,"logit_bias":[],"n_probs":0,"min_keep":0,"grammar":"","samplers":["top_k","tfs_z","typical_p","top_p","min_p","temperature"]},"prompt":"","truncated":false,"stopped_eos":true,"stopped_word":false,"stopped_limit":false,"stopping_word":"","tokens_cached":101,"timings":{"prompt_n":23,"prompt_ms":183.544,"prompt_per_token_ms":7.980173913043479,"prompt_per_second":125.31055223815542,"predicted_n":79,"predicted_ms":5507.942,"predicted_per_token_ms":69.72078481012659,"predicted_per_second":14.342925179676909}}

//generate status see generater_status
//return enum generater_status
int32 HTTP_SERVER_API api_Generate_status(HANDLE handle);

//stop generate
bool HTTP_SERVER_API api_stop(HANDLE handle);

//warmup not must
bool HTTP_SERVER_API api_warmup(HANDLE handle);

//reset model not must
void HTTP_SERVER_API api_Reset(HANDLE handle);

#ifdef __cplusplus
}
#endif

#endif // HTTP_SERVER_C_INTERFACE_3_H
