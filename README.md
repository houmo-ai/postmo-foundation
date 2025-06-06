# Houmo HI.CPP
Houmo HI.CPP是Houmo软件平台为Windows/Linux下打造的使用C++开发的推理服务

![M30](https://houmo.cn/UserFiles/upload/image/20240705/20240705122738020_t.jpg)
## 历史版本

| 版本    | 日期          | 描述                                                 |
|-------|-------------|----------------------------------------------------|
| 1.3.0-daily-20250606 | 2025年6月6日   | 支持以linux后台service服务形式跑hi.cpp推理服务，能以openai api对外提供能力，同时提供麒麟OS对应版本的SDk，操作系统支持麒麟OS v10，模型包括qwen3 8b和deepseek 7b |
| 1.2.0 | 2025年5月30日  | 实现Windows服务化安装，兼容更多OpenAI 接口参数，增加端口配置和日志Level与路径配置 |
| 1.1.0 | 2024年04月11日 | 正式版本                                               |

更多资讯欢迎关注

![HOUMO](https://houmo.cn/assets/images/code-wechat.jpg?v=592573c)


## 部署要求

1. 硬件要求（最低）

- CPU结构：x86_64
- 硬盘：200G
- AI硬件：力谋LM30智能加速卡

2. 软件要求

- 操作系统：Windows 11/10
- 力谋驱动版本：v2.3.0
- 力谋固件版本：v2.3.0（boot.img & loader.img）

## Windows服务部署

### 模型配置
目前应用软件将 DeepSeek 模型作为推理服务的默认模型配置，若用户需要修改使用模型，如：QWen2.5 或 DeepSeek 等其他模型时，
请联系后摩智能人员获取指定模型文件

### 支持的模型部署包
用户可在后摩智能指定的 JFrog Artifactory 网站上根据所需下载相关模型，下载地址：http://10.10.1.53:8082/ui/native/application_software/models

### 模型部署包命名说明
完整部署包名称为：houmo-application-software-<product>-<model_type>-<release>.tar.gz
- <product> 表示产品类型
  - chat：表示为应用部署包
  - model：表示为模型部署包
- <model_type> 表示模型类型
    - deepseek：表示该模型部署包为DeepSeek-r1 模型
    - qwen25：表示该模型部署包为QWen2.5 模型
    - qwen2：表示该模型部署包为QWen2 模型
- <release>表示应用软件版本号
  如 houmo-application-software-model-deepseek-14b-4core-8k-1batch-v2.3.0.tar.gz


### 模型包适配

hi.cpp 模型包目前基于公共模型包进行适配修改
我们需要下载两个模型包，一个是主包（含prefill和decode）, 另一个附包（含embedding和tokenizer模型）

例如下载完成如下：

主包：houmo-application-software-model-deepseek-14b-4core-8k-1batch-v2.3.0.tar.gz

附包：houmo-application-software-model-deepseek-14b-embedding-cpp-v2.3.0.tar.gz

我们创建自己model文件夹，如 deepseek_14b_256_8k_4cores ， 然后
将主包中4core_8k目录下的hmm格式的模型文件全部复制到 deepseek_14b_256_8k_4cores 根目录， 然后将附包中内容也全部复制到 deepseek_14b_256_8k_4cores 根目录。
适配完成之后的目录树如下：

```
models/hmm_deepseek_256_4096_4cores_20250317
$ tree
.
|-- DeepSeek-R1-Distill-Qwen-1.5B-IQ2_M.gguf
|-- deepseek_decode_part1.hmm
|-- deepseek_decode_part2.hmm
|-- deepseek_prefill_part1.hmm
|-- deepseek_prefill_part2.hmm
`-- hmquant
    |-- quant_embedding_traced.pt

```

### 安装过程
【注意】
- 安装本系统前请确保已安装好后摩推理加速卡、并且升级驱动和固件为指定版本
- 若有启动同类型的大模型推理系统，请先下线对应系统，以保证该服务的正常运行
- 请保证已获取到了完整的安装部署包（如应用或服务部署包、模型部署包等）
- 请确保系统已经安装有Visual C++ Redistributable for Visual Studio 2015-2022 , 如未安装请下载并安装，微软[官方链接](https://learn.microsoft.com/zh-cn/cpp/windows/latest-supported-vc-redist?view=msvc-170)

1. 双击hi_cpp.msi开始执行安装
2. 同意安装协议，然后点击下一步
3. 开始选择安装服务的路径，可以修改路径， 如果选择C盘，需要确保当前用户在C盘有默认的管理员写入权限（ 可尝试在C盘某一个目录下面右键创建文件，查看是否被限制为单一需要管理员权限的文件夹选项），选择完成后点击下一步
4. 需要指定模型路径，请选择你放置对应模型位置，选到选定模型的跟路径，然后下一步
5. 直接点击下一步进行安装，等待安装完成
6. 至此，没有任何问题的话，服务会自动以Windows服务形式启动运行

## 服务配置
### 推理服务配置
配置文件位于安装目录(默认： C:\Program Files\HoumoAI\hi_cpp)的config下面，sys.cfg
```ini
[server]
port = 8808

[log]
level = 0 # 0-Info, 1-Debug
location = C:\ProgramData\HoumoAI\hi\logs\

[model]
cur_model = D:\models\hmm_deepseek_256_4096_4cores_20250317\

```

修改配置后，需要重启HI Service， 可以打开Windows服务，找到HI Service，然后右键进行重启。

## 服务使用
### 推理服务访问
- API访问地址：http://ip:8808
  目前，后端API默认访问端口为8808，可以通过配置文件修改（需重启）。

### 日志
推理服务在终端提供实时日志信息，如果启动服务过程中出现错误，可查看相关日志信息进行简单排错；并且请求和响应数据对通过输出到日志文件进行后续查看和问题排查，可以将日志以.log形式存放在log文件夹内。

log文件路径默认在C:\ProgramData\HoumoAI\hi_cpp\logs\

### 服务故障排查
1. 安装失败：可以检查当前用户对于C盘的操作权限，确保默认可以新建文件
2. 启动失败：查看对应日志
3. 接口失败：查看返回消息&系统日志
4. 其他无法定位问题请联系HOUMO


## SDK使用

### 目录结果

```
hi.cpp\v1.2.0\sdk
.
|-- CMakeLists.txt
|-- bin
|   |-- asmjit.dll
|   |-- c10.dll
|   |-- cyghdplrt_asic.dll
|   |-- cyghm800_hal.dll
|   |-- cyghm800_hal_i2c.dll
|   |-- fbgemm.dll
|   |-- ggml-base.dll
|   |-- ggml-cpu.dll
|   |-- ggml.dll
|   |-- houmo-llm-demo-lenovo-14B.exe
|   |-- houmo-llm-demo-lenovo.exe
|   |-- houmo-llm.dll
|   |-- libiomp5md.dll
|   |-- libiompstubs5md.dll
|   |-- libwinpthread-1.dll
|   |-- llama.dll
|   |-- tcim_runtime_lite.dll
|   |-- torch.dll
|   |-- torch_cpu.dll
|   |-- torch_global_deps.dll
|   |-- utf8_console.dll
|   `-- uv.dll
|-- demo
|   |-- CMakeLists.txt
|   `-- src
|       |-- demo-lenovo-14B.cpp
|       `-- demo-lenovo.cpp
|-- include
|   |-- houmo-llm.h
|   |-- http_server_c_interface_3.h
|   |-- utils
|   |   `-- include
|   |       |-- utf8
|   |       |   |-- checked.h
|   |       |   |-- core.h
|   |       |   |-- cpp11.h
|   |       |   |-- cpp17.h
|   |       |   `-- unchecked.h
|   |       |-- utf8.h
|   |       `-- utf8_console.h
|   `-- utils.hpp
|-- lib
|   |-- houmo-llm.lib
|   `-- utf8_console.lib
|-- models
|   `-- deepseek_14b_256_8k_4cores
|       |-- DeepSeek-R1-Distill-Qwen-1.5B-IQ2_M.gguf
|       |-- deepseek_decode_part1.hmm
|       |-- deepseek_decode_part2.hmm
|       |-- deepseek_prefill_part1.hmm
|       |-- deepseek_prefill_part2.hmm
|       `-- hmquant
|           `-- quant_embedding_traced.pt
|-- run_demo_lenovo_win11.bat
```

### 构建编译
如目录结果所示，这是一个CMake工程，对应依赖的include与lib已经提供，存在两个demo示例：demo-lenovo-14B和demo-lenovo（7B)

```cmd
cmake -B build-release
cmake --build build-release -j 8 --config Release
```

### Demo运行
如目录结果所示，默认的模型防止在sdk目录下面，可以根据自己模型的位置进行修改编译，然后再运行测试。
