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

- 操作系统：麒麟OS v10/Ubuntu 20.04
- 力谋驱动版本：v2.3.0
- 力谋固件版本：v2.3.0（boot.img & loader.img）

## 服务部署

### 模型配置
目前应用软件将 DeepSeek 模型作为推理服务的默认模型配置，若用户需要修改使用模型，如：QWen2.5/3等其他模型时，
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

主包：houmo-application-software-model-deepseek-7b-4core-8k-1batch-v2.3.0.tar.gz

附包：houmo-application-software-model-deepseek-7b-embedding-cpp-v2.3.0.tar.gz

我们创建自己model文件夹，如 DeepSeek-R1-Distill-Qwen-7B-1Batch-4core_8k ， 然后
将主包中4core_8k目录下的hmm格式的模型文件全部复制到 DeepSeek-R1-Distill-Qwen-7B-1Batch-4core_8k 根目录， 然后将附包中内容也全部复制到 DeepSeek-R1-Distill-Qwen-7B-1Batch-4core_8k 根目录。
适配完成之后的目录树如下：

```
models/DeepSeek-R1-Distill-Qwen-7B-1Batch-4core_8k
$ tree
.
|-- DeepSeek-R1-Distill-Qwen-1.5B-IQ2_M.gguf
|-- decoder.hmm
|-- prefill.hmm
`-- hmquant
    |-- quant_embedding_traced.pt

```

### 安装过程
【注意】
- 安装本系统前请确保已安装好后摩推理加速卡、并且升级驱动和固件为指定版本
- 若有启动同类型的大模型推理系统，请先下线对应系统，以保证该服务的正常运行
- 请保证已获取到了完整的安装部署包（如应用或服务部署包、模型部署包等）

````bash
sudo ./hicpp-1.3.0.run
````

安装完成后，后摩的推理服务默认就开启了，执行下面的命令，查看推理服务是否正常启动，
尤其注意步骤2 models文件夹的路径和内容正确，否则下面的命令会输出load model失败之类的错误提示
当前版本-按麒麟需求，默认加载deepseek distll qwen模型。


## 管理服务
````bash
sudo systemctl start hicpp   # 启动服务-安装run包默认就已经开启了hicpp服务，不需要再start
sudo systemctl stop hicpp    # 停止服务
sudo systemctl restart hicpp # 重启服务
sudo systemctl status hicpp  # 查看状态
````

### 模型切换

如需切换模型，修改service文件HICPP_MODEL_PATH的变量，指向对应的模型文件夹路径。

````bash
sudo vi /etc/systemd/system/hicpp.service
````

比如：

Environment="HICPP_MODEL_PATH=/opt/hicpp/models/qwen3-8b-1batch/4core_8k" 对应qwen3-8B

Environment="HICPP_MODEL_PATH=/opt/hicpp/models/DeepSeek-R1-Distill-Qwen-7B-1Batch/4core_8k" 对应Deepseek-R1-Distill-Qwen-7B

````bash
#修改完成后：重启hicpp服务
sudo systemctl daemon-reload
sudo systemctl restart hicpp.service
````


## Docker

````bash
docker run --privileged -d --name HLIECppService -p 7901:7901 -v /opt/houmo/models:/opt/houmo/models -v /opt/houmo/logs/HLIECpp:/opt/HLIECpp/logs -it harbor.houmo.ai/application_software/hlie.cpp-xh1-20250910:v2.4.2-x86_64

````





