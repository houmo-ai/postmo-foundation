# Houmo Lanyue Inference Engine Cpp
HLIECpp是Houmo软件平台为Windows/Linux下打造的使用C++开发的推理服务

![M50](https://www.houmoai.com/assets/images/swiper_img/m50/m50_chip_banner_mb.jpg)

更多资讯欢迎关注

![HOUMO](https://houmo.cn/assets/images/code-wechat.jpg?v=592573c)


## 部署要求

1. 硬件要求（最低）

- CPU结构：x86_64
- 硬盘：200G
- AI硬件：后摩漫界®M50

2. 软件要求

- 操作系统：Windows 11/10
- 星汉2驱动版本：v0.6.0 (最新Release)
- 星汉2固件版本：v0.6.0 (最新Release)

## Windows服务部署

### 安装过程
【注意】
- 安装本系统前请确保已安装好后摩推理加速卡、并且升级驱动和固件为指定版本
- 若有启动同类型的大模型推理系统，请先下线对应系统，以保证该服务的正常运行
- 请保证已获取到了完整的安装部署包（如应用或服务部署包、模型部署包等）

1. 双击HLIECpp.exee开始执行安装
2. 同意安装协议，然后点击下一步
3. 开始选择安装服务的路径，可以修改路径， 如果选择C盘，需要确保当前用户在C盘有默认的管理员写入权限（ 可尝试在C盘某一个目录下面右键创建文件，查看是否被限制为单一需要管理员权限的文件夹选项），选择完成后点击下一步
4. 需要指定模型路径，请选择你放置对应模型位置，选到选定模型的跟路径，然后下一步
5. 直接点击下一步进行安装，等待安装完成
6. 至此，没有任何问题的话，服务会自动以Windows服务形式启动运行

## 服务配置
### 推理服务配置
配置文件位于安装目录(默认： C:\Program Files\HoumoAI\HLIECpp)的config下面，sys.cfg
```ini
[server]
port = 7901
# HOUMO设备ID, 以数字表示，默认只有0，多个设备用逗号隔开, 如 0,1
devices = 0
# 实例副本数量，会启动多个进程，端口会递增+1， 用于负载均衡，多个设备同时运行同一种模型
replicas = 1

[log]
# debug, info, warn, info
level = info

# 4-debug, 3-warn, 2-error 1-info
tcim_level = 2
# max files to keep
max_files = 7
# Just for Windows Mode, Linux Service use fixed path： /opt/HLIECpp/logs
location = C:\ProgramData\HoumoAI\HLIECpp\logs\

# SDK & Demo using
open_console_window = true

[llm_params]
#llama-server的参数透传，请参考下面或者llamacpp的设置, 请使用完整option: --xxx = xx， 没有值的请留空
--jinja =
--reasoning-format = none
#--reasoning_budget = 0

[model]
cur_model = D:\models


```

修改配置后，需要重启HLIECpp Service， 可以打开Windows服务，找到HLIECpp Service，然后右键进行重启。

## 服务使用
### 推理服务访问
- API访问地址：http://ip:7901
  目前，后端API默认访问端口为7901，可以通过配置文件修改（需重启）。

### 日志
推理服务在终端提供实时日志信息，如果启动服务过程中出现错误，可查看相关日志信息进行简单排错；并且请求和响应数据对通过输出到日志文件进行后续查看和问题排查，可以将日志以.log形式存放在log文件夹内。

log文件路径默认在C:\ProgramData\HoumoAI\HLIECpp\logs\

### 服务故障排查
1. 安装失败：可以检查当前用户对于C盘的操作权限，确保默认可以新建文件
2. 启动失败：查看对应日志
3. 接口失败：查看返回消息&系统日志
4. 其他无法定位问题请联系HOUMO

