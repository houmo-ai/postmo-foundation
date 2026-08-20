# hicpp Backend 接入约束

本文描述当前单一 HOUMO TCIM SDK 下的模型 backend 接入方式。`backend` 指 LLM、ASR、TTS、SD、Document 等模型能力，不代表可动态替换的硬件 SDK 插件。

## 两类变更

### 同一 backend 的新模型/新 artifact

优先只修改以下位置：

1. core `HLIEBackendDescriptor` 的 alias、marker、capability、memory/placement policy；
2. app `HLIEModelFamilyRegistry`/模型 metadata 解析；
3. 已有 adapter 的模型参数或 TCIM 调用；
4. descriptor、artifact parsing、Embedded/Process golden 和性能测试。

不得为一个新模型名在 Workflow、ApiRouter 或 ModelController 增加分支。模型无法被现有 descriptor 唯一识别时，应先修复 descriptor/metadata，而不是在 HTTP 路由中猜测字符串。

### 全新的模型能力/backend kind

接入点应包括：

1. 在 core 注册 stable backend id、capability、支持的 Runtime kind 和纯 policy；
2. 在 app 注册具体 `IHLIEBackendAdapter`；
3. 让 `IHLIEModelBackendLauncher` 返回中性 launch artifact；
4. 为 Embedded/Process 提供 `IHLIEInferenceEndpoint` capability dispatch；
5. 增加 lifecycle、资源回滚、stale generation、stream cancellation、API golden 和真机测试。

当前 P1-7 仍在迁移默认 embedded startup 与历史 status/alias handler。全新 service kind 如果必须修改 Workflow 类型分支，说明扩展点尚未完成，应先把该分支提取到 app application service/adapter，再接入 backend；不能把临时 Workflow switch 当成最终方案。

## 依赖边界

- core 不 include `HLIEModelInfo`、HTTP、TCIM/HAL/GGML、具体 Service 或文件系统。
- ModelController 不创建 Service、不读取 Singleton、不解析 artifact、不执行 HTTP。
- ResourceScheduler 不停止 Runtime；它只管理 reservation 和生成 residency plan。
- DiscoveryService 不写 runtime state；它只在双 snapshot 校验通过后提交 deferred catalog update。
- ApiRouter 不选择具体 Service；InferenceDispatcher 通过 capability 和 generation-bound endpoint dispatch。
- composition root 只连接当前 HOUMO 实现，不提供多 SDK provider/ABI 选择。

## 生命周期要求

- load 必须使用 generation ticket，并经过 reserve -> bind -> launch -> publish -> track/commit；
- backend 已启动但 Runtime publication 失败时，只有确认 cleanup 后才能释放 reservation；
- stop/unload/eviction 必须经 Controller，confirmed stop 前不得发布 Unloaded 或释放硬件账本；
- stale READY/exit/response 必须 fail-closed，不能修改新 generation；
- TCIM 不支持的执行中硬取消必须停止后续输出、保留资源事实并记录 cancellation lag，不能伪造已取消。

## 日志与指标

- `DEBUG`：高频路由、匹配细节、安全拒绝的 stale event；
- `INFO`：成功 load/READY/commit/unload、预期 stop、正常策略驱逐；
- `WARN`：可恢复的容量拒绝、busy、fallback、未完成 eviction；
- `ERROR`：未恢复失败、不变量破坏、未确认 cleanup。

新增 backend 必须提供 active/rejected/failure/latency 或 backend 等价指标；轮询不得持续打印 INFO/DEBUG 状态，状态值应通过 health/Prometheus 暴露。

## 合入检查

- Windows/Linux Release CTest 与全部 architecture gate；
- Linux ASAN+UBSAN、TSAN control-plane；
- Embedded/Process API golden，包含 streaming disconnect；
- 主进程模型、Worker crash/recovery、graceful unload、ledger baseline；
- warm-up 后性能对比，控制面改动不得让 steady-state 请求边界回退超过既定 3% 门槛；
- 涉及线程、catalog cache、Runtime handle 或 Service 生命周期时执行短时 mixed workload，并审计 handle/thread/RSS/manifest。
