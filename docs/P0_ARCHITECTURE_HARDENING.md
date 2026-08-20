# HLIECpp P0 架构治理详细设计

状态：Draft
更新时间：2026-08-15
适用范围：`hlie-app` 控制面、模型生命周期、设备资源管理、请求准入、模型子进程和系统级测试

## 1. 背景与目标

HLIECpp 当前已经具备 LLM、视觉语言、Embedding、Rerank、ASR、TTS、图像生成和 PDF OCR 等完整产品能力，LLM 后端也已经具备 continuous batching、并行 slot、KV cache、取消和 speculative decoding 等执行能力。

当前的主要风险不在 decode loop，而在 decode loop 之上的控制面：

- 模型状态 claim、设备选择、显存预留、Service 发布不是一个原子或可回滚事务。
- `HLIEModelServiceManager` 暴露可变 `shared_ptr` 引用，生命周期同步依赖调用方约定。
- 部分模型服务使用无界队列，框架缺少统一的过载拒绝、deadline 和取消传播。
- 子进程状态依赖被动查询，进程退出与 Registry 状态可能长时间不一致。
- 局部单测覆盖较好，但关键的并发加载、卸载、进程故障和真机长稳场景缺少自动化验证。

P0 的目标不是重写后端，也不是一次性拆完 `HLIEWorkflow`，而是先建立以下生产级不变量：

1. 同一个模型 generation 在任意时刻只有一个有效运行实体。
2. `LOADED/READY` 必须同时意味着 Service/Worker 可用、资源预留已提交、请求可以安全获取运行时 lease。
3. 任何模型加载失败都能完整回滚 Registry、Service slot、设备 reservation、子进程和等待者状态。
4. 所有推理队列有界；请求断开、deadline 到期和服务停止能够传播到排队及执行中的任务。
5. 子进程异常退出后，控制面能够在确定时间内收敛到 `FAILED/UNLOADED`，不能继续对外宣称模型已加载。
6. 不允许捕获业务对象裸 `this` 的 detached 后台线程。
7. P0 关键路径必须具备并发测试、故障注入测试、sanitizer 和真机长稳验证。

## 2. 优先级与实施顺序

| 编号 | 工作项 | 主要风险 | 前置依赖 |
| --- | --- | --- | --- |
| P0-1 | Service 生命周期并发安全 | data race、UAF、重复加载、错误 Service 被发布 | 无 |
| P0-2 | 事务式设备资源预留 | 显存超卖、错误驱逐、加载失败资源泄漏 | P0-1 的 generation/ticket |
| P0-3 | 统一准入与过载保护 | 无界排队、HTTP 线程耗尽、取消无效 | P0-1 的运行时 lease |
| P0-4 | WorkerSupervisor | 子进程崩溃后状态陈旧、僵尸进程、错误路由 | P0-1、P0-2 |
| P0-5 | 系统级测试与故障注入 | 无法证明上述不变量成立 | 与 P0-1～P0-4 同步建设 |
| P0-6 | 最小可观测性与健康语义 | 故障不可发现、无法验收 P0 | 与所有工作项同步建设 |

建议按照 P0-1 → P0-2 → P0-3 → P0-4 的顺序合入。P0-5 和 P0-6 不应放在最后补，而应随每个工作项一起提交。

## 3. P0 目标结构

```text
HTTP / WebSocket
       |
       v
RequestContext
       |
       v
AdmissionController ---- queue/deadline/cancellation ----+
       |                                                  |
       v                                                  |
ModelController                                           |
  |        |            |                                 |
  |        |            +--> WorkerSupervisor             |
  |        |                    |                         |
  |        |                    +--> Process Worker       |
  |        |                    +--> Embedded Worker (*)  |
  |        |                                              |
  |        +--> ResourceManager --> DeviceLedger          |
  |                                                       |
  +--> RuntimeDirectory --> ServiceHandle/WorkerHandle <--+
             |
             +--> ModelRegistry（对外状态和驻留元数据）
```

`(*)` P0 不要求立即完成主进程本地模型与子进程模型的完整统一，但新接口必须允许后续把本地模型表示为 `EmbeddedWorker`，避免继续增加本地/子进程分支。

## 4. 全局状态与并发不变量

### 4.1 模型 generation

每次从非运行状态进入加载流程时分配单调递增的 `generation`：

```cpp
using ModelGeneration = std::uint64_t;

struct ModelRuntimeId {
    std::string model_key;
    ModelGeneration generation = 0;
};
```

所有异步回调、Service 发布、Worker ready、Worker exit、资源 commit 和 Registry 更新都必须携带 generation。旧 generation 的迟到事件必须被忽略，不得覆盖新实例状态。

### 4.2 P0 运行时状态机

建议在 HLIECpp 控制面定义自己的状态，不再只依赖上游 `server_model_status`：

```text
UNLOADED
   |
   v
RESERVING -> LOADING -> READY -> DRAINING -> STOPPING -> UNLOADED
    |          |          |          |           |
    +----------+----------+----------+-----------+--> FAILED
```

状态含义：

- `UNLOADED`：无有效 Service/Worker，无已提交资源。
- `RESERVING`：已取得生命周期 claim，正在生成资源计划或 reservation。
- `LOADING`：reservation 已建立，Service/Worker 正在加载。
- `READY`：运行实体已就绪，资源已 commit，可以接受新请求。
- `DRAINING`：拒绝新请求，等待 active request 归零。
- `STOPPING`：运行实体正在停止，不得持有 Workflow/Registry/Manager 锁执行 stop。
- `FAILED`：本 generation 加载或运行失败，记录稳定的 failure reason 和 exit code。

OpenAI `/v1/models` 仍可映射到 loaded/loading/unloaded 等兼容状态，但内部状态不能被兼容枚举限制。

### 4.3 必须始终成立的不变量

- Registry 为 `READY` 时，RuntimeDirectory 中必须存在相同 model key 和 generation 的可用 handle。
- RuntimeDirectory 中的 handle 被移除后，新的 request lease 必须立即失败。
- active request 大于零时正常卸载不能直接进入 `STOPPING`，只能先进入 `DRAINING`；drain deadline 到期后必须先触发取消，再允许强制停止。
- pending reservation 和 committed reservation 必须有唯一 reservation id，所有退出路径只能结束一次。
- Service `init/run/stop`、HAL 查询、进程创建/等待/终止、HTTP 调用均不得在 Registry 或 ServiceManager 内部锁下执行。
- 组件内部 mutex 不互相嵌套。跨组件一致性由 per-model lifecycle ticket、generation 校验和补偿回滚保证。

## 5. P0-1：Service 生命周期并发安全

### 5.1 当前问题

当前 Manager：

- 通过 `std::shared_ptr<T>&` 暴露内部 slot。
- Manager 自身没有同步机制。
- Workflow、路由、卸载和状态查询直接读取同一个 `shared_ptr` 成员。
- 同模型 loading claim 由 Registry 保护，但不同模型争用同一个 typed slot 时仍可能并发进入。
- 部分启动状态通知使用捕获 `this` 的 detached thread。

Registry 的 mutex 只能保护 Registry 数据，不能保护 Manager 的 `shared_ptr` 对象。

### 5.2 P0 最小改造方案

P0 暂时保留 typed slot，但必须将同步收回 Manager 内部：

```cpp
struct ServiceSnapshot {
    ModelRuntimeId runtime_id;
    ModelServiceKind kind;
    std::shared_ptr<IHLIEModelService> service;
};

struct ServiceLoadTicket {
    ModelRuntimeId runtime_id;
    ModelServiceKind kind;
    std::uint64_t slot_epoch = 0;
};

class HLIEModelServiceManager {
public:
    Expected<ServiceLoadTicket, ServiceError> begin_load(
        const std::string& model_key,
        ModelGeneration generation,
        ModelServiceKind kind);

    bool publish(const ServiceLoadTicket& ticket,
                 std::shared_ptr<IHLIEModelService> service);

    std::optional<ServiceSnapshot> snapshot(ModelServiceKind kind) const;

    std::optional<ServiceSnapshot> detach(
        const ModelRuntimeId& runtime_id,
        ModelServiceKind kind);

    void fail_load(const ServiceLoadTicket& ticket);
};
```

约束：

- 删除所有返回 `shared_ptr&` 的 getter。
- getter 只返回 `shared_ptr` 副本或 `ServiceSnapshot`。
- `begin_load()` 在 Manager 锁内把 slot 标记为 `LOADING`，防止不同模型同时占用同一 typed slot。
- Service 对象创建和 `init()` 在 Manager 锁外执行。
- `publish()` 重新校验 `slot_epoch + generation`；若 ticket 已失效，停止刚创建的 Service 并回滚。
- `detach()` 只在锁内交换所有权，不在锁内调用 `stop()`。

### 5.3 运行时请求 lease

现有 `ModelRequestLease` 只保护 Registry 计数。P0 应升级为同时固定 Registry generation 和 Service handle：

```cpp
class ModelRuntimeLease {
public:
    const ModelRuntimeId& runtime_id() const;
    const std::shared_ptr<IHLIEModelService>& service() const;
    bool active() const;

private:
    ModelRuntimeId runtime_id_;
    std::shared_ptr<IHLIEModelService> service_;
    HLIEModelRegistry* registry_ = nullptr;
};
```

获取顺序：

1. Registry 对指定 model key/generation 执行 `begin_request_ticket()`。
2. Manager/RuntimeDirectory 获取相同 generation 的 ServiceSnapshot。
3. 若 Service 不存在、generation 不一致或状态不是 READY，立即释放 Registry request count 并失败。
4. 返回 lease。流式响应继续通过响应包装器持有 lease，直到 EOF、取消或连接关闭。

路由层不得再次直接访问 `model_service_manager_.llm()` 等可变 slot，而应只使用 lease 中固定的 Service handle。

### 5.4 生命周期事务

本地 Service 加载建议统一为：

```text
ModelController.begin_load(model_key)
  -> Registry: claim generation, state=RESERVING
  -> ServiceManager: begin_load(), obtain ticket
  -> ResourceManager: reserve(), obtain reservation
  -> Registry: state=LOADING
  -> Factory: create service
  -> service.init()                     // no control-plane lock
  -> service.run()                      // no control-plane lock
  -> ServiceManager.publish(ticket)
  -> ResourceManager.commit(reservation)
  -> Registry: mark READY with generation
  -> notify waiters
```

任意一步失败：

```text
stop temporary service if created
ServiceManager.fail_load(ticket)
ResourceManager.rollback(reservation)
Registry.mark_failed(generation, reason)
notify waiters
```

不得通过散落的 `rollback_model_start()` 调用依赖开发者记住所有资源；应使用显式 transaction/scope guard 汇总回滚动作。

### 5.5 后台任务生命周期

- 所有自动加载、延迟状态通知、扫描和 TTL 工作线程必须由 `BackgroundTaskGroup`、`std::jthread` 或专用 worker 类持有。
- `HLIEWorkflow::stop()` 必须先停止任务接收，再请求任务取消，最后 join。
- 后台任务若允许抛异常，线程入口必须捕获并记录；不能在 `std::thread` 中重新抛出导致 `std::terminate`。
- 禁止新增 `.detach()`，除非线程完全不捕获业务对象且所有状态均由独立 shared state 管理，并经过专项评审。

### 5.6 P0-1 验收条件

- 并发启动两个不同 LLM 时，只允许一个占用本地 LLM slot，另一个得到确定结果或进入子进程路径。
- 1000 次 request/unload 交错测试无 data race、UAF 和 active request 计数泄漏。
- Service getter 不再返回引用；Workflow 中不存在对 Manager 内部 `shared_ptr` 的无保护读写。
- Workflow 中捕获 `this` 的 detached 状态线程清零。

## 6. P0-2：事务式设备资源预留

### 6.1 当前问题

当前设备准入分为 dry-run 和真实 acquire。dry-run 不保留资源，真实可用性主要查询 HAL；框架内部增加的逻辑占用不会稳定参与下一次判断，并可能被 refresh 覆盖。

此外，当前 GGUF 估算虽然计算了 KV cache，但最终结果没有直接纳入 KV cache、context、parallel slot、mmproj、文档 layout 和其他附属模块。

### 6.2 DeviceLedger

ResourceManager 为每个设备维护独立账本：

```cpp
struct DeviceLedgerEntry {
    int device_id = -1;
    std::uint64_t total_mb = 0;

    // 最近一次 HAL 观测；不覆盖框架账本。
    std::uint64_t observed_used_mb = 0;
    std::uint64_t observed_available_mb = 0;
    std::chrono::steady_clock::time_point observed_at;

    // 已被框架接受但尚未完成加载。
    std::uint64_t pending_reserved_mb = 0;

    // 已加载模型的框架账面占用。
    std::uint64_t committed_reserved_mb = 0;

    // 已 commit 但尚未在 HAL observed_used 中体现的部分。
    std::uint64_t committed_unobserved_mb = 0;

    bool exclusive = false;
};
```

保守可用量：

```text
effective_used = observed_used_mb
                 + committed_unobserved_mb
                 + pending_reserved_mb

admissible_mb = total_mb - effective_used - safety_margin_mb
```

`observed_used_mb` 可能已经包含 committed 模型，所以不能把全部 `committed_reserved_mb` 再次相加。reservation commit 时先增加 `committed_unobserved_mb`；后续 HAL 观测到显存上涨后，reconciler 将可归因的增量从 `committed_unobserved_mb` 中扣除。这样既避免重复计算，也避免模型刚开始分配、HAL 尚未反映时出现准入窗口。

如果 observed 变化无法可靠归因，P0 应选择保守策略：保持对应 unobserved 占用，直到模型进入 READY 后连续多次观测稳定，或者该 reservation 被 release。宁可暂时降低利用率，也不能通过猜测释放账面占用。

### 6.3 ResourceRequest 与 MemoryBreakdown

```cpp
struct MemoryBreakdown {
    std::uint64_t weights_mb = 0;
    std::uint64_t kv_cache_mb = 0;
    std::uint64_t runtime_workspace_mb = 0;
    std::uint64_t auxiliary_models_mb = 0;
    std::uint64_t safety_margin_mb = 0;

    std::uint64_t total_mb() const;
};

struct ResourceRequest {
    ModelRuntimeId runtime_id;
    ModelServiceKind kind;
    std::size_t device_count = 1;
    std::vector<int> requested_devices;
    MemoryBreakdown memory;
    bool exclusive = false;
};
```

内存估算必须由 backend-aware provider 生成：

```cpp
class IMemoryEstimator {
public:
    virtual Expected<MemoryBreakdown, EstimateError>
    estimate(const HLIEModelInfo&, const EffectiveLaunchParams&) const = 0;
};
```

LLM 至少考虑：

- GGUF tensor/weight 实际大小与量化类型。
- context size、parallel slots、KV 数据类型、block/head/key/value 维度。
- batch/ubatch workspace。
- mmproj 和视觉编码器。
- speculative draft/MTP 附加模型与 buffer。
- document layout module 等随模型生命周期加载的附属模块。
- backend 固定开销和可配置安全余量。

ASR/TTS/SD 应由各自 provider 返回权重、workspace 和附属模型占用，不能继续依赖模型名称的统一固定值。

### 6.4 Reservation API

```cpp
enum class ReservationState {
    Pending,
    Committed,
    RolledBack,
    Released,
};

struct ResourceReservation {
    std::string reservation_id;
    ModelRuntimeId runtime_id;
    std::vector<std::pair<int, std::uint64_t>> device_memory_mb;
    ReservationState state = ReservationState::Pending;
    std::chrono::steady_clock::time_point expires_at;
};

class ResourceManager {
public:
    Expected<ResourceReservation, ResourceError> reserve(const ResourceRequest&);
    bool commit(const std::string& reservation_id, const ModelRuntimeId&);
    bool rollback(const std::string& reservation_id, const ModelRuntimeId&);
    bool release(const std::string& reservation_id, const ModelRuntimeId&);
    void reconcile();
};
```

规则：

- `reserve()` 在一个 ResourceManager 临界区内完成 HAL refresh、设备选择和 pending 记账，不能再提供不记账的 dry-run 作为加载准入接口。
- `commit/rollback/release` 必须幂等，并校验 model key、generation 和 reservation id。
- pending reservation 设置 TTL，控制面崩溃式失败或加载线程失联后可由 reconciler 回滚。
- TTL 回收必须验证所有权活性：当前 generation 仍处于 LOADING/LOADED，或对应 Worker 仍存活时，只能继续保护，不能仅凭 wall-clock 到期释放。动态 Worker 失联优先由进程退出/READY timeout 立即收敛；TTL 只处理 owner 已消失但显式清理遗漏的 orphan。
- 主进程本地 backend load 永久卡死时，安全终止加载并确认硬件已释放需要独立 watchdog/backend cancel 语义或进程隔离。该能力作为后续控制面增强项，不阻塞本阶段事务式资源账本验收；在实现前必须保持保守占用，不能通过 TTL 猜测释放。
- release 只能在 Service/Worker 已确认停止后执行，避免模型仍占用显存时提前放行新加载。
- `refresh_device_usage()` 只更新 observed 字段，不得覆盖 pending/committed 账本。
- reconciler 负责把已被 HAL 观测到的 committed 增量从 `committed_unobserved_mb` 中消化，但不能直接修改 reservation 的所有权和生命周期状态。
- `release_model()` 不能保持空实现；应按 runtime id/reservation id 释放，而不是按易冲突的模型名称释放。

### 6.5 多模型驱逐与重新准入

资源不足时：

1. 根据目标设备和每卡所需内存计算缺口。
2. ResidencyManager 选择可驱逐集合，而不是默认只尝试一个候选。
3. 候选进入 `DRAINING`，停止接受新请求。
4. active request 归零后停止 Service/Worker并释放 committed reservation。
5. 重新执行完整 `reserve()`，不能复用旧 dry-run 结果。

驱逐选择应考虑“释放后是否能满足目标设备缺口”，避免驱逐了模型但仍无法加载目标模型。

### 6.6 P0-2 验收条件

- 100 个并发 reservation 请求不会使任一设备账本超过总容量。
- HAL 观测刷新不会清除 pending/committed reservation。
- 任意加载失败路径最终 pending reservation 为零。
- 同一个 reservation 的重复 commit/rollback/release 不会产生下溢。
- 大 context、多 parallel slot 的估算结果随参数单调增加。
- 模型停止完成之前，相应显存不能被新加载事务重新预留。

## 7. P0-3：统一准入、队列和取消

### 7.1 RequestContext

所有 HTTP、WebSocket 和内部 pipeline 请求进入模型服务前统一构造：

```cpp
struct RequestContext {
    std::string request_id;
    std::string model_key;
    ModelGeneration generation = 0;
    std::chrono::steady_clock::time_point deadline;
    std::function<bool()> is_cancelled;
    int priority = 0;
    std::string tenant_id;
};
```

P0 可以暂不实现复杂租户计费和 cost-based scheduler，但字段应保留，避免再次修改所有模型服务接口。

### 7.2 AdmissionController

```cpp
struct AdmissionPolicy {
    std::size_t max_running = 1;
    std::size_t max_queued = 0;
    std::chrono::milliseconds max_queue_wait{0};
};

class AdmissionTicket {
public:
    bool admitted() const;
    bool cancelled() const;
    bool expired() const;
};

class AdmissionController {
public:
    Expected<AdmissionTicket, AdmissionError>
    acquire(const RequestContext&, const AdmissionPolicy&);
};
```

每个模型 generation 拥有独立 admission state：

```text
QUEUED -> RUNNING -> FINISHED
   |         |
   +--------> CANCELLED
   +--------> EXPIRED
```

### 7.3 各后端接入方式

- LLM：保留 llama-server 的 continuous batching 和 slot scheduler；HLIECpp 在进入后端队列前增加有界 front-door admission，限制等待任务数和最大等待时间。
- Whisper/SD/CosyVoice：构造 `HLIEBlockingQueue` 时必须传入有限 `max_size`；逐步迁移到统一 executor。
- Qwen3-TTS：不能只依赖一个大 mutex 让 HTTP worker 无界等待；先进入 admission queue，取得 running ticket 后再进入模型临界区。
- 文档 pipeline：PDF layout 与 OCR 可能是复合请求，必须持有同一个 deadline/cancellation，并为内部区域 OCR 设置并发上限，不能绕过外层 admission。
- ASR WebSocket：保留已有 backpressure，同时接入统一模型 generation 和 shutdown draining。

### 7.4 取消与 deadline 语义

- 请求排队时客户端断开：立即从队列移除并释放 admission ticket。
- 请求开始前 deadline 到期：不执行推理。
- 请求执行中取消：调用 backend 支持的 cancel/stop；不支持硬取消的后端至少停止后续流输出，并记录 cancellation lag。
- 流式响应的 `ModelRuntimeLease` 和 AdmissionTicket 持续到响应完成回调或连接关闭。
- 服务进入 `DRAINING` 后拒绝新 admission，但允许已 RUNNING 请求完成到 drain deadline。

建议错误映射：

| 场景 | HTTP 状态 | OpenAI error type |
| --- | --- | --- |
| 队列已满 | 429 | `rate_limit_error` |
| 模型 draining/stopping/failed | 503 | `unavailable_error` |
| 排队 deadline 到期 | 504 | `timeout_error` |
| 客户端已取消 | 499（内部/兼容扩展） | `cancelled_error` |
| 请求参数不可执行 | 400 | `invalid_request_error` |

### 7.5 配置草案

配置命名在实现前需加入 `config_schema.json`，建议新增 `[admission]`：

```ini
[admission]
llm_max_queued = 64
whisper_max_queued = 8
tts_max_queued = 8
sd_max_queued = 2
document_max_queued = 4
default_queue_timeout_ms = 30000
drain_timeout_sec = 120
```

默认值需通过真机压测确定。首次上线应偏保守，并允许按模型 preset 覆盖。

### 7.6 P0-3 验收条件

- 任一模型服务等待队列均存在确定上限。
- 队列满时请求在 100 ms 内得到明确拒绝，不继续占用推理队列。
- 排队请求取消后不执行 backend 推理。
- shutdown/draining 期间没有新任务进入，所有已接收任务在 deadline 内完成或取消。
- 压测时 HTTP 线程数、排队任务数和内存占用均有上限。

## 8. P0-4：WorkerSupervisor

### 8.1 职责边界

`HLIEProcessManager` 降级为低层进程原语：spawn、graceful signal、wait、force kill、获取 exit code。新的 `WorkerSupervisor` 负责业务生命周期：

- Worker id、model key、generation、PID、端口和设备记录。
- 启动 deadline 和 ready handshake。
- 持续存活监控与 heartbeat。
- Worker exit 与 Registry/ResourceManager 自动对账。
- graceful drain、stop deadline 和 force kill。
- 可选重启策略与退避。

### 8.2 Worker 状态机

```text
CREATED -> STARTING -> READY -> DRAINING -> STOPPING -> EXITED
               |         |          |           |
               +---------+----------+-----------+--> FAILED
```

```cpp
struct WorkerRecord {
    std::string worker_id;
    ModelRuntimeId runtime_id;
    ProcessID process_id = 0;
    int port = 0;
    std::vector<int> devices;
    WorkerState state = WorkerState::Created;
    int exit_code = 0;
    std::string failure_reason;
    std::chrono::steady_clock::time_point last_heartbeat;
};
```

### 8.3 启动与 ready handshake

父进程创建 launch ticket：

```cpp
struct WorkerLaunchTicket {
    std::string worker_id;
    ModelRuntimeId runtime_id;
    std::string auth_token;
    int reserved_port = 0;
    std::string launch_manifest;
};
```

启动流程：

1. ResourceManager 建立 pending reservation。
2. Supervisor 分配并实际检查端口可用性，不能只假设 `master_port + instance_id` 一定空闲。
3. manifest 写入 worker id、generation、reservation id、父进程地址和一次性 token。
4. spawn 后进入 STARTING，并启动 deadline。
5. Worker 完成模型加载后调用内部 ready endpoint 或 IPC，携带 worker id、generation、PID、端口、设备和 backend version。
6. Supervisor 校验 ticket 和 generation 后 commit reservation，更新 RuntimeDirectory/Registry 为 READY。
7. 超时、进程提前退出或 ready 信息不匹配时终止 Worker并回滚。

内部状态接口不应继续与公开 `/models/status` 完全共用。至少使用独立路径、一次性 token 和 generation 校验，避免旧 Worker 或外部请求覆盖当前模型状态。

当前端口分配实现采用两段式防护：父进程在串行分配区间内实际 bind 探测连续候选端口，跳过冲突端口后再 spawn；动态子进程在模型加载前对目标端口再次 bind 预检，若探测到抢占则立即失败并由现有 failure reconciliation 回滚。该方案解决“已知占用端口仍启动并长时间停留 LOADING”的问题，但不是原子端口租约：父进程探测释放 socket 到子进程监听之间仍存在极短 TOCTOU 窗口。只有在验收要求升级为严格零竞态时，才需要增加 socket handoff 或平台句柄继承。

### 8.4 持续监督与状态对账

Supervisor 使用专用 `std::jthread` 或平台事件等待：

- Windows：进程 handle wait + 可选 heartbeat。
- Linux：`waitpid`/pidfd + 可选 heartbeat。
- 建议 heartbeat interval 2 秒，timeout 6～10 秒，最终值配置化。
- 进程已退出时以 OS 状态为准，不等待下一次 `/models` 或 `/status` 查询。

异常退出处理：

```text
detect exit(worker_id, generation)
  -> ignore if generation is stale
  -> RuntimeDirectory remove handle/route
  -> Registry mark FAILED(exit_code, reason)
  -> reject new admissions
  -> cancel/fail queued requests
  -> release committed reservation after process exit confirmed
  -> notify model-load waiters
  -> optional restart policy
```

P0 默认建议 `max_restarts = 0`，先保证状态正确；自动重启在确认模型失败不会形成崩溃循环后再开启。

### 8.5 优雅停止

1. Registry/RuntimeDirectory 标记 DRAINING，AdmissionController 拒绝新请求。
2. 等待 active request 归零或 drain deadline。
3. 调用 Worker 内部 shutdown endpoint/IPC。
4. 等待 graceful stop timeout。
5. 超时后发送 TERM/CTRL_BREAK；再次超时才 force kill。
6. OS 确认进程退出后释放 reservation，状态转 UNLOADED。

Windows 不应把 `TerminateProcess` 作为正常停止的第一选择。

### 8.6 主进程本地模型

P0 保留当前本地模型以降低改造范围，但应建立最小 `WorkerRecord`：

- `worker_id = embedded:<model_key>:<generation>`。
- state、generation、资源 reservation 和 failure reason 使用相同语义。
- 路由仍可直接调用 Service，不强制增加本地 HTTP hop。
- P1 再把本地模型完整接入 `EmbeddedWorkerTransport`，消除 Workflow 中两套生命周期分支。

### 8.7 P0-4 验收条件

- Worker 启动后未 ready 能在配置 deadline 内失败并完整回滚。
- Worker 被外部 kill 后，Registry 在 heartbeat/OS wait SLA 内离开 READY。
- 旧 generation 的迟到 ready/status 不能覆盖新 Worker。
- 端口冲突不会导致模型保持 LOADING 五分钟；应在启动阶段快速失败或重新分配。
- 正常卸载优先优雅退出，超时才 force kill。
- 主进程停止后无遗留 Worker、进程 handle、临时 manifest 和 pending reservation。

## 9. P0-5：系统级测试与故障注入

### 9.1 为可测试性引入的接口

P0 不应直接在测试中依赖真实 HAL 和真实模型。建议抽象：

```cpp
class IHalProvider;
class IProcessLauncher;
class IServiceFactory;
class IClock;
class ITaskExecutor;
class IPortAllocator;
```

生产实现包装现有 HAL、CreateProcess/fork、Factory 和 steady clock；测试实现可控制内存变化、进程退出、时间推进、加载阻塞与异常。

### 9.2 测试矩阵

#### 单元测试

- ServiceManager 同 slot 并发 `begin_load()` 只有一个成功。
- publish、detach、fail_load 对旧 generation/epoch 幂等失败。
- DeviceLedger 并发 reserve 不超卖。
- commit/rollback/release 重复调用不下溢。
- Admission 队列满、deadline、取消、draining 行为。
- Supervisor ready 超时、旧 generation、非零 exit code和 graceful timeout。

#### 组件测试

- Fake Worker 完成 spawn → ready → infer → drain → exit 全流程。
- Worker 启动后立即退出。
- Worker ready 后崩溃，Registry 与资源账本自动收敛。
- 模型加载到一半取消或主进程 stop。
- 代理流式响应中客户端断开，request lease 和 admission ticket 被释放。

当前已提供不依赖真实 HAL/模型的跨平台 Fake Worker，可模拟健康检查、推理、ticket 鉴权 shutdown、ready 前退出、ready 后崩溃和挂起。Windows、Linux Release CTest 已覆盖 ProcessManager 的真实 spawn、协议交互、expected/unexpected exit 分类和强制清理；Registry/DeviceLedger 的真机父进程端到端收敛也已由硬件 smoke/soak 覆盖。Linux 可在普通用户下启动，日志目录不可写时自动回退到用户态 state 目录。

Sanitizer 门禁现已落地：ASAN+UBSAN 使用正常 Linux 测试构建覆盖 unit、控制面 stress 和 Fake Worker supervisor；TSAN 使用 `HLIE_TESTS_ONLY=ON` 的纯控制面目标，避免把不可插桩的 TCIM/HAL/推理 SDK 带入竞态检测进程。统一入口为 `ci/run_linux_sanitizers.sh`，并强制 C/C++ 使用同版本编译器运行时。

#### 并发回归测试

- 同模型 100 个 auto-load 请求，只创建一个 generation/Worker。
- 两个不同 LLM 同时争用本地 slot。
- 多种模型同时竞争同一设备。
- request/unload、request/TTL eviction、manual unload/auto eviction 交错。
- discovery 更新与模型 loading 同时发生。

#### 真机与长稳测试

- 不同 context、parallel、模型大小的估算值与实际峰值对比。
- 连续 load/unload 500 次，无显存和 handle 累积。
- 24 小时 mixed workload soak，无持续内存增长、队列增长、僵尸进程和死锁。
- 模型 Worker 随机 kill、请求随机断开、网络代理随机超时。
- 记录吞吐、TTFT、P95/P99、队列时间和失败率基线。

### 9.3 CI 建议

- Windows Release：构建 + 全部单测 + Fake Worker 组件测试。
- Linux Debug：单测 + ASAN/UBSAN。
- Linux TSAN：只构建不依赖专有推理库的控制面测试 target。
- 真机 Nightly：模型生命周期、OOM、进程故障和短时压力测试。
- 真机 Weekly：长稳和性能回归。

当前 `hlie-testable-core` 应逐步加入新的 ModelController、ServiceManager、ResourceManager、AdmissionController 和 WorkerSupervisor，而不是继续绕开控制面核心。

## 10. P0-6：最小可观测性和健康语义

完整 tracing 可以放到 P1，但 P0 必须具备验证正确性的最小观测面。

### 10.1 健康接口

- `/live`：进程事件循环可响应，不检查模型是否加载。
- `/ready`：控制面初始化完成；DeviceLedger 可用；所有宣称 READY 的模型均存在有效 runtime handle；关键 Worker 未失联。
- `/health`：保持兼容，但响应中增加 `live/ready/degraded` 摘要，不应在无模型或 Worker 已崩溃时无条件返回简单 `ok`。

### 10.2 P0 指标

至少暴露：

- `hlie_model_state{model,kind}`
- `hlie_model_active_requests{model}`
- `hlie_model_load_duration_seconds{model,kind,result}`
- `hlie_model_load_failures_total{model,kind,reason}`
- `hlie_admission_running{model,kind}`
- `hlie_admission_queued{model,kind}`
- `hlie_admission_rejected_total{model,kind,reason}`
- `hlie_device_observed_used_mb{device}`
- `hlie_device_pending_reserved_mb{device}`
- `hlie_device_committed_reserved_mb{device}`
- `hlie_worker_state{worker,model}`
- `hlie_worker_restarts_total{model}`
- `hlie_worker_heartbeat_age_seconds{worker}`
- `hlie_backend_cancellation_lag_observations_total{lane,reason}`
- `hlie_backend_cancellation_lag_milliseconds_sum{lane,reason}`
- `hlie_backend_cancellation_lag_milliseconds_max{lane,reason}`

模型路径、prompt、API key 等敏感或高基数字段不得作为 label。

### 10.3 结构化失败信息

Registry/状态接口至少记录：

```json
{
  "state": "failed",
  "generation": 42,
  "failure_phase": "worker_startup",
  "failure_code": "ready_timeout",
  "failure_message": "worker did not report ready before deadline",
  "exit_code": 1,
  "updated_at": "..."
}
```

## 11. 建议代码落点

为避免继续扩大 `hlie_workflow.cpp`，P0 新代码建议按职责建立目录：

```text
include/hlie_app/runtime/
  hlie_model_controller.h
  hlie_runtime_types.h
  hlie_runtime_lease.h

include/hlie_app/resource/
  hlie_resource_manager.h
  hlie_memory_estimator.h

include/hlie_app/admission/
  hlie_admission_controller.h
  hlie_request_context.h

include/hlie_app/worker/
  hlie_worker_supervisor.h
  hlie_process_launcher.h
  hlie_port_allocator.h
```

P0 期间 `HLIEWorkflow` 仍负责 HTTP request/response 适配，但生命周期步骤应逐步委托给 `ModelController`。不得只是把 4200 行 Workflow 按文件切开而保留原有共享状态和锁；拆分必须伴随所有权迁移。

## 12. 分阶段迁移计划

### M0：基线与护栏

- 固化当前 110 个单测和 Release 构建基线。
- 增加控制面 fake 接口和最小状态/指标结构。
- 添加禁止新 detached thread 的代码评审规则。

### M1：ServiceManager 安全化

- getter 改为 snapshot。
- 引入 generation、slot epoch 和 load ticket。
- 现有 route 逐个迁移到 `ModelRuntimeLease`。
- 移除捕获 `this` 的 detached 状态线程。

### M2：ResourceManager

- 引入 DeviceLedger 和 reservation。
- 迁移 LLM 加载，再迁移 Whisper/TTS/SD。
- 修正内存估算并增加实际峰值对比工具。
- ResidencyManager 改为根据设备缺口选择驱逐集合。

### M3：AdmissionController

- 先限制现有 BlockingQueue 大小并提供快速拒绝。
- 接入统一 RequestContext、deadline 和 cancellation。
- 最后在 LLM backend 前增加 front-door queue limit。

### M4：WorkerSupervisor

- ProcessManager 降级为进程原语。
- 引入 launch ticket、ready deadline、generation 校验和 OS exit watcher。
- Registry/ResourceManager 自动对账。
- 加入 graceful shutdown，再保留 force kill 兜底。

### M5：真机验收

- 并发 load/unload 和随机故障注入。
- 显存估算校准。
- 24 小时 mixed workload soak。
- 建立 P95/P99、TTFT、TPS、失败率和资源泄漏基线。

以一名熟悉现有代码的高级 C++ 工程师估算，P0 主体约为 4～6 周；若同时补齐真机自动化和跨平台 supervisor，建议按 6～8 周规划。估算不包含大规模 Workflow 重构和完整插件化。

## 13. 兼容与灰度策略

P0 涉及模型加载和请求主路径，不能一次性切换。建议使用 observe → enforce 的灰度方式。

### 13.1 资源账本灰度

第一阶段 ResourceManager 只做 shadow reservation：

- 保留现有加载行为。
- 同时计算新 DeviceLedger 是否会接受请求。
- 记录 `legacy_allowed/new_rejected` 和 `legacy_rejected/new_allowed` 差异。
- 不因为 shadow 结果改变生产流量。

完成真机校准后再切换为 enforce。enforce 上线初期保留快速回退配置，但回退只能切换准入决策，不能绕过已建立 reservation 的 commit/release 清理。

### 13.2 Admission 灰度

- 先统计每种服务当前真实并发数、排队长度和等待时间。
- 再以较大 queue limit 开启 enforce，确认没有非预期拒绝。
- 按压测结果逐步收紧，并观察 429、P99 queue time、HTTP worker 使用量和内存。
- 客户端 SDK/API 文档需要明确 429/503 的重试条件，避免立即重试形成重试风暴。

### 13.3 Supervisor 灰度

- 先让 Supervisor 只观测进程退出并与现有 ProcessManager 结果对比。
- 再启用 Registry 自动纠正，但暂不自动重启。
- ready handshake 稳定后，停止接受未携带 worker id/generation/token 的旧状态回调。
- 最后启用 graceful shutdown 和可选 restart policy。

### 13.4 API 兼容

- 保持现有 OpenAI-compatible 路由和成功响应结构。
- `/v1/models` 保留现有状态字段，新内部状态作为扩展字段输出。
- 新的 429、499、503、504 语义需要在错误响应中保持统一 `error.type/code/message`。
- 新增 `/live`、`/ready` 和 `/metrics`，现有 `/health` 在过渡期继续保留。

## 14. P0 完成门槛

以下条件全部满足后，P0 才算完成：

- Release 主程序和全部单测通过。
- 新增控制面关键类均进入自动化测试 target。
- 并发加载不存在重复本地 Service、重复 Worker 或设备超卖。
- 所有模型队列有确定上限，取消和 deadline 可观测且不会继续执行排队任务。
- Worker 崩溃后在配置 SLA 内离开 READY，路由不再转发到失效端口。
- load/unload 500 次无 reservation、显存、线程、进程 handle 和临时文件持续增长。
- TSAN/ASAN/UBSAN 控制面测试通过。
- 24 小时真机长稳无死锁、僵尸进程、无界队列和持续内存增长。
- `/live`、`/ready` 和关键 P0 metrics 能反映真实状态。

## 15. P0 之后立即推进的工作

P0 完成后进入 P1：

- 将主进程本地模型正式建模为 `EmbeddedWorkerTransport`。
- 把 Workflow 拆成 Router、ModelController、ResourceScheduler、WorkerSupervisor 和 BackendRegistry。
- 让新增 backend 通过 descriptor/adapter 注册，减少 Factory、Manager、Workflow 和 route 中的类型 switch。
- 补充完整 tracing、租户配额、优先级、公平调度和性能回归平台。

以下次一级问题可在 P0/P1 主路径稳定后治理：

- 统一旧 C SDK 与 hlie-app 的两套 LLM serving 实现。
- CMake target 化、移除全局 include/definition、停止 Linux `-w`、减少 GLOB。
- 后端插件化和依赖版本清单。
- 管理面与数据面端口、安全策略、TLS 和 CORS 收口。
