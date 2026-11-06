# P1 实施进度

更新时间：2026-08-19

## 当前状态

- P1-0、P1-1、P1-2、P1-3、P1-4 已通过各自退出审计，状态为 **Completed**。
- P1-5 已通过 Windows、Linux 与真机退出审计，状态为 **Completed**：Controller 现在拥有 Registry、runtime event、load/unload/eviction transaction、confirmed-stop/Orphaned 和 `ensure_ready()` coalescing。
- P1-6a 至 P1-6f 已通过 Windows/Linux、sanitizer、真实流式断连、指标、P1-5/P1-6 性能对比和 exact-source 2 小时 mixed-workload soak，状态为 **Completed**。
- P0 安全基线已提交为 `2622af08`（`feat(hicpp): checkpoint P0 control-plane hardening`）。
- P1 没有改写模型加载协议、TCIM SDK 或 token/decode loop；P1-6 只在 app 请求边界统一数据面 session/endpoint，Embedded 仍直接调用原 handler。

## P1-0 第一批落地

- 新增真实生产级 `hlie-core` static target，使用显式 source list。
- 完整生产构建使用 `hlie-app -> hlie-core -> hlie-base`，同时保留 `hlie-app -> hlie-base` 直接依赖。
- `HLIE_TESTS_ONLY` 也编译同一个 `hlie-core` target；测试不再手抄 reservation ledger 的生产实现。
- 第一批进入 core 的构件：
  - `HLIEModelLaunchTicket`；
  - `HLIEAdmissionController`；
  - `HLIEServiceSlot`；
  - `HLIEDeviceReservationLedger` 及唯一 `.cpp` 实现。
- app 内部调用方已经改用 `hlie_core/...` include；原 `hlie_app/runtime/...` 路径保留 forwarding header，给迁移期兼容使用。
- 测试专用 target 从误导性的 `hlie-control-plane-test-core` 改名为 `hlie-control-plane-test-support`。它只承载尚未满足 core 边界的过渡源码，不再冒充生产核心。

## P1-0 第二批落地：中性运行态读模型

- 新增不依赖 app/SDK 的 `HLIEModelIdentity`、`HLIEModelLifecycleState` 和 `HLIEModelRuntimeSnapshot`。
- 生命周期显式表达 `Unloaded/Reserving/Starting/Ready/Sleeping/Draining/Stopping/Failed/Orphaned`；`Sleeping` 暂时作为旧接口的只读兼容投影。
- 现有 `HLIEModelRegistry` 仍是唯一存储和写入者，仅新增到 core snapshot 的单向 mapper，没有建立影子状态或双写路径。
- health、ready、metrics 和过期 reservation owner 判定已改为消费中性 snapshot，不再读取 app 私有的 `server_model_status`。
- snapshot 不暴露 launch token；generation、failure、request count 和 residency 属性保留为只读控制面事实。

## P1-0 第三批落地：驻留策略进入 core

- `HLIEModelResidencyManager` 的生产实现已迁入 `hlie-core`，app 原 include 路径仅保留 forwarding header。
- LRU/TTL、pinned/default/active-request 保护和候选排序现在只消费 `HLIEModelRuntimeSnapshot`，不再依赖 Registry、`server_model_meta` 或具体 Service。
- 普通构建与 `HLIE_TESTS_ONLY` 都链接同一个 residency 实现；测试不再手工编译 app 版本。
- core 源码审计未发现对 `hlie-app`、HTTP、TCIM、GGML 或具体 server/model 实现的反向 include。

## P1-0 第四批落地：运行态存储进入 core

- 新增 `HLIEModelRuntimeRegistry`，由 core 统一拥有 lifecycle、generation、launch ticket、port、failure、request lease 和 residency 运行态事实。
- launch token 由注入的 factory 生成，只通过 ticket API 暴露；普通 snapshot 不包含 token。恢复 `Ready` 记录时若无法获得有效 token，会 fail-closed 为 `Failed`。
- `Unloaded/Failed` 会清除旧 token；stale generation 的 request end、READY、failure 和 unload 更新继续被拒绝。
- app `HLIEModelRegistry` 现在是兼容 façade：catalog map 只保存 `HLIEModelInfo` 与 API 展示元数据，旧 `server_model_meta.status/port/exit_code` 在读取时由 core snapshot 投影，不再作为第二份运行态存储。
- 删除了无调用方且锁释放后返回可变引用的 `get_instance()`；兼容调用只允许获取快照。
- token 生成策略由 app composition 边界注入，core 不依赖 UUID、JSON、logger、文件系统或模型 SDK。
- 新增 core-only Registry 单测，覆盖状态迁移、alias 歧义消解、generation fence、stale lease、Ready 恢复和 token 失败关闭。
- 新增 CTest 架构门禁 `hlie.architecture.core-dependencies`，自动拒绝 core 对 app、server、TCIM、GGML、HTTP/uWebSockets 和 `HLIEModelInfo` 的 include。

## P1-1 第一批落地：领域类型与 BackendRegistry

- `HLIEModelCapabilities`、`HLIEModelRole` 和 `HLIERuntimeKind` 已成为 core domain 类型，`HLIEModelInfo` 只消费它们，不再自行定义一份 app 私有类型。
- 新增 SDK/文件系统无关的 `HLIEModelSpec`；`HLIEModelInfo::to_model_spec()` 是 app catalog 到 core policy 的单向映射，Factory 和 BackendRegistry 不再接收 app 私有模型对象。
- 新增 core-owned `HLIEBackendRegistry` 和 `HLIEBackendDescriptor`，显式注册当前产品真实存在的 LLM、SD、Whisper、CosyVoice、Qwen3-TTS 与 Document companion 六类 backend。
- descriptor 当前集中 stable backend id、contract version、legacy alias/name marker、匹配优先级、默认能力、embedded/process 支持、document companion 声明和纯内存估算策略。
- 匹配优先级通过 golden test 固化，保持旧行为中的 `Qwen3-TTS > Whisper > CosyVoice > generic LLM/SD` 规则；`tts` 继续是 CosyVoice 的兼容 alias。
- `HLIEModelServiceFactory` 已改为委托 BackendRegistry 做分类和 model type 校验，并通过 app-owned BackendAdapter registry 构造具体 Service；TCIM/HAL 类型没有进入 descriptor。
- 内存估算已拆成两层：app 观察经验配置、GGUF 和 companion 文件，core 的 descriptor 只消费 `HLIEModelMemoryObservation`，以纯函数选择 empirical/configured/artifact/heuristic 并返回来源与总量。
- `HLIEModelInfo::get_estimated_mem()` 生产路径已切换到 descriptor 策略；旧 qwen2.5/qwen3/gte/bge/whisper fallback 数值保持不变，负值被钳制，总量加法做饱和保护。
- 新增 app-owned `HLIEModelMemoryEstimator` 作为默认 BackendRegistry 的组合 façade；`HLIEModelInfo` 与 Workflow 裸路径现在都通过它调用同一 descriptor policy，Workflow 不再串联自己的 empirical/GGUF/legacy fallback。
- 纯规则型 `HLIEModelFamilyRegistry` 及 projector pairing 实现已迁入 core；app 原 header 只保留 forwarding include，test support 不再手工编译它。
- BackendRegistry 明确不是多 SDK/provider 或动态插件系统；P1 仍只面向单一 HOUMO TCIM SDK。

## P1-1 第二批落地：中性控制命令与结果

- 新增 `HLIEModelSelector`、`HLIEPlacementIntent`、`HLIEModelLoadCommand`、`HLIEModelUnloadCommand`、`HLIEDeadline` 和 `HLIEModelOperationReceipt`。
- placement intent 只表达调用方约束，不携带 HAL snapshot、TCIM handle 或已提交 reservation；设备观察和事务式申请仍属于后续 ResourceScheduler。
- 新增 `HLIEControlErrorCode`、`HLIEControlErrorStage` 与 `HLIEControlResult<T>`；成功值和 typed error 通过 variant 强制互斥，不再依赖空指针、魔法整数或 HTTP JSON 表达 core 失败。
- error contract 保留 app native code 的映射槽位，但 core 不理解 HTTP status、OpenAI error body 或 TCIM 错误枚举。
- 契约测试覆盖成功/失败互斥、模型/设备/期限校验和 move-only payload，为后续 Controller/runtime handle 留出所有权语义。

## P1-1 第三批落地：placement contract 与 BackendAdapter

- `HLIEPlacementIntent` 已从 command 文件移入独立 resource contract；descriptor 新增 `HLIEPlacementConstraints` 和纯 `plan_placement()`。
- placement 只校验设备数量、显式/自动选择能力和调用方约束，返回的 `HLIEPlacementPlan` 明确不是 reservation grant；硬件可用性仍由 P1-3 ResourceScheduler 观察和决定。
- 新增 app-owned `IHLIEBackendAdapter` 与 `HLIEBackendAdapterRegistry`，使用 stable backend id 将 core descriptor 连接到当前 HOUMO 具体 Service factory。
- Factory 中 LLM/SD/Whisper/CosyVoice/Qwen3-TTS/Document 的构造 switch 已删除；默认 adapter 表会 fail-fast 校验每个 descriptor 都有对应 app adapter。
- `HLIEModelServiceManager::load_by_model_info()` 成为迁移期统一加载 façade；Workflow 的 Worker manifest 和 default model 两处初始化 switch 已删除。
- LLM 参数解析、默认 mmproj、Document companion rollback、ServiceSlot 和 TCIM `init/run/stop` 顺序保持原样；Runtime abstraction 不在 P1-1 提前实现，仍留给 P1-4。

## P1-2：独立 WorkerSupervisor

- 新增 app-owned `HLIEWorkerSupervisor`，独立拥有 supervisor thread、poll/READY/graceful-shutdown 配置、逻辑 port lease、Worker spawn、READY probe、expected/unexpected exit 分类、OS exit confirmation、manifest cleanup 与 Worker 指标。
- 新增 core 中性的 `HLIERuntimeEvent` 与 `IHLIERuntimeEventSink`；事件携带 runtime id、instance/port、launch ticket、exit code 和 confirmed/expected 标志，不携带 Registry、reservation、HTTP response 或 SDK 对象。
- Supervisor 的 event callback 一律在内部锁外调用；停止操作用 launch ticket 校验 PID/instance/generation，拒绝 stale runtime identity，避免 PID 复用或旧 generation 停掉新 Worker。
- `HLIEProcessManager` 已恢复为通用 OS process adapter，不再解析 `--launch_manifest` 或拥有 Worker manifest 生命周期；所有生产 Worker spawn/stop 均改经 Supervisor。
- P1-2 当时由 Workflow 作为迁移期 event sink；P1-5 已由 ModelController 完成替换，READY/exit/timeout 不再回到 Workflow。
- Worker 指标的唯一 owner 已迁入 Supervisor；`/health` 与 `/metrics` 只读取其 snapshot，不再读取 Workflow 私有原子计数器。
- 新增架构门禁 `hlie.architecture.worker-supervisor-dependencies`：Supervisor 实现不得依赖 Workflow、Registry、DeviceService、reservation 或具体 ModelService，public header 不得泄露 HTTP/JSON/ProcessManager 实现类型。
- Fake Worker 组件测试现在直接驱动真实 Supervisor，覆盖 READY + inference、stale stop 拒绝、graceful exit、READY 前/后 crash、manifest cleanup、shutdown deadline 强杀和 confirmed READY timeout。
- pending reservation 的周期对账没有塞进 Supervisor；P1-3 后由现有 residency worker 只提供兼容 tick，owner matching、过期 rollback、tracked binding 清理与指标均由 ResourceScheduler 完成。

## P1-3：ResourceScheduler 与 reservation 所有权

- 新增 app-owned `HLIEResourceScheduler`，统一拥有 placement grant、reservation state transition、Worker reservation binding map、expired-pending reconciliation、ledger/metrics snapshot 和 residency plan。
- 新增 core 中性的 `IHLIEDeviceReservationProvider` port；Scheduler 只依赖该 port，`HLIEDeviceService` 是当前 HOUMO/HAL adapter。Scheduler 不 include Workflow、Registry、ProcessManager、HTTP、具体 Service 或 TCIM/GGML 类型。
- 新增 move-only `HLIEPendingReservationHandle` 与 `HLIECommittedReservationHandle`：启动事务先取得 pending，bind 时绑定 generation ticket；动态 Worker 只有在 READY 后才能从 pending 转为 committed，stale ticket 不能 commit/release 新 generation。
- Workflow 的 `model_reservations_`、`reservation_mutex_` 及 bind/track/commit/release helper 已删除；所有生产 reservation 操作改经 Scheduler，架构门禁会拒绝这些 ownership 回流。
- `HLIEPlacementIntent` 已接入真实申请路径：Workflow 只解析 API 中的显式/自动设备意图，Scheduler 校验并生成中性 reservation request，provider 根据硬件观察产生 grant。
- LRU resource-pressure、`max_loaded_models`、idle-TTL 的候选计算已由 Scheduler 返回 `HLIEResidencyPlan`；旧 Workflow adapter 暂时只执行 plan 中的 unload，未提前引入 P1-5 Controller。
- P1-3 曾保留本地 Service 接收 raw reservation id 的显式转交桥；P1-4 已删除该桥。外部 grant 现在对 Service 是 borrowed，commit/release 的唯一 owner 始终是 Scheduler。
- Reconciler 的周期 timer 暂留 Workflow，但它只提交 generation-fenced live-owner tickets；过期判断、owner matching、rollback、binding cleanup 和资源指标均在 Scheduler 内完成。
- `/health`、`/metrics` 的 reservation 指标与 ledger snapshot 改从 Scheduler 读取，并新增 tracked pending/committed gauge。
- 新增 `hlie.architecture.resource-scheduler-dependencies` 门禁与 Scheduler 契约用例，覆盖 typed transition、首次 alias-to-canonical bind、stale generation/replacement、explicit rollback、外部 grant 所有权、expired owner protection、placement validation 和 residency plan。
- 修复 `ci/run_linux_sanitizers.sh` 首次运行无 CMake cache 时因裸 `return` 在 `set -e` 下提前退出的问题。

## P1-4：统一 ModelRuntime 控制面

- 新增 core-owned `IHLIEModelRuntime` 控制面契约与中性值对象，统一表达 runtime identity、capabilities、health、draining 和 deadline-aware confirmed stop。接口不包含 HTTP DTO、JSON、具体 Service、TCIM/GGML、`void*` 或 `std::any`。
- `HLIEEmbeddedModelRuntime` 以 Pimpl 包装 `HLIEModelServiceManager` 的本地 Service bundle；LLM 的 Document companion 与主模型共用一个 runtime 生命周期。
- `HLIEProcessModelRuntime` 包装 `HLIEWorkerSupervisor` 的 Worker identity，通过 launch ticket/generation 校验健康、drain 和 graceful/forced stop，避免 PID 复用或旧 generation 操作新 Worker。
- 新增线程安全的 `HLIERuntimeDirectory`，只保存 generation-fenced runtime handle；`HLIEModelRuntimeRegistry` 继续是 lifecycle source of truth，Directory 不形成第二份状态机。
- 默认启动、动态本地模型和动态 Worker 都在 Ready 发布前建立 Runtime handle；启动回滚、READY timeout、unexpected exit、手动 unload、LRU/TTL eviction、drain 和 shutdown 的主路径均通过 Runtime 控制面。
- Runtime stop 只有在确认底层 Service 已停止或 Worker 已退出后才允许 Scheduler release；未确认停止会恢复同一 handle 并中止 eviction，资源不被提前复用。
- 删除 `transfer_to_local_runtime()`。动态本地 Service 使用外部 reservation id 时只借用 grant，不 commit/release；Workflow 在 Runtime 发布成功后让 Scheduler track/commit，并在 confirmed stop 后 release。无外部 grant 的默认启动模型仍保留 Service 自申请/自释放兼容路径。
- `/health` 与 `/metrics` 新增 runtime kind/state 观测；Failed/Orphaned runtime 会让 health 降级，轮询值只做 snapshot，不制造重复生命周期日志。
- 新增 `hlie.architecture.model-runtime-dependencies` 门禁，约束 core runtime、两个 adapter、RuntimeDirectory 和 Workflow 集成边界；单测覆盖 identity、health/stop、generation replacement/stale rejection、detach_all，以及 8 线程并发 publish/find/detach。
- P1-4 明确只统一控制面。推理 handler 仍使用 typed Service getter；中性 `IHLIEInferenceEndpoint`、command 和 Dispatcher 迁移属于 P1-6，避免为了抽象而增加本地 HTTP hop 或把 transport DTO 泄露进 core。

## 本轮验证

- Windows Release 主程序 `houmo-llm-hi-cpp-win`：构建通过。
- Windows 完整 CTest：8/8 通过：
  - `hlie.architecture.core-dependencies`；
  - `hlie.architecture.worker-supervisor-dependencies`；
  - `hlie.architecture.resource-scheduler-dependencies`；
  - `hlie.architecture.model-runtime-dependencies`；
  - `hlie.unit`；
  - `hlie.stress.control-plane`；
  - `hlie.component.worker-supervisor`；
  - `hlie.component.mirage`。
- Windows `HLIE_TESTS_ONLY=ON`：`hlie-core`、test support、unit target 构建通过，CTest 5/5 通过（含三条架构门禁和真实 Scheduler 契约测试）。
- Linux P1-4 精确源码在隔离 worktree 中 clean build 通过；产品目标 `houmo-llm-hi-cpp-linux` 构建通过，完整 CTest 7/7 通过（四条 architecture gate、unit、stress、WorkerSupervisor component）。
- Linux 构建发现并修复 `hlie-core` 静态库进入 `libhlie-app.so` 时缺少 PIC 的目标属性问题；`hlie-core` 现显式启用 `POSITION_INDEPENDENT_CODE`。
- Linux 真机短时 golden smoke 在 `test-user@test-host.example.com`、隔离端口 `isolated-port` 通过：
  - 主进程模型加载、推理和 reservation commit；
  - 动态 Worker READY、代理推理、READY 后崩溃收敛；
  - generation 恢复、graceful unload 和额外 1-cycle lifecycle；
  - 最终 reservation ledger 回到基线，INFO/WARN/ERROR 日志契约通过。
- golden 使用 Qwen3-ASR 0.6B 主模型与 1.7B 动态 Worker，共享单卡；现有 `existing-service-port` 服务未停止或修改，验证产物保存在远端隔离 worktree 的 `artifacts/p1_1_linux_golden_retry`。
- P1-2 Linux clean build 通过；CTest 5/5 通过（含两条架构门禁与真实 Supervisor Fake Worker component）。
- P1-2 真机 golden 在隔离端口 `isolated-port` 通过：动态 Worker READY/代理推理、READY 后 crash 的 Registry/route/process/reservation 收敛、generation 恢复、graceful unload、额外 lifecycle cycle 与最终 ledger 回零全部 PASS；产物位于 `artifacts/p1_2_linux_golden_retry`。
- 真机首次 P1-2 golden 发现 WebService 在 Workflow 启动后重复 `set_server_port()` 会替换并停止已运行 Supervisor；现已改为相同端口保持实例，真实端口变化时保留 running 状态完成重建/重启。修复前 crash 无监督，修复后约 1 秒完成全链路收敛。
- P1-3 Linux 产品目标 `houmo-llm-hi-cpp-linux` 构建通过，CTest 6/6；Linux `HLIE_TESTS_ONLY=ON` CTest 5/5。默认 `all` 仍会顺带编译一个与当前 vendored llama headers API 不匹配的独立旧 `houmo-llm-sdk` 目标，发布/CI 产品目标不依赖它，本轮未把该无关兼容升级混入 P1-3。
- P1-3 真机 golden 最终精确源码在隔离端口 `isolated-port` 通过：主进程 load/infer/commit、动态 Worker READY/infer、READY 后 crash 收敛、generation 恢复、graceful unload、额外 lifecycle cycle、ledger baseline 和日志契约全部 PASS；产物位于 `artifacts/p1_3_linux_golden_exact`，existing-service-port 未停止或修改。
- P1-4 真机 golden 在隔离端口 `isolated-port` 通过：Embedded/Process Runtime 发布、主进程 load/infer、Worker READY/infer、READY 后 crash 收敛、generation 恢复、graceful unload、额外 lifecycle cycle、最终 ledger baseline 和 INFO/WARN/ERROR 日志契约全部 PASS。RSS `455888896 -> 455970816`、private bytes `431874048 -> 431955968`、handles `18 -> 18`、threads `32 -> 32`；产物位于 `artifacts/p1_4_linux_golden/summary.json`，existing-service-port 未停止或修改。
- P1-4 精确源码 Linux ASAN+UBSAN 3/3、TSAN 6/6 通过，没有 sanitizer 或数据竞争报告。
- 新增 Registry 中性 snapshot 状态映射、基于 snapshot 的 residency 策略、ModelSpec 投影、BackendDescriptor 内存/placement 策略、typed control contract 和 app adapter registry 测试。
- `git diff --check` 通过；core 禁止依赖和旧源码 target 引用审计通过。
- 构建时发现 existing-service-port 调试服务占用 `hlie-app.dll`；按既有授权只停止该 listener 的进程后重试，链接通过。

## P1-5：ModelController 唯一生命周期入口

- 新增 app-owned `HLIEModelController`，私有持有 `HLIEModelRegistry`；Workflow 不再保存 Registry 成员，也不能直接调用 Registry API。
- Supervisor 的两个创建/重建路径都把 Controller 注册为 `IHLIERuntimeEventSink`；Workflow 不再实现 runtime event callback。
- Controller 已接管 generation-fenced READY、重复/旧 READY、unexpected exit、READY timeout、启动回滚、Runtime confirmed stop、全量 drain/shutdown，以及对应的 Scheduler release/retain 决策。
- 新增 `Orphaned` 写迁移：未确认 Worker exit、READY-timeout cleanup 或 Runtime stop 时保留 launch ticket、port、Runtime handle 与 reservation，禁止新 generation 和 eviction；只有 confirmed stop 才允许 release。
- 正常 Runtime eviction 已成为 Controller 原子应用操作：Controller claim eviction、按 ticket detach/stop、confirmed 后 release 并发布 Unloaded；Runtime handle 缺失时 Controller 会撤销 eviction claim，Workflow 不再扫描裸 Service/Process 执行第二套 fallback。
- 新增 Controller-owned `ensure_ready()`/`wait_until_ready()`：并发自动加载只有一个调用方获得 launch ticket，其余调用方等待同一 generation；Workflow 的 `slave_cv_` 和旧 `wait_for_model_load()` 已退出生产路径。
- LLM auto-load 与 speech loading wait 已接入 Controller coalescing；deadline、draining/client cancellation、Missing/Failed/Unloaded/Orphaned/Unavailable 均有明确结果，不再轮询 app 私有 status。
- 新增中性的 `IHLIEModelRuntimeFactory` create port 与当前唯一的 `HLIEDefaultModelRuntimeFactory`：Controller 只提交 Embedded/Process create request 并发布/清理 `IHLIEModelRuntime`；具体 ServiceManager/Supervisor 和 adapter 构造只存在于默认 factory 实现。它不是多 SDK/provider 或插件体系。
- 新增 app-owned `IHLIEModelBackendLauncher` port 与当前唯一的 `HLIEDefaultModelBackendLauncher`。该 port 只执行“启动本地 Service 或 Worker”并返回中性 artifact；具体 ServiceManager、DeviceService、manifest、端口租约和 Supervisor 只存在于默认实现，不进入 Controller/core。
- `HLIEModelController::load_model()` 已把 claim、reservation bind、backend launch、RuntimeFactory create、Runtime publish、track/commit、Ready/await-READY 和 rollback 编成一个 generation-fenced transaction。
- LLM、SD、Whisper、CosyVoice、Qwen3-TTS 五条动态启动路径均改为调用 `load_model()`；Workflow 的三个 `start_*` helper、直接 `ServiceManager::start_*`、`reserve_port/spawn`、`begin_load/adopt_*` 已删除。
- backend 已启动但 adapter 创建/发布失败时，Controller 只有在 Launcher/Runtime 确认清理后才释放 reservation；未确认清理会保留 generation-bound pending reservation 并写入 `Orphaned`。
- 动态 unload、API stop、resource-pressure、loaded-limit 与 TTL eviction 均直接调用 Controller；Workflow 的 `evict_model` 与裸 Service/Process 恢复分支已删除。
- 新增 `hlie.architecture.model-controller-dependencies`：禁止 Controller 依赖 Workflow、具体 ServiceManager、ProcessManager、DeviceService、HTTP 或 SDK，并拒绝 Workflow 重新出现 `model_registry_`/runtime event sink。
- 新增 Controller 组件级单测与 core Orphaned 状态测试，覆盖 READY commit、confirmed unexpected exit、unconfirmed ownership retention、资源不误释放、重载拒绝、confirmed eviction，以及 16 路 `ensure_ready()` 只有一个 launcher。
- Windows Release 主程序与 unit target 构建通过；加入 `hlie.architecture.model-backend-launcher-dependencies` 后完整 CTest 10/10 通过，其中六条 architecture gate、unit、stress、WorkerSupervisor component 与 Mirage 全绿。
- Linux 精确源码产品目标构建通过，完整可用 CTest 8/8、无 SDK clean build/CTest 7/7 通过。
- Linux 真机短时 golden 在隔离端口 `isolated-port` 通过：主进程 load/infer、Worker READY/infer、READY 后 crash 收敛、generation 恢复、graceful unload、额外 1-cycle lifecycle、ledger baseline 和 INFO/WARN/ERROR 日志契约全部 PASS；产物位于 `artifacts/p1_5_controller_golden`，existing-service-port/other-service-port 未停止或修改。
- P1-5 完整 load transaction 精确源码在 Linux 独立目录 `build-p1-5-load-transaction` 构建通过，CTest 9/9；新增 BackendLauncher gate 后六条 architecture test、unit、stress 和 WorkerSupervisor component 全绿。
- P1-5 最新控制面源码 Linux ASAN+UBSAN 3/3、TSAN 8/8 通过，没有 sanitizer、undefined behavior 或数据竞争报告。
- 最新真机 golden 在 `test-user@test-host.example.com` 隔离端口 `isolated-port` 通过：主进程本地模型 load/infer/commit、动态 Worker READY/proxy inference、READY 后 crash 收敛、generation 恢复、graceful unload、额外 1-cycle lifecycle、最终 ledger baseline 和 INFO/WARN/ERROR 日志契约全部 PASS。产物位于 `artifacts/p1_5_load_transaction_golden`，既有 `existing-service-port/other-service-port` 未停止或修改。

## P1-6：InferenceDispatcher 与 ApiRouter（Completed）

- 新增 app-owned `IHLIEInferenceEndpoint`、`HLIEInferenceSession` 和 `HLIEInferenceDispatcher`。Dispatcher 先从 Controller 获得 generation-fenced request ticket，再精确匹配 RuntimeDirectory handle 与 Endpoint identity；stale/missing 路径 fail-closed 并归还 lease。
- guarded response 固定上游 response/proxy 先析构、endpoint/Runtime 后析构、最后 `end_request()`；`lifetime_guard()` 支持 httplib chunked callback、Qwen3-TTS streaming 和 Process speech proxy 跨 handler 返回继续持有 session。
- 默认 resolver 已实现 Embedded/Process endpoint：Embedded 直接调用原 LLM、SD、Whisper、Document、Qwen3-TTS/CosyVoice handler，Process 沿用 `ServerHttpProxy` timeout/cancel/trailer；本地主进程没有增加 HTTP hop。
- steady-state endpoint 按 generation 强缓存；现有 reconciliation tick 清除 RuntimeDirectory 中已消失的 endpoint，避免每请求重新构造 endpoint/Service snapshot，也不让旧 generation 长期持有 Service。
- Embedded resolver 除校验 service kind 外，还校验 endpoint ticket 的 canonical model key 与具体 Service snapshot；同类 slot 与 runtime identity 不一致时 fail-closed，不能把请求 identity 直接套到另一模型的 Service 上。
- LLM chat/completions/responses/embedding/rerank/props/slots、SD image、Whisper/LLM transcription、PDF Document、Qwen debug、Whisper WebSocket 和 speech 已切到 Dispatcher；Workflow 不再直接调用具体 inference handler、不构造 Proxy，也不定义私有 ModelRequestLease。
- 新增 `HLIEInferenceHttpErrorMapper`，集中映射 typed dispatch status 到现有 HTTP/OpenAI error contract；新增 health/Prometheus dispatch active/rejected/failure/cache 指标。
- route table 已迁入 `HLIEApiRouter`；`HLIEWebService` 不再 include/持有 Workflow，只依赖 `IHLIEHttpRouteProvider`、`IHLIEWebServiceStateProvider` 和 `IHLIEAuthPolicy`。
- 新增第七条架构门禁 `hlie.architecture.inference-dispatcher-dependencies`，禁止 Dispatcher/Router/WebService 反向依赖 Workflow、ServiceManager/具体 Service、DeviceService 或 Proxy，并禁止 Workflow 重新获得直接推理/lease ownership。
- Windows Release 产品与 unit target 构建通过，完整 CTest 11/11；Linux Release 产品构建和 CTest 10/10、ASAN+UBSAN 3/3、TSAN 9/9 通过，没有 sanitizer、undefined behavior 或数据竞争报告。
- P1-6 定向测试 8 个用例、292 条断言通过：ApiRouter 精确锁定 30 组/48 个 public path、HTTP method 和全部 inference operation 映射；Dispatcher 覆盖 stream teardown、async lifetime guard、stale/unsupported/draining fail-close、generation cache/reconciliation。2000 次性能循环的断言已移出计时区，Windows cached request-boundary 平均开销为 3 us。
- 真机 `artifacts/p1_6_stream_cancel_golden_exact` 在 embedded/process 两条路径均验证：首个 SSE data event 后主动断连时 session 正在持有，随后 `active_sessions` 归零、普通推理恢复；`/health.inference_dispatch` 与 Prometheus 全部字段一致，之后 Worker crash/recovery、graceful unload、ledger 和日志契约全绿。
- P1-5/P1-6 使用相同模型、参数和真机顺序采集 1400 个 warm-up 后样本。P1-6 embedded 的 mean/p50/p99 变化为 `-0.423%/-0.259%/-0.621%`，Process 为 `-0.427%/-0.414%/+0.559%`，均满足 3% 门槛；原始样本位于 `p1_5_dispatch_performance_baseline*` 与 `p1_6_dispatch_performance_candidate*` artifacts。
- exact-source 2 小时 mixed-workload soak 在隔离端口 `18931` 完成 1975 个 Worker load/infer/unload cycle，elapsed 7202 秒，最终 ledger 回到基线且 manifest 零泄漏。handles `20 -> 20`、threads `32 -> 33`、RSS `456540160 -> 461012992` bytes、private bytes `431964160 -> 434913280` bytes；最后 20% 从 cycle 1575/1600 到 1975 的 RSS/private 均只增长 12288 bytes，无线性增长。最终 `graceful_shutdowns=1976`、`forced_shutdowns=0`，INFO/WARN/ERROR 日志契约通过；产物位于 `artifacts/p1_6_mixed_workload_soak_2h_exact`。

## P1-7：Discovery、Catalog、Reconciler 与 Composition（In Progress）

本轮新增 `HLIEWorkerControlService`：历史 Worker status 发布、typed
status application、processor snapshot、UNLOADED 清理和 shutdown ticket
幂等均已退出 Workflow；HTTP/JSON 仍只存在于 adapter。新增
`hlie.architecture.worker-control-service-dependencies` 门禁以及服务级
并发/幂等测试，详细契约见 `docs/P1-7_WORKER_CONTROL_SERVICE.md`。

默认 embedded startup 现已由 `HLIEEmbeddedStartupService` 负责 Runtime
发布、设备快照、startup rollback 和 ready callback；
`HLIEDefaultEmbeddedStartupTargetProvider` 负责 ServiceManager 槽位到 typed
target 的兼容映射。Workflow 保留 primary/admission/secondary 的原顺序，
详细边界见 `docs/P1-7_EMBEDDED_STARTUP_SERVICE.md`。

Document request orchestration 现已由 `HLIEDocumentApplicationService`
负责：PDF sniff、preprocess/OCR operation 选择、generation-fenced dispatch
委托和 LLM-owned document companion stop callback 均已退出 Workflow。
PDFium/layout/OCR/merge 底层实现保持不变，本地路径没有增加 HTTP hop 或
payload copy。正常 PDF 路由观测使用 DEBUG，详细边界见
`docs/P1-7_DOCUMENT_APPLICATION_SERVICE.md`。

ASR request/realtime orchestration 现已由 `HLIEAsrApplicationService` 负责：
direct Whisper/LLM selection、Qwen3-ASR + ForceAligner word timestamp
pipeline、PCM-to-WAV、stream decoder、admission lifetime 和 realtime listener
start/stop 均已退出 Workflow。uWebSockets 只存在于默认 transport adapter，
详细契约见 `docs/P1-7_ASR_APPLICATION_SERVICE.md`。

Speech request orchestration 现已由 `HLIESpeechApplicationService` 负责：
CosyVoice/Qwen3-TTS selection、loading wait、generation-bound target、Qwen
admission guard、backend header 和 endpoint dispatch 均已退出 Workflow。
stream lifetime 继续由 Dispatcher guard 保证，详细契约见
`docs/P1-7_SPEECH_APPLICATION_SERVICE.md`。

默认模型初始化已由 `HLIEDefaultModelInitializationService` 接管：configured
primary/SD3 selection、default mmproj 校验、LLM 参数准备、Service slot init 和
zero-load/failed typed outcome 均已退出 Workflow。后续 `run()` 与 generation
Runtime publication 继续由 `HLIEEmbeddedStartupService` 负责，详细契约见
`docs/P1-7_DEFAULT_MODEL_INITIALIZATION_SERVICE.md`。

`/v1/models` compatibility view 已由 `HLIEModelListApplicationService` 接管：
public-name de-dup、loaded 优先、本地主进程/Worker instance、alias key、device
fallback 与 `loaded_info` extensions 均已退出 Workflow。它只消费 composition
提供的中性 snapshot，不直接依赖 Controller/Manager/ProcessManager，详细契约见
`docs/P1-7_MODEL_LIST_APPLICATION_SERVICE.md`。

Worker legacy HTTP façade 已由 `HLIEWorkerControlApplicationService` 接管：
GET processor DTO、POST status conversion/result mapping 和 internal shutdown
ticket response 均已退出 Workflow；typed fencing/lifecycle 继续由
`HLIEWorkerControlService` 负责。逐请求 receipt 为 DEBUG，成功状态迁移/首次
shutdown 为 INFO，lookup/invalid ticket 为 WARN，且不记录 launch token。

- 新增 app-owned `HLIEControlPlaneReconciler`，独立拥有可中断 timer 和 stop/join。每秒 reconciliation 汇总外部 Process owner 与 Controller generation snapshot，调用 Scheduler 回收 expired pending、调用 Dispatcher 清理 stale endpoint；TTL 到期后只请求 Scheduler plan，并通过 Controller 执行 eviction。
- Reconciler 内不访问 Singleton、ProcessManager、DeviceService、Registry、HTTP 或具体 Service；配置和外部 live owner 由 composition callback 注入。正常 TTL 驱逐使用 INFO，未完成驱逐使用 WARN，循环异常使用 ERROR，无变化 tick 不打印 INFO。
- 新增 app-owned `HLIEModelDiscoveryService`，接管 discovery thread、CV、request/completed generation、cooldown、目录 snapshot 与稳定窗口。目录解析使用 deferred update + 二次 snapshot 校验，下载期间发生变化时不发布；相同 snapshot 不重复触发 GGUF/model parse。
- DiscoveryService 不依赖 Workflow、Controller、Registry、ConfigService、GGUF 或 HTTP。Workflow 只注入 options provider 与“准备 catalog update”回调；旧 `HLIEModelDiscoverySync` 和 Workflow discovery 状态已删除。
- 新增 core read-only `IHLIEModelCatalog`，只暴露 `HLIEModelSpec resolve/snapshots`；新增 app `HLIEModelCatalog` 保存 `HLIEModelInfo` 与兼容 meta 的静态 payload。Registry 不再持有 `catalog_entries_`，只组合 Catalog snapshot 与 core RuntimeRegistry snapshot。
- 新增 `HLIEApplicationCompositionRoot`，集中构造当前唯一 HOUMO TCIM 图：DeviceService、Catalog、Scheduler、RuntimeFactory、BackendLauncher、Controller、EndpointResolver、Dispatcher、Reconciler、Discovery 和 ModelLoadService。root 不依赖 Workflow、Singleton、ConfigService 或 ProcessManager；它不是多 SDK/plugin 层。
- 新增 app-owned `HLIEModelLoadService`：接收不含 HTTP/JSON 的 typed request，统一完成 model alias/path identity、memory estimate、loaded-limit/LRU plan、placement、pending reservation 与 `Controller::load_model()` 提交。服务不写 Registry、不启动具体 Service/Worker、不 commit/release reservation；Controller 仍唯一拥有 generation-fenced lifecycle transaction。
- 显式 `/models/load` 现在只负责 DTO 解析和 `HLIEControlHttpErrorMapper` 映射；auto-load 直接调用同一个 typed service，已删除构造 `server_http_req`、序列化 JSON 后回调 `start_service()` 的内部 HTTP 模拟路径。
- 新增 app-owned `HLIEModelReadinessService`：独立拥有 auto-load task group、preclaim rollback、Controller `ensure_ready()/wait_until_ready()` coalescing、client/drain cancellation 和 stop/join。Workflow 不再拥有 auto-load thread，LLM auto-load 与 speech loading wait 统一消费 typed readiness status。
- Workflow 已删除未使用的 public model-service/API-key getter，线程数契约更名为 `calculate_http_thread_count()`；public API 不再暴露 model-specific getter。Reconciler/Discovery metrics 已进入 `/health` 与 Prometheus。
- 新增四条架构门禁：`control-plane-reconciler`、`model-discovery`、`model-catalog`、`application-composition`；旧 Runtime/BackendLauncher 门禁已改为验证 composition root ownership。
- 新增 8 个定向测试用例、76 条断言：Reconciler 3/22、Discovery 3/36、Catalog 2/18。Registry 17/172、Controller 12/108 回归通过；Windows Release 产品构建与完整 CTest 15/15 通过。
- Linux exact-source Release 产品构建与完整 CTest 14/14 通过；ASAN+UBSAN 3/3、TSAN 13/13 通过，没有 sanitizer、undefined behavior 或 data race 报告。
- TSAN 验证暴露了 Reconciler 停止路径的真实 condition-variable 丢唤醒窗口：stop predicate 可能在 worker 完成 predicate 检查后、进入 wait 前发布，导致停止等待完整 5 秒周期。现已在 `wait_mutex_` 下发布 `stop_requested_` 后再 notify；TSAN 下仍执行全部功能断言，仅关闭不适用于插桩构建的 `<100 us` 微基准阈值，Release 性能门禁保持不变。
- 真机 180 秒 mixed-workload 在 `test-user@test-host.example.com` 隔离端口 `isolated-port` 通过：完成 59 个 Worker load/infer/unload cycle，主进程 load/infer/commit、Worker READY/proxy inference、READY 后 crash 收敛、generation 恢复与 graceful unload 全绿；最终 ledger 回基线、manifest 零泄漏、`graceful_shutdowns=60`、`forced_shutdowns=0`。handles `20 -> 20`、threads `32 -> 32`、RSS `433250304 -> 436580352` bytes、private bytes `408055808 -> 410087424` bytes，尾段无持续增长；INFO/WARN/ERROR 日志契约通过。产物位于 `artifacts/p1_7_control_services_short_soak_retry2`，既有 `existing-service-port/other-service-port` 未停止或修改。
- ModelLoadService 定向测试 4 个用例、28 条断言通过，覆盖 embedded load transaction、显式设备数量拒绝、replica-mode 拒绝与 typed control error 映射；Windows Release 产品构建与完整 CTest 16/16、Linux exact-source clean build 与 CTest 15/15 通过。
- 新 typed load 路径在真机隔离端口 `isolated-port` 完成 20-cycle mixed-workload：主进程 load/infer/commit、Worker READY/proxy inference、READY 后 crash 收敛、generation 恢复、graceful unload、最终 ledger/manifest 和 INFO/WARN/ERROR 契约全绿，handles/threads 保持 `20/32`。产物位于 `artifacts/p1_7_model_load_service_smoke`。
- auto-load 在隔离端口 `isolated-port` 单独验证：catalog 建立并卸载后，普通 `/v1/chat/completions` 成功触发 typed load、完成真实推理并再次卸载；最终 loaded/runtime 为 0、`active_sessions=0`、tracked pending/committed 和设备 pending/committed reservation 全为 0。产物位于 `artifacts/p1_7_model_load_auto_load_smoke_retry3`。
- ModelReadinessService 新增 2 个并发/取消用例、20 条断言：12 个并发调用只触发一个 load function，全部等待同一 generation 进入 Ready；单个 waiter cancellation 不回滚当前 generation。Windows Release CTest 17/17、tests-only 15/15，Linux exact-source Release CTest 16/16 通过。
- Readiness/ModelLoad 最新源码通过 Linux ASAN+UBSAN 3/3 与 TSAN 15/15，无 sanitizer、undefined behavior 或 data race 报告。隔离端口 `isolated-port` 再次完成真实 auto-load/chat/unload，generation 递增到 2，最终 loaded/runtime/session/tracked reservation/设备 reservation 全为 0；产物位于 `artifacts/p1_7_model_readiness_auto_load_smoke`。
- 新增 `HLIEDocumentApplicationService` 与 document application-service 架构门禁；Workflow 不再直接 inspect PDF、选择 Document operation 或停止 document companion。定向测试 2 个用例、15 条断言通过；Windows Release 主程序构建和完整 CTest 20/20、`HLIE_TESTS_ONLY` CTest 18/18 通过。
- 新增 `HLIEAsrApplicationService`、`IHLIEAsrRealtimeTransport` 与 ASR application-service 架构门禁；Workflow 不再包含 transcription pipeline、realtime handler、Qwen stream decoder 或 WebSocket 类型。定向测试 3 个用例、15 条断言覆盖 direct Whisper、Qwen3-ASR + ForceAligner 和 realtime lifecycle；正常逐请求路由日志降为 DEBUG，失败日志不再输出 transcript/response body。Windows Release 主程序构建与完整 CTest 21/21、`HLIE_TESTS_ONLY` CTest 19/19 通过。
- 新增 `HLIESpeechApplicationService` 与 speech application-service 架构门禁；Workflow 不再选择 CosyVoice/Qwen3-TTS、不等待 speech readiness、不 acquire Speech session 或组合 Qwen admission guard。定向测试 4 个用例、17 条断言在完整与 SDK-free target 均通过；正常 speech route 日志由 INFO 降为 DEBUG。Windows Release 主程序与单测 target 构建通过，完整 CTest 22/22、`HLIE_TESTS_ONLY` CTest 20/20 通过。
- 新增 `HLIEDefaultModelInitializationService` 与架构门禁；默认 primary/SD3 选择、mmproj 检查、LLM 参数准备和 Service 初始化不再位于 Workflow。定向测试在完整与 SDK-free target 均通过 4 个用例、19 条断言；Windows Release 主程序和两个单测 target 构建通过，完整 CTest 23/23、`HLIE_TESTS_ONLY` CTest 21/21 通过。
- 新增 `HLIEModelListApplicationService` 与架构门禁；Workflow 不再直接组装 `/v1/models` JSON 或合并 Service/Process instance。定向测试在完整与 SDK-free target 均通过 2 个用例、16 条断言；Windows Release 主程序和两个单测 target 构建通过，完整 CTest 24/24、`HLIE_TESTS_ONLY` CTest 22/22 通过。
- 新增 `HLIEWorkerControlApplicationService` 与架构门禁；Workflow 不再直接解析/组装 worker status 与 shutdown JSON。定向测试在完整与 SDK-free target 均通过 3 个用例、32 条断言；Windows Release 主程序和两个单测 target 构建通过，完整 CTest 25/25、`HLIE_TESTS_ONLY` CTest 23/23 通过。
- 新增 `HLIEModelUnloadApplicationService` 与架构门禁；Workflow 不再解析 unload selector、映射 unload HTTP 错误或直接发起单模型 eviction。Controller 仍是 eviction/generation/Runtime/Registry/Scheduler 的唯一事务 owner。定向测试在完整与 SDK-free target 均通过 4 个用例、57 条断言；Windows Release 主程序和两个单测 target 构建通过，完整 CTest 26/26、`HLIE_TESTS_ONLY` CTest 24/24 通过。HTTP facade 成功映射记为 DEBUG，Controller 的真实 lifecycle transition 保持 INFO，busy/no-loaded 保持 WARN。
- 新增 `HLIEStopServiceApplicationService` 与架构门禁；Workflow 不再解析 `/hlie/v1/stop` DTO、不再执行本地批量 eviction claim/rollback，也不再直接编排 Controller/Supervisor Worker stop。legacy Worker 改用停止前模型快照完成 Registry/Scheduler 收敛；本地 eviction 失败只 abort 尚未执行的尾部 claim，不会重新开放当前 orphaned runtime。定向测试在完整与 SDK-free target 均通过 6 个用例、52 条断言；Windows Release 主程序和两个单测 target 构建通过，完整 CTest 27/27、`HLIE_TESTS_ONLY` CTest 25/25 通过。
- 新增 `HLIEObservabilityApplicationService` 与架构门禁；live/ready/health JSON 和 Prometheus rendering 已退出 Workflow。readiness 使用轻量 model summary，health/metrics 共用 composition 组装的中性 snapshot；不新增轮询日志。定向测试在完整与 SDK-free target 均通过 4 个用例、43 条断言，覆盖完整 health JSON 与 Prometheus exact golden；Windows Release 主程序和两个单测 target 构建通过，完整 CTest 28/28、`HLIE_TESTS_ONLY` CTest 26/26 通过。
- 新增 `HLIELlmApplicationService` 与架构门禁；model selection、discovery retry、POST-only auto-load DTO、readiness error mapping、document-aware dispatch、response-lifetime admission、`/props` 与 `/slots` 已退出 Workflow。ASR 通过 `handle_admitted()` 复用其跨转写/对齐持有的 permit，避免 admission limit 为 1 时重复占位。SDK-free 定向测试通过 7 个用例、92 条断言；Windows Release 产品主程序和两个单测 target 构建通过，完整 CTest 29/29、`HLIE_TESTS_ONLY` CTest 27/27 通过。详细类图、时序和兼容矩阵见 `docs/P1-7_LLM_APPLICATION_SERVICE.md`。
- 新增 `HLIEBootstrapApplicationService` 与架构门禁；主进程/Worker 分岔、launch ticket、一次性/legacy manifest、初始 catalog 注册、discovery baseline 顺序和 default initialization 结果映射已退出 Workflow。一次性 manifest 在 parse attempt 后无论成功失败都消费，legacy `--json_info` 保持不删除；必要 composition adapter 缺失时立即失败，Worker 路径仍先消费一次性 manifest；Runtime start/publication 仍属于后续 run 阶段。定向测试通过 7 个用例、59 条断言；Windows Release 产品和两个单测 target 构建通过，完整 CTest 30/30、`HLIE_TESTS_ONLY` CTest 28/28 通过。回归中暴露的旧 stress 固定 yield-count 启动窗口已改为 2 秒 wall-clock deadline，两套 target 各连续 5 次通过 `7628 assertions`。详细类图和时序见 `docs/P1-7_BOOTSTRAP_APPLICATION_SERVICE.md`。
- 新增 `HLIEApplicationLifecycleService` 与架构门禁；run/drain/stop 顺序、一次性启动、单调 draining、并发 stop 等待与同结果返回已退出 Workflow。新增 Running-to-Serving publication gate，将 WebService、balancer 和 AppController 的 serving 标志作为一个可与 stop 串行化的发布事务，避免并发停止暴露半启动 HTTP 图。EmbeddedStartupService 对“已启动但未发布 Runtime”的模型缺失、ticket 获取失败和 Runtime publication failure 增加即时 cleanup callback，减少 shutdown residual sweep 的正常命中范围；residual adapter 仍保留为未确认所有权的最终安全网。shutdown 固定先停 Discovery/Reconciler/Readiness 和 ASR realtime ingress，再停 Controller Runtime、Worker、Supervisor、残余未发布 Service slot 与 WorkerControl，避免 realtime ASR 与 Whisper/LLM Runtime 销毁并发。Lifecycle 启动失败会先清理部分对象图并阻止 AppController 发布 WebService；残余 native stop 与 Worker `UNLOADED` publication 分别进入 shutdown aggregate，任一未确认都不记录完整关闭。残余 slot 仍由 composition compatibility adapter 清理，不形成第二套 Runtime 状态机。EmbeddedStartup 与 Lifecycle 定向测试在完整与 SDK-free target 合计通过 19 个用例、92 条断言；Windows Release 产品和两个单测 target 构建通过，完整 CTest 31/31、`HLIE_TESTS_ONLY` CTest 29/29 通过。WorkerSupervisor graceful-timeout 用例在一次高负载全量运行中抖动，单独连续 3 次及最终全量复核均通过。详细类图、时序和失败矩阵见 `docs/P1-7_APPLICATION_LIFECYCLE_SERVICE.md`。

P1-7 已完成本轮功能与所有权收敛：Document、ASR、Speech、model list、Worker
control/application、default initialization、embedded startup、bootstrap 和
application lifecycle 均有明确 owner；单模型 unload、stop-service、observability
与 LLM compatibility routing 已退出 Workflow 的具体 Service 编排。Workflow 当前
约 1265 行，其中剩余内容是兼容入口、配置读取和阶段顺序桥接，不再拥有第二套
Registry/Runtime/Scheduler 状态。约 500 行是后续削薄目标，不作为本轮 P1 完成门槛。

## 尚未完成项

- `/v1/models` 的 `server_model_meta` 兼容投影已进入 ModelListApplicationService，静态 payload 位于显式 `HLIEModelCatalog`；core port 只暴露中性 `HLIEModelSpec`，不会把 `HLIEModelInfo`/HTTP DTO 迁入 core。
- HTTP DTO、模型参数解析和 GGUF/file 观察仍在 app；dispatch typed error 映射位于 `HLIEInferenceHttpErrorMapper`，动态加载 typed error 映射位于 `HLIEControlHttpErrorMapper`，其他控制面 API 映射不应进入 Controller 或 core。
- cancellation metrics 已改为只消费 `hlie-core` 的 `HLIECancellationObservation`；具体 queued-request control 通过 app 边界适配，不再由 metrics include model service。
- GGUF/file parsing 与 LLM 参数解析继续留在 app/base，不进入 core。
- `HLIEModelServiceFactory::default_registry()` 现在是默认 BackendRegistry 的唯一来源；MemoryEstimator 和 ModelLoadService 均从该 accessor 取得策略，composition root 显式传入同一来源。
- Document、ASR、Speech 与 LLM request orchestration 已分别进入对应 application service。实际 load 与 readiness coordination 已分别进入 ModelLoadService/ModelReadinessService，route table 与 backend 数据面 dispatch 已分别进入 ApiRouter/Dispatcher；Workflow 仍保留部分 legacy lifecycle façade，后续只应继续缩小 callback，不得恢复具体 Service 分支。
- 默认模型启动已明确分为两阶段：DefaultModelInitializationService 负责选择/Service init，EmbeddedStartupService 负责 `run()`/Runtime publication。Workflow 只读取兼容配置、映射 typed result 和维持阶段顺序。
- `hlie-control-plane-test-support` 仍手工编译上述过渡源码；只有这些职责完成解污染并迁入真实 target 后才能删除该 source list。
- Workflow 只把 transport-specific devices 字符串解析为 typed device ids；ModelLoadService 向 Scheduler 请求初始 pending grant，Controller 负责 bind/track/commit/rollback/confirmed-stop release。live-owner snapshot、expired reconciliation、endpoint cleanup 和 TTL plan/eviction trigger 已迁入 Reconciler。
- 具体 Service snapshot 与 Worker proxy 只存在于默认 endpoint resolver；Dispatcher 不应反向进入 P1-4 的 core Runtime 控制接口。

## P1 收尾与下一阶段

P1-1 至 P1-4 退出条件已经满足。P1-5 代码退出条件也已满足：Controller 是 Registry 唯一 owner/runtime event sink 和动态 load/unload transaction owner；Workflow 不再直接启动 backend、发布 Runtime 或执行裸生命周期 fallback，六条架构门禁阻止职责回流。

P1-6 和 P1-7 已完成。P1-7 的 Discovery/Reconciler/Catalog/composition root、ModelLoadService、ModelReadinessService、WorkerControlService/ApplicationService、DefaultModelInitializationService、BootstrapApplicationService、ApplicationLifecycleService、EmbeddedStartupService、DocumentApplicationService、AsrApplicationService、SpeechApplicationService、ModelListApplicationService、ModelUnloadApplicationService、StopServiceApplicationService、ObservabilityApplicationService 与 LlmApplicationService 已落地并由架构门禁保护。本轮收尾还完成了 Workflow 的 Controller/Runtime snapshot 读路径、共享 BackendRegistry 来源和中性 cancellation contract。

下一阶段再处理残余 compatibility adapter、Workflow 进一步削薄以及 `hlie-control-plane-test-support` 的 source list 治理；这些工作不得反向改写 Controller、Dispatcher、Supervisor 已建立的所有权契约。

详细架构和阶段退出条件见 [P1_WORKFLOW_ARCHITECTURE_REFACTORING.md](P1_WORKFLOW_ARCHITECTURE_REFACTORING.md)。
