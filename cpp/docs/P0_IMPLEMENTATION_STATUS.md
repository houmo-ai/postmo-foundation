# P0 实施进度

## 2026-08-17 Code Review Hardening

- Runtime request leases now retain the Registry generation/token. A stale
  response from an exited Worker cannot decrement the active request count of
  a reloaded generation. The unsafe model-key-only request release API has
  been removed. Upstream responses are destroyed before their model lease or
  admission permit is returned.
- Shutdown enters `DRAINING` before closing the HTTP server. Existing
  auto-load tasks are joined before the final Worker snapshot, and Worker
  reservations remain owned unless process exit is confirmed.
- Linux Worker creation uses `posix_spawnp`; the multi-threaded post-fork C++
  allocation/logging window has been removed. `waitpid(EINTR)` is retried and
  process-query failures no longer imply exit.
- Worker readiness is also polled by the supervisor, so a saturated inference
  HTTP pool cannot prevent a loading Worker from becoming routable. Admission
  reserves control-plane capacity through
  `admission.control_plane_reserved_threads`. READY publication is fail-closed:
  a rejected Worker callback enters draining, while a supervisor-side
  reservation reconciliation failure terminates and reaps the Worker before
  releasing ownership.
- Aggregate HAL growth is no longer attributed to framework reservations.
  `committed_unobserved_mb` remains conservative until reservation release;
  this can temporarily reduce utilization but prevents external/root service
  memory from opening an oversubscription window.
- LLM admission now covers the word-timestamp ASR pipeline and realtime ASR
  LLM path. Stream payload logging moved to byte-count-only `TRACE`, and
  `launch_token`/generic token fields are redacted.
- Windows Release validation after the review: main target built; unit tests
  passed with 144 cases / 8554 assertions; Worker supervisor component passed
  with 4 cases / 36 assertions; CTest passed 4/4 including Mirage.
- Linux x86_64 Release validation after the review: main target, unit tests and
  Worker supervisor component built; CTest passed 3/3. ASAN+UBSAN passed 3/3
  and TSAN passed 2/2 without sanitizer findings.

更新时间：2026-08-17

## 已落地

- Service slot 的并发 load ticket、快照、detach 和后台任务生命周期收敛。
- Device reservation ledger：显存 reserve/commit/rollback/release、保守 HAL accounting，以及本地/父进程加载路径接入。聚合 HAL 增长不再用于抵扣框架 reservation。
- Pending reservation TTL/orphan reconciler：reservation 记录创建与过期时间，监督器只回收已超时且没有当前 Registry loading/loaded owner、也没有存活 Worker 的 pending；活跃 owner 不会仅凭时间被释放。回收次数和“已过期但仍受 owner 保护”的 gauge 进入 `/health` 和 Prometheus。
- GGUF 内存估算纳入 KV cache、context、parallel slots、workspace 和 projector 附加开销，并保持不低于历史估算。
- Whisper、CosyVoice、Qwen3-TTS、SD 和 LLM 的加载失败/停止按 reservation id 回滚或释放。
- LLM/Qwen3-TTS front-door admission、HTTP 429、设备账本健康信息和有界队列。
- Whisper、CosyVoice 和 SD 的排队项携带 request cancellation/deadline；同步等待会提前返回，worker 在进入 backend 前再次校验，服务停止会完成所有残留 promise。排队取消返回 499，排队超时返回 504。
- Backend cancellation lag：排队控制记录首次 cancellation/deadline 的原因与时间点；Whisper、CosyVoice、SD 在已经进入 backend 后，于实际执行结束时累计 completion lag，并在 `/health` 和 Prometheus 指标中按 lane/reason 暴露 observation、sum、max。排队前已拦截的请求不计入该指标。
- 异步取消生命周期修复：Whisper backend abort callback 改为只引用 QueueItem 共享的 atomic cancellation flag；Whisper/CosyVoice/SD 不再把栈绑定 HTTP request 引用保存到后台队列，消除客户端断开后的 use-after-free。Windows standalone mode 同时修复 CRT/WinAPI 环境变量不同步问题。
- WorkerSupervisor：子进程异常退出、READY 超时、Registry 失败状态和父进程 reservation 收敛。
- Worker generation/launch-ticket 握手：每次加载分配单调递增 generation 和一次性 token，父进程 reservation、ProcessInfo、子进程启动参数及 `/models/status` 使用同一 ticket；旧 READY/UNLOADED、旧进程退出和旧 READY 超时不能覆盖或释放新 generation。
- 动态 Worker graceful shutdown：父进程通过 ticket 鉴权的内部接口请求子进程自行停止 Workflow/WebService 并退出；在配置化 deadline 内未退出时才强制终止。正常退出标记为 expected exit，不计入 worker crash，并分别暴露 graceful/forced shutdown 指标。
- 动态 Worker 端口冲突快速处理：父进程串行探测 `master_port + instance_id` 候选端口，跳过已占用端口并重分配 instance id；子进程在昂贵模型加载前执行二次端口预检，冲突时快速失败。探测次数由 `worker_supervisor.port_probe_attempts` 限制，冲突次数进入 health/metrics。
- 跨平台 Fake Worker 故障注入工具及 CTest 组件门禁：覆盖 ready → infer → ticket 鉴权 shutdown → expected exit、ready 前非零退出、ready 后崩溃和永久挂起强杀；Windows、Linux Release 均已实跑通过。
- 控制面并发压力门禁：`hlie.stress.control-plane` 将 ServiceSlot/Registry generation/atomic eviction 与 reservation ledger 的 reserve/bind/commit/release/rollback 交错执行；当前每轮覆盖 1000 次请求/卸载/重载交错和 4000 次并发 reservation transaction，并持续检查生命周期和账本不变量。
- 可重复真机硬件门禁：`hlie-p0-hardware-smoke` 在独立端口自启/自清理服务，覆盖主进程模型、资源不足 503、READY 前 Worker crash、READY 后 Worker crash、generation 恢复、代理推理、graceful unload、最终账本归零和日志等级契约。脚本只终止本次服务返回的 PID，结果写入 `artifacts/p0_hardware_smoke/<timestamp>`。
- Worker launch manifest 生命周期收口：文件名从秒级时间戳改为“时间戳 + UUID”，避免并发启动覆盖；Worker 解析一次性 manifest 后立即删除，父进程在 CreateProcess/fork 失败、启动后立即退出、异常退出和强杀路径兜底删除，端口分配失败和写文件失败也会清理。修复前 `meta_cache` 会每次启动遗留一个 `.hlif`，修复后的 500-cycle soak 新增残留为 0。
- 主端口配置统一为 env-first：WebService listener、Workflow、Worker port allocator/preflight 现在共同读取 `HLIE__SERVER__PORT`；修复了控制面按环境变量计算 isolated-port、真实 HTTP socket 却仍绑定 `sys.cfg` existing-service-port 的端口分裂。
- 控制面日志分级收口：高频路由、HAL 候选和 stale generation 降为 DEBUG；正常 reserve/commit/unload/expected exit 使用 INFO；容量拒绝、busy、fallback 和成功 orphan 回收使用 WARN；Worker crash、READY timeout、reservation bind/commit 失败保留 ERROR。Supervisor 状态只在变化时记录，当前值由 health/metrics 暴露。主 logger、TCIM 和 LLM backend 的日志等级统一使用 env-first 配置，`HLIE__LOG__LEVEL` 不再被 `sys.cfg` 的旧值覆盖。
- Linux 主程序解除 root 硬依赖：移除 `geteuid() == 0` 启动拦截；删除与 `HLIEProcessManager` 争抢 `waitpid` 的全局 `SIGCHLD` handler；配置的 `/var/log/HLIECpp` 不可写时回退到 `$XDG_STATE_HOME/HLIECpp/logs` 或 `~/.local/state/HLIECpp/logs`；动态 Worker 使用 `/proc/self/exe` 解析出的当前程序路径启动。安装/注册 systemd 服务仍可要求管理员权限，但服务进程本身不再因 UID 非 0 被拒绝。
- Linux sanitizer 门禁：新增 hicpp 自有 `HLIE_SANITIZE_ADDRESS`、`HLIE_SANITIZE_UNDEFINED`、`HLIE_SANITIZE_THREAD` CMake 开关和 rootless `ci/run_linux_sanitizers.sh`。ASAN+UBSAN 覆盖完整 unit、控制面 stress 和 Fake Worker supervisor；TSAN 使用修复后的 `HLIE_TESTS_ONLY=ON` 纯控制面目标，不加载 TCIM/HAL/推理 SDK。脚本强制选择同版本 C/C++ 编译器并在编译器变化时重建缓存，避免混用不同 sanitizer runtime。

## P0 完成状态

- P0 控制面加固已完成并通过当前约定门槛：`/live`、`/ready`、控制面健康信息和 Prometheus 指标已落地，Windows/Linux Release、并发压力、故障注入、sanitizer、真机模型 smoke 与 2 小时 mixed-workload soak 均通过。
- 24 小时 mixed-workload soak 保留为周末扩展长稳门禁，不阻塞本轮 P0 封板；若扩展长稳暴露问题，按 P0 回归处理。

## 后续设计项（不阻塞当前 P0）

- 主进程本地模型如果在 backend 加载内部永久卡死，同时 Registry 一直保持 LOADING，当前保守策略会继续保护 reservation，并通过 `expired_pending_protected` 暴露，而不会冒险释放仍可能被硬件占用的资源。该场景需要本地加载 watchdog、可终止的 backend load handle，或进一步的模型进程隔离；由于涉及加载线程终止语义和硬件状态确认，归入后续控制面设计，不作为当前资源账本 P0 的阻塞项。
- CosyVoice/SD 执行中的原生硬取消依赖 TCIM/backend 提供可中断点或 cancel handle；当前 TCIM 不支持，因此 hicpp 侧以“排队前阻止启动、执行中停止后续输出、记录 cancellation lag、保证服务与账本收敛”为边界。真机 CosyVoice 测得客户端断开后 backend 仍运行约 19.4 秒。该限制需要跟随 TCIM 能力演进，不作为当前 hicpp P0 阻塞项。
- 当前 Worker 端口方案属于 spawn 前探测加子进程二次预检，仍有极短的 TOCTOU 窗口；如需严格的零竞态端口租约，应实现父子进程 socket handoff/句柄继承。

## 当前验证结果

- Release 主程序目标：通过。
- Worker 端口专项单测：3 cases，16 assertions，全部通过；包含真实 TCP listener 占用检测。
- Cancellation 专项单测（生命周期修复后）：6 cases，21 assertions，全部通过。
- Fake Worker 组件测试：4 cases，36 assertions，全部通过；包含父进程对异常退出 Worker 的 launch manifest 兜底清理。
- C++ 单测：Windows 144 cases / 8554 assertions，全部通过。
- 控制面压力测试：2 cases，7628 assertions；连续重复 10 轮全部通过，单轮约 199～308 ms。
- CTest：`hlie.unit`、`hlie.stress.control-plane`、`hlie.component.worker-supervisor`、`hlie.component.mirage`，4/4 通过。
- Linux x86_64 Release：主程序、单测和 Worker supervisor 组件目标构建通过；CTest 3/3 通过（远端未安装可选 Mirage），单测为 144 cases / 8553 assertions，少于 Windows 的 1 条为平台条件断言。
- Linux 非 root 真机启动：用户 `test-user` 在端口 isolated-port 启动成功，HAL 初始化和 device 0 枚举成功；`/live` 返回 200，空载 `/ready` 按语义返回 503，`/health` 返回 200；应用、access、boot 和 TCIM 日志回退到 `~/.local/state/HLIECpp/logs` 且属主为 `test-user:test-user`。SIGTERM 后进程和 isolated-port listener 均清理，既有 existing-service-port root service 未受影响。
- Linux ASAN+UBSAN：`hlie.unit`、`hlie.stress.control-plane`、`hlie.component.worker-supervisor` 3/3 通过，开启 leak detection、遇错立即终止；未发现内存越界、use-after-free、泄漏或未定义行为。
- Linux TSAN：专有 SDK 无关的控制面 unit/stress 2/2 通过；未报告 Registry、ServiceSlot、admission、cancellation metrics 或 device reservation ledger 的 data race。
- Sanitizer 首轮暴露并修复两项构建问题：远端默认 `cc=gcc-9`、`c++=g++-13` 导致 `libasan.so.5/.8` 同进程冲突；测试又将静态 zlib 同时链接进 `libhlie-base.so` 与单测 executable，触发 `z_errmsg` ODR violation。现已统一编译器，并由 hlie-base 提供 gzip 读取入口以保持单一 zlib 实例。
- `git diff --check`：通过（仅保留 Git 对换行格式的提示）。
- 日志等级真机验证：以 `HLIE__LOG__LEVEL=info` 启动并连续访问空载 health，新增主日志中 DEBUG 行为 0，`model service not initialized` 行为 0；资源拒绝记录为 WARN，进程层异常退出记录为 WARN，WorkerSupervisor 收敛记录为 ERROR，正常 reservation release/unload 记录为 INFO。
- 自动化硬件 smoke：Windows Release、独立主端口 isolated-port 通过；主模型 committed `16272MB`，资源不足返回 503 且账本不变，READY 前观测 pending `16485MB` 后 crash 回滚，Worker generation 按 `1 → 2 → 3` 推进，2 次 unexpected exit 和 1 次 graceful shutdown 均收敛，最终两个设备 pending/committed 回到基线。
- Code Review 收尾后于 2026-08-17 再次运行同一真机 smoke：主进程加载/推理、资源拒绝、READY 前 crash、READY 后 crash、代理推理、generation 恢复、graceful unload、最终账本归零和 INFO/WARN/ERROR 契约全部通过；产物见 `artifacts/p0_hardware_smoke/20260817-120207`。
- 500-cycle 真机 lifecycle soak：500 次动态 Worker load/READY、每 10 轮一次代理推理、500 次 graceful unload 全部通过；父进程 handle `300 → 300`、线程 `38 → 34`、RSS `747388928 → 766697472` bytes、private bytes `904216576 → 923987968`，最后 100 轮仅约增长 1 MB；forced shutdown 为 0，新增 manifest 为 0，最终账本回基线。产物见 `artifacts/p0_hardware_smoke/20260816-165022`。
- Linux 2 小时 mixed-workload soak：2026-08-17 在 `test-user@test-host.example.com` 的独立端口 isolated-port 连续运行 7200 秒，完成 2542 次动态 Worker load/READY/unload，每 10 轮执行一次代理推理；主进程本地模型加载/推理、READY 后故障注入、generation 恢复、graceful unload、最终账本归零和日志等级契约全部通过。父进程 handles `18 → 18`，threads `32 → 33` 后保持稳定，RSS `428044288 → 433856512` bytes，private bytes `404987904 → 409313280` bytes，新增 manifest 为 0；累计 2543 次 graceful shutdown，1 次预期注入的 unexpected exit。远端仅有 device 0 且既有 existing-service-port 服务占用同卡，因此本轮启用 shared-device，并跳过已由前序 smoke 覆盖的资源不足拒绝和 READY 前 crash；既有 existing-service-port 服务未停止或修改。产物见远端 `artifacts/p0_hardware_soak/linux-2h/summary.json`。

## 真机模型验证

- Qwen3.5 0.8B 主进程本地模型：加载、chat、卸载通过。
- 第二个 Qwen3.5 0.8B 变体：动态 Worker READY、existing-service-port 代理推理、graceful unload 通过。
- 动态 Worker 外部 kill：约 2 秒内离开路由，unexpected exit 增长，reservation 和端口释放；随后新 generation 恢复加载成功，旧 generation status 被拒绝。
- 动态 Worker READY 前 kill：父进程先观测到 device 1 `pending=16485MB, committed=0`，在启动后 246ms 强杀 Worker；下一次约 500ms 的监督周期内 pending 归零、端口和进程释放，本地模型保持 loaded。
- 真机资源不足拒绝：关闭自动驱逐后，device 0 已有 `16272MB` committed、仅余 `1832MB` admissible；再申请 `16485MB` 返回 HTTP 503，前后 pending/committed 不变且未启动 Worker。
- HAL 对账：2026-08-16 真机曾观测到 HAL used 从 0 增至 `847MB`；Code Review 后不再把这种无法归因的聚合增长用于消化 `committed_unobserved_mb`。按当前保守策略，该 reservation 在相同观测下仍保留 `16272MB` committed-unobserved，避免外部/root 服务增长制造超卖窗口。
- Whisper medium：正常转写通过；修复前可稳定复现客户端断开后的进程崩溃，修复后同一 503 ms 断开测试服务保持存活，backend cancellation lag 为 127 ms，后续正常转写和卸载通过。
- CosyVoice 0.5B：正常 WAV 合成通过；504 ms 客户端断开后服务保持存活，backend cancellation lag 为 19413 ms，后续合成和卸载通过，reservation 归零。

详细记录见 [P0_MODEL_SMOKE_20260816.md](P0_MODEL_SMOKE_20260816.md)。

P0 的设计、验收门禁和灰度策略见 [P0_ARCHITECTURE_HARDENING.md](P0_ARCHITECTURE_HARDENING.md)。
