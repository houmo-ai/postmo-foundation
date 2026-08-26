# P0 真机模型 Smoke 记录（2026-08-16）

> 历史说明（2026-08-17）：本文保留 2026-08-16 当时实现和真机观测。
> Code Review 后 HAL accounting 已改为更保守的策略：聚合 HAL used 增长
> 无法可靠归因给 hicpp reservation，因此不再抵扣
> `committed_unobserved_mb`。下文旧的增量抵扣数值仅作为历史测试记录，
> 不再代表当前账本契约。

## 测试环境

- Windows Release：`houmo-llm-hi-cpp-win.exe`
- 主服务：`127.0.0.1:7901`
- 配置 bundle：`cmake-build-release-visual-studio/bin/config`
- 模型目录：`E:\models`
- 设备：2 张，每张健康接口报告 `18304 MB`

## LLM 主进程与动态 Worker

### 主进程本地模型

加载：

- 模型：`HiModel_xh2_qwen3.5_0.8b_256_256k_b1_1chip_2cores_v1.3.0_20260429_vit_448_448`
- device：0
- instance：0
- port：7901

结果：

- `/v1/models` 正确报告 loaded、instance 0、device 0。
- `/v1/chat/completions` 返回 HTTP 200。
- admission accepted 计数增长。
- 256K context 的 reservation 为 `16272 MB`；HAL observation 在本次环境中保持 0，因此账本依赖 committed-unobserved 防止超卖。

### 动态 Worker

加载同一基础模型的 `vit_896_896` 变体到 device 1：

- 首次分配 instance 1 / PID 102428 / port 7902。
- 父进程 7901 代理 `/v1/chat/completions` 成功。
- `/health` 报告 2 个 loaded model、1 个 running Worker。
- device 1 reservation 为 `16485 MB`。

正常卸载结果：

- unload API 在约 1.36 秒内返回。
- `graceful_shutdowns` 从 0 增长到 1。
- Worker PID 消失，7902 释放。
- Registry 转为 unloaded，device 1 reservation 归零。
- 主进程本地模型继续 READY。

### 外部 crash 与 generation 恢复

重新加载动态 Worker 后从外部强杀 PID：

- Supervisor 约 2 秒内检测退出。
- `unexpected_exits` 增长。
- Worker 路由和端口被移除，device 1 reservation 归零。
- 主进程本地模型保持 READY。
- 模型可再次加载到新 instance/port，并以新 generation 恢复为 loaded。
- 使用当前 token 发送旧 generation 的 `/models/status` 更新返回拒绝，当前 generation 的 port/PID 未被覆盖。

注意：失败后公开模型状态当前表现为 `status.value=unloaded`，同时携带 `failed=true` 和 `exit_code`。控制面能识别 degraded，但只检查 `status.value` 的客户端可能漏掉失败语义，后续应继续收口结构化失败状态。

### READY 前失败回滚与资源拒绝

补跑动态 Worker 在 READY 前死亡的真实事务回滚：

- 父进程已加载 `vit_448_448` 到 device 0，device 0 committed reservation 为 `16272 MB`。
- 启动 `vit_896_896` 到 device 1 后，父进程观测到 device 1 `pending_reserved_mb=16485`、`committed_reserved_mb=0`。
- 在 Worker 启动后 246ms 强杀进程；下一次约 500ms 的监督周期内，device 1 pending/committed 均归零，Worker 和监听端口消失，unexpected exit 增长。
- device 0 本地模型继续 loaded，证明失败回滚没有影响其他 generation 或其他设备 reservation。

补跑资源不足拒绝：

- 以 `model.auto_unload=false` 启动服务，device 0 已被本地模型占用 `16272 MB`，admissible 为 `1832 MB`。
- 再向 device 0 申请需要 `16485 MB` 的动态变体，返回 HTTP 503。
- 请求前后 pending/committed reservation 不变，目标模型仍 unloaded，没有启动 Worker，本地模型保持 loaded。

HAL 对账当时获得真机观测：HAL used 从 0 增至 `847 MB` 后，旧实现将 `committed_unobserved_mb` 从 `16272 MB` 降至 `15425 MB`，`effective_used_mb` 保持 `16272 MB`。该行为已在 2026-08-17 被保守策略取代；当前实现不会用无法归因的聚合 HAL 增长抵扣 hicpp reservation。

代码侧补充 pending reservation TTL/orphan reconciler：默认 TTL 300 秒，可配置为 30～3600 秒；只有 Registry 当前 owner 和 Worker 进程都已消失时才回滚过期 pending。仍在 LOADING/LOADED 的 owner 或存活 Worker 会被保护，避免仅因超时就把仍被硬件占用的资源重新放行。

## Whisper cancellation 真机复现

模型：

- `HiModel_xh2_whisper-medium_1chip_2cores_v1.4.0_20260721`
- device：0

基线转写：

- 输入 `E:\test_data\audio\61-70968-0000.wav`。
- HTTP 200，约 1.0～1.1 秒完成。

### 修复前故障

对 143.405 秒的 `long.wav` 发起请求，并在约 0.5 秒主动断开客户端：

- HTTP 层返回/记录 499。
- 主进程随后崩溃，7901 退出。

根因：

- Whisper backend 的 `abort_callback_user_data` 指向栈上的 `httplib::Request`。
- 队列控制对象还保存了捕获 `&req` 的 lambda。
- HTTP handler 在 499 返回后销毁 request，但异步 backend 继续访问它，形成 use-after-free。

### 修复

- backend abort callback 只读取随 QueueItem 共享的 atomic cancellation flag。
- HTTP connection callback 只在 HTTP 等待线程内调用；检测断开后设置共享 flag。
- Whisper、CosyVoice、SD 均不再把栈绑定 request 引用保存进异步队列。
- Windows standalone mode 同时改为更新 CRT environment，bootstrap logger 能正确识别 standalone。

### 修复后结果

重复相同取消测试：

- 客户端在 503 ms 超时断开。
- 服务保持存活。
- Whisper cancellation lag 更新为：observations 1、total/max lag 127 ms。
- 随后的正常短音频转写仍返回 HTTP 200。
- 卸载后 device reservation 归零。

## CosyVoice cancellation 真机验证

模型：

- `HiModel_xh2_cosyvoice3-0.5b-2512_256_2k_b1_1chip_2cores_v1.4.0_20260720`
- device：0

正常合成：

- `/v1/audio/speech` 返回 HTTP 200 和 WAV。
- 普通短句约 14.7 秒完成；取消测试后的 `Hi.` 仍能正常合成，约 13.8 秒完成。

客户端断开：

- 长文本请求在 504 ms 主动断开。
- 服务保持存活，没有访问已销毁 HTTP request。
- backend 没有原生硬取消，继续执行直至完成。
- cancellation lag：observations 1、total/max lag `19413 ms`。
- 卸载后 device 0 reservation 归零。

该结果证明生命周期安全修复有效，同时量化了 CosyVoice adapter 缺少原生 cancel handle 的实际影响：客户端离开后硬件仍可能占用约 20 秒。

## 当前结论

已在真机覆盖：

- 主进程本地模型加载、推理、卸载。
- 动态 Worker spawn、READY、代理推理、graceful shutdown。
- 动态 Worker READY 前 crash、父进程 pending reservation 自动回滚。
- 外部 crash、Registry/route/reservation 收敛。
- 真机资源不足拒绝不污染账本，以及 HAL observation/committed-unobserved 对账。
- 新 generation 恢复和旧 generation 状态拒绝。
- Whisper 客户端断开、backend 原生 abort、cancellation lag 和崩溃回归。
- CosyVoice 客户端断开生命周期安全和无硬取消时的真实 cancellation lag。

后续又将上述 LLM/Worker/资源账本步骤固化为
`tests/python/p0/control_plane_hardware_smoke.py` 和 CMake target
`hlie-p0-hardware-smoke`。Windows Release 独立端口 17901 首次完整自动化运行通过：
主模型 committed `16272MB`，READY 前 Worker pending `16485MB` 后回滚，
generation `1 → 2 → 3`，2 次 unexpected exit、1 次 graceful shutdown，最终
两个设备 pending/committed 均回到基线，INFO 运行无 DEBUG 行。产物位于
`artifacts/p0_hardware_smoke/20260816-163839`。

随后补跑 `hlie-p0-hardware-soak` 的 500-cycle 硬门槛：500 次动态 Worker
load/READY、每 10 轮一次代理推理、500 次 graceful unload 全部通过。父进程
handle `300 → 300`、线程 `38 → 34`、RSS 约增加 18.4 MB、private bytes
约增加 18.9 MB，且最后 100 轮只增加约 1 MB；forced shutdown 为 0，最终
账本回到基线。该轮同时验证 launch manifest 生命周期修复，新增 `.hlif`
残留为 0。产物位于 `artifacts/p0_hardware_smoke/20260816-165022`。

仍需补：

- Linux 路径实跑。
- SD 的真实 cancellation lag，以及 CosyVoice/SD stop-during-inference 测试。
- 结构化 FAILED 状态对公开 API 的兼容设计。
- 长时间 mixed workload soak、ASAN/TSAN/UBSAN。
