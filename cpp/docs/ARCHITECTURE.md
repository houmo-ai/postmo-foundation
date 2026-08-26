# HLIECpp Architecture

This document describes the current model loading, service lifecycle, registry, and residency architecture.

## Architecture Hardening Roadmap

P0 control-plane hardening is complete: service lifecycle concurrency, transactional device reservations, bounded request admission, worker supervision, system-level testing, and minimum production observability are now protected by production invariants and automated gates. P1 uses those guards to decompose `HLIEWorkflow`, unify embedded/process runtimes, and establish explicit control-plane ownership:

- [P0 Architecture Hardening Detailed Design](P0_ARCHITECTURE_HARDENING.md)
- [P0 Implementation Status](P0_IMPLEMENTATION_STATUS.md)
- [P1 Workflow Architecture Refactoring Detailed Design](P1_WORKFLOW_ARCHITECTURE_REFACTORING.md)
- [P1 Implementation Status](P1_IMPLEMENTATION_STATUS.md)

P1 establishes three primary production layers: `hlie-base`, `hlie-core`, and
`hlie-app`. The app links both core and base directly; core also links base,
while reverse dependencies are forbidden. Core owns lifecycle, generation,
resource, and runtime invariants. HTTP, process management, discovery, concrete
model services, and the current HOUMO TCIM implementation remain internal app
modules rather than separate platform/device/backend libraries. Multi-SDK and
multi-generation NPU compatibility are explicitly deferred until a real ABI or
product requirement appears.

Model metadata crosses the app/core boundary through the SDK-free
`HLIEModelSpec`. File, GGUF, and empirical-rule inspection stays in app; core
backend descriptors consume neutral observations and apply deterministic memory
policy without filesystem or inference-SDK access.

Control operations cross future Router/Controller boundaries as typed model
commands and `HLIEControlResult<T>`. Core failures are protocol-neutral; the app
maps them to HTTP/OpenAI-compatible responses and preserves native backend codes
only as diagnostic context.

Concrete model services are joined to core descriptors by the app-owned
`HLIEBackendAdapterRegistry` using stable backend ids. Core descriptors never
contain TCIM-aware service constructors. The current Factory remains a
compatibility façade over the adapter registry.

P1-2 moves dynamic Worker process ownership into app-owned
`HLIEWorkerSupervisor`. It owns logical port leases, spawn, READY probes,
expected/unexpected exits, manifest cleanup and confirmed shutdown, and emits
generation-fenced `HLIERuntimeEvent` values. It does not access Registry,
DeviceService or reservations; P1-5 `HLIEModelController` is now its only
production event sink.

P1-3 moves placement grants, reservation state transitions, Worker reservation
bindings, expired-pending reconciliation and residency planning into the
app-owned `HLIEResourceScheduler`. It consumes a core-neutral
`IHLIEDeviceReservationProvider`; `HLIEDeviceService` implements that port for
the current HOUMO/HAL stack. Workflow no longer owns a reservation map or calls
ledger reservation APIs directly.

P1-4 establishes a unified runtime control plane. Core-owned
`IHLIEModelRuntime` describes identity, capabilities, health, draining and
confirmed stop without HTTP, JSON, concrete Service or SDK types. App-owned
`HLIEEmbeddedModelRuntime` and `HLIEProcessModelRuntime` adapt local Service
bundles and supervised Workers. `HLIERuntimeDirectory` stores only
generation-fenced runtime handles; `HLIEModelRuntimeRegistry` remains the
lifecycle source of truth. The Scheduler is the sole owner of externally
reserved grants from reserve through confirmed-stop release.

P1-5 moves lifecycle orchestration into app-owned `HLIEModelController`.
The Controller privately owns the app Registry façade, consumes all Worker
events, coordinates complete dynamic load/unload transactions with Scheduler,
preserves unconfirmed ownership as `Orphaned`, and owns `ensure_ready()`
generation coalescing/wait notification. Concrete startup side effects enter
through app-owned `IHLIEModelBackendLauncher`; Runtime adapter construction
enters through `IHLIEModelRuntimeFactory`. Workflow no longer starts concrete
services or Workers, publishes Runtime handles, implements the runtime event
sink, or scans raw Service/Process state as an eviction fallback.

## Goals

- Keep model service construction centralized.
- Keep local service lifecycle explicit and reusable.
- Keep Workflow focused on orchestration instead of concrete service ownership.
- Support multi-model residency with safe unload, LRU eviction, active-request protection, and pinned models.
- Avoid holding Workflow locks while stopping model services or subprocesses.

## Main Components

### HLIEResourceScheduler

Location:

- `include/hlie_app/resource/hlie_resource_scheduler.h`
- `src/hlie_app/hlie_resource_scheduler.cpp`
- `include/hlie_core/resource/hlie_device_reservation_provider.h`
- `include/hlie_core/resource/hlie_reservation_handle.h`

Responsibilities:

- Validate neutral placement intent and request a hardware-backed grant through `IHLIEDeviceReservationProvider`.
- Own dynamic Worker reservation bindings and pending-to-committed state transitions.
- Fence bind/commit/release by model generation ticket.
- Reconcile expired pending grants against live-owner ticket snapshots.
- Return pure resource-pressure, loaded-limit, and idle-TTL eviction plans.
- Expose ledger and scheduler metrics as immutable snapshots.

Non-responsibilities:

- It does not parse HTTP requests or device strings.
- It does not read Registry or ProcessManager directly.
- It does not start, stop, or evict a runtime.
- It does not depend on a concrete model service or proprietary SDK type.

### HLIEWorkerSupervisor

Location:

- `include/hlie_app/runtime/hlie_worker_supervisor.h`
- `src/hlie_app/hlie_worker_supervisor.cpp`
- `include/hlie_core/runtime/hlie_runtime_event.h`

Responsibilities:

- Own Worker thread, runtime identity, logical port lease and launch-manifest cleanup.
- Spawn and observe child processes through the generic ProcessManager adapter.
- Probe READY once per generation and classify confirmed expected/unexpected exits.
- Attempt ticket-authenticated graceful shutdown, then force termination at the configured deadline.
- Publish typed events outside internal locks and expose immutable metrics snapshots.

Non-responsibilities:

- It does not mutate Registry or reservation state.
- It does not decide eviction, placement or recovery policy.
- It does not construct HTTP API responses or concrete model services.

### IHLIEModelRuntime and HLIERuntimeDirectory

Locations:

- `include/hlie_core/runtime/hlie_model_runtime.h`
- `include/hlie_app/runtime/hlie_embedded_model_runtime.h`
- `include/hlie_app/runtime/hlie_process_model_runtime.h`
- `include/hlie_app/runtime/hlie_runtime_directory.h`

Responsibilities:

- Expose a transport-neutral control contract for identity, capabilities,
  health, draining and deadline-aware confirmed stop.
- Adapt a local Service bundle or a supervised Worker without exposing its
  concrete implementation in the public runtime contract.
- Publish and resolve runtime handles by exact launch ticket/generation.
- Reject stale publication and prevent an old generation from stopping a new
  runtime.
- Keep health inspection outside the Directory lock and expose immutable
  runtime snapshots to health/metrics.

Ownership rules:

- `HLIEModelRuntimeRegistry` is the lifecycle source of truth; the Directory is
  a handle index, not a second state machine.
- Embedded Runtime owns the control lifecycle of its Service bundle; Process
  Runtime delegates Worker lifecycle to `HLIEWorkerSupervisor`.
- A Runtime stop result must be confirmed before Scheduler releases its
  external reservation. On unconfirmed stop, Workflow restores the exact
  handle and aborts eviction.
- Inference dispatch remains intentionally absent from `IHLIEModelRuntime`.
  P1-6 app-owned `IHLIEInferenceEndpoint` and `HLIEInferenceDispatcher` now
  acquire a Controller request ticket, exact Runtime handle and matching
  generation endpoint as one request session without changing the core Runtime
  control contract.
- Embedded endpoint resolution also verifies that the concrete Service snapshot
  belongs to the ticket's canonical model key; matching only the service kind is
  insufficient and fails closed.

Non-responsibilities:

- RuntimeDirectory does not write Registry state, allocate devices, stop
  replacement handles or choose eviction policy.
- Core Runtime does not understand HTTP DTOs, Worker proxy URLs, concrete
  Service requests, TCIM/GGML types, `void*` or `std::any`.

### HLIEModelServiceFactory

Location:

- `include/hlie_app/hlie_model_service_factory.h`
- `src/hlie_app/hlie_model_service_factory.cpp`

Responsibilities:

- Classify a model into a service kind: LLM, SD, Whisper, or CosyVoice.
- Validate supported model service types.
- Create concrete local service objects.
- Keep constructor details out of `HLIEWorkflow`.

Non-responsibilities:

- It does not initialize, run, stop, or unload services.
- It does not touch registry state.
- It does not make device allocation decisions.

### HLIEModelServiceManager

Location:

- `include/hlie_app/hlie_model_service_manager.h`
- `src/hlie_app/hlie_model_service_manager.cpp`

Responsibilities:

- Own live local service pointers:
  - `LLModelService`
  - `SDModelService`
  - `WhisperModelService`
  - `CosyVoiceModelService`
- Provide typed getters for request routing.
- Provide `get_by_type(...)` for generic service lookup.
- Provide `load_*` APIs for local service creation and initialization.
- Provide `start_*` APIs for local service creation, initialization, and `run()`.
- Provide `stop_and_reset_*` APIs for non-lock stop paths.
- Provide `detach_*` APIs so Workflow can remove ownership under lock and stop outside the lock.

Lifecycle semantics:

```text
load_*  = create service + init
start_* = create service + init + run
```

Safety rule:

- `load_*` does not silently replace an existing live service.
- If a service already exists, callers must explicitly unload, detach, stop, or reset it first.

Non-responsibilities:

- It does not update `HLIEModelRegistry`.
- It does not call `update_status(...)`.
- It does not create subprocesses.
- It does not parse HTTP requests or write HTTP responses.
- It does not own model residency policy.
- It does not decide device placement.
- It does not know server ports, master/slave topology, or replicas mode.

This keeps the manager as a local service lifecycle manager, not a small Workflow.

### HLIEModelController

Locations:

- `include/hlie_app/control/hlie_model_controller.h`
- `src/hlie_app/hlie_model_controller.cpp`

Current responsibilities:

- Privately own `HLIEModelRegistry` and expose immutable model/runtime reads.
- Be the sole `IHLIERuntimeEventSink` for Worker READY, timeout and exit facts.
- Fence state changes by launch ticket/generation.
- Coordinate Runtime stop, Scheduler release and Registry publication.
- Mark unconfirmed stop/exit ownership `Orphaned` without dropping the launch
  token, endpoint, Runtime handle or reservation.
- Coalesce concurrent `ensure_ready()` callers so one receives the launch
  ticket and all others wait for that same generation.
- Own the state-change condition variable used by load waiters.
- Sequence dynamic claim/bind, backend launch, Runtime creation/publication,
  track/commit, Ready/await-READY and rollback as one generation-fenced
  `load_model()` transaction.

Non-responsibilities:

- It does not parse HTTP requests or map HTTP/OpenAI error bodies.
- It does not include concrete ModelServiceManager, ProcessManager,
  DeviceService or proprietary SDK types.
- It does not implement inference dispatch.
- It does not parse model artifacts or directly construct concrete Runtime
  adapters; it submits neutral create requests to `IHLIEModelRuntimeFactory`.
- It does not include concrete startup dependencies; it submits app-neutral
  launch requests to `IHLIEModelBackendLauncher`.

### HLIEWorkflow

Location:

- `include/hlie_app/hlie_workflow.h`
- `src/hlie_app/hlie_workflow.cpp`

Responsibilities:

- Parse API requests.
- Resolve model metadata and model paths.
- Submit placement/reservation commands to `HLIEResourceScheduler`.
- Submit catalog and lifecycle commands to `HLIEModelController`.
- Map Controller/Launcher results to the existing HTTP response contract.
- Route requests to local services or subprocess proxies.
- Coordinate master/slave status updates.
- Keep long-running service stop operations outside the Workflow mutex.

Workflow no longer defines rollback/Runtime-stop reconciliation helpers. Its
remaining `mark_local_model_started(...)` compatibility helper performs
transport notification around a Controller state command.

### HLIEModelRegistry

Responsibilities:

- Own model instance metadata.
- Track model status and port.
- Track active request leases.
- Protect loaded models from eviction while they are in use.
- Track last-used timestamps for LRU selection.
- Track eviction state to avoid duplicate unload of the same model.

The Registry is the source of truth for model residency state and is privately
owned by `HLIEModelController`. Service pointers live in
`HLIEModelServiceManager`; Runtime handles live in `HLIERuntimeDirectory`.

### HLIEModelResidencyManager

Responsibilities:

- Apply an injected residency policy and select unload candidates for:
  - memory pressure
  - `max_loaded_models`
  - idle TTL
- Respect pinned models.
- Respect active request leases.
- Prefer LRU-style eviction when unload is allowed.

The core residency manager is a pure selector. ResourceScheduler owns plan
construction; Workflow temporarily performs the actual unload orchestration.

## Start Flow

### Local service start

```text
HTTP /start
  -> Workflow parses request and model info
  -> Workflow validates version and device requirements
  -> ResourceScheduler returns a PendingReservationHandle
  -> Workflow calls ModelController.load_model(...)
  -> ModelController claims/validates the generation ticket
  -> ModelController binds the pending grant to the ticket
  -> BackendLauncher starts the concrete local service
       -> Factory creates concrete service
       -> service.init(...)
       -> service validates and borrows the external grant
       -> service.run()
  -> RuntimeFactory creates EmbeddedModelRuntime
  -> ModelController publishes the exact-generation Runtime
  -> ModelController asks Scheduler to track and commit the grant
  -> ModelController marks Ready for the same launch ticket
  -> Workflow updates master status if needed
```

The Service never commits or releases an externally supplied grant. Scheduler
remains the sole owner until the Runtime reports confirmed stop. Initial/default
models without an external grant retain their existing self-reserve lifecycle
for startup compatibility.

On failure:

```text
service init/start fails
  -> ModelServiceManager cleans up local service pointer
  -> ModelController rolls back the still-owned pending handle and generation
  -> HTTP error response

Runtime creation/publication fails after backend start
  -> BackendLauncher or Runtime performs cleanup
  -> confirmed: ModelController releases the exact-generation grant
  -> unconfirmed: ModelController retains the grant and marks Orphaned
```

### Subprocess start

```text
HTTP /start for a model that already has a local base service
  -> Workflow obtains a pending grant and calls ModelController.load_model(...)
  -> ModelController claims/binds an exact-generation launch ticket
  -> BackendLauncher writes the one-time manifest
  -> BackendLauncher asks WorkerSupervisor to lease a port and spawn
  -> RuntimeFactory creates ProcessModelRuntime
  -> ModelController tracks the pending grant and publishes the Runtime
  -> WorkerSupervisor confirms READY for the same generation
  -> ModelController commits the tracked grant and publishes Ready
```

Subprocess services still use `HLIEModelServiceFactory` and
`HLIEModelServiceManager` inside the Worker during initialization.
Process/port/manifest ownership is in WorkerSupervisor/BackendLauncher;
Registry publication and READY/exit reconciliation enter Controller only.

## Unload Flow

Manual unload and eviction follow the same lock principle:

```text
Registry
  -> claim eviction for the exact generation
RuntimeDirectory
  -> detach the exact Runtime handle
Runtime
  -> begin_draining()
  -> stop(deadline, reason) outside Workflow/Directory locks
confirmed
  -> ResourceScheduler releases the generation-bound reservation
  -> Registry marks the same generation Unloaded
unconfirmed
  -> RuntimeDirectory restores the same handle
  -> Registry aborts eviction
```

This avoids deadlock and avoids holding Workflow locks while model services stop external resources or subprocesses.

## Residency Flow

Residency is triggered by:

- memory pressure during model start
- `max_loaded_models`
- idle TTL background worker

General flow:

```text
Workflow detects pressure or limit
  -> ResourceScheduler builds a plan through core HLIEModelResidencyManager
  -> Registry rejects candidates that are pinned, busy, active, or already evicting
  -> Workflow detaches the generation-fenced Runtime handle
  -> Runtime drains and stops outside orchestration locks
  -> Scheduler releases resources only after confirmed stop
  -> Registry marks the same generation unloaded
```

## Locking Rules

- Do not call service `stop()` while holding Workflow mutex.
- Do not call subprocess stop while holding Workflow mutex.
- Use `detach_*` under lock, then stop outside lock.
- Registry eviction state must be aborted on all-or-nothing stop refusal.
- Active request leases must prevent eviction of in-flight models.
- Request leases are generation-bound. Releasing a stale response from an old
  Worker generation must not decrement the active count of a reloaded model.
- Registry request accounting exposes only ticket-based begin/end operations;
  model-key-only release is intentionally unavailable because it cannot prove
  generation ownership.
- Response cleanup runs before its model lease and admission permit are
  released.
- Device reservations are released only after local service teardown or
  confirmed Worker process exit.
- Worker READY publication is fail-closed. A rejected callback makes the
  Worker enter draining, and a supervisor-side reservation reconciliation
  failure terminates the Worker before releasing its reservation.

## Logging Levels

Control-plane logs use the following severity contract:

- `DEBUG`: high-frequency routing details, HAL search candidates, request-level diagnostics, client cancellation, and stale generation events that were safely rejected.
- `INFO`: successful lifecycle transitions such as reserve/bind/commit/release, model load/unload, Worker start, expected exit, graceful shutdown, and completed recovery.
- `WARN`: recoverable abnormal conditions such as capacity rejection, busy unload, forced fallback, protected expired reservations, orphan reservation rollback, and invalid external control requests.
- `ERROR`: unrecovered failures or invariant violations such as process creation failure, unexpected Worker crash, READY timeout, reservation commit/bind failure, and service initialization failure.

Logging rules:

- Do not log normal request routing or polling loops at `INFO`.
- Do not emit the same state continuously from a supervisor loop; log transitions and expose the current value through metrics.
- Include stable identifiers where available: model key, generation, reservation id, process id, instance id, device id, and failure reason.
- Streaming payload contents and Worker launch tokens are never logged;
  request-level stream diagnostics use `TRACE` and byte counts only.
- Keep prompts, request bodies, launch tokens, and other sensitive payloads out of lifecycle logs. Sensitive process arguments must remain redacted.
- A successful fallback or reconciliation is normally `WARN`, not `ERROR`; the original unrecovered failure may still be `ERROR` at the layer that owns it.
- `log.level` and `log.tcim_level` honor environment overrides during logger initialization; backend verbosity must resolve the same effective level as the application logger.

## Current Boundaries

Good boundaries:

- Core BackendRegistry owns neutral backend matching and pure policy.
- App BackendAdapterRegistry/Factory owns concrete service construction.
- Manager owns local service slots; EmbeddedModelRuntime owns their control lifecycle.
- WorkerSupervisor owns dynamic Worker process/port/manifest supervision.
- Core IHLIEModelRuntime defines the transport-neutral control contract;
  app Runtime adapters own embedded/process control lifecycle.
- RuntimeDirectory owns only generation-fenced handles; Registry remains the
  sole lifecycle state authority.
- Core runtime Registry owns lifecycle and generation state; the app Registry is a compatibility/catalog façade.
- Core residency manager owns pure eviction candidate selection.
- ResourceScheduler owns placement grants, Worker reservation bindings,
  pending/committed transitions, reconciliation and residency plans.
- ModelController privately owns Registry writes, Worker event reconciliation,
  confirmed Runtime eviction/stop and `ensure_ready()` coalescing.
- Core `IHLIEModelCatalog` exposes immutable ModelSpec reads; app
  `HLIEModelCatalog` owns static `HLIEModelInfo`/compatibility payload, while
  Registry combines it with RuntimeRegistry snapshots at read time.
- ModelDiscoveryService owns directory snapshots, stability, generations and
  its worker thread. Parsed updates are published only after a second snapshot
  verifies that the directory did not change during inspection.
- ControlPlaneReconciler owns the periodic timer, live-owner reconciliation,
  endpoint cleanup trigger and idle-TTL trigger. Scheduler still plans and
  Controller still executes lifecycle changes.
- ApplicationCompositionRoot is the single constructor for the current HOUMO
  DeviceService/Scheduler/Controller/Dispatcher/Discovery/Reconciler graph. It
  receives config/process observations as callbacks and does not read globals.

Known transitional areas:

- Workflow still parses model-specific HTTP start parameters and requests the
  initial pending grant. Controller owns bind/track/commit/rollback and concrete
  backend sequencing through BackendLauncher.
- `HLIEModelServiceManager` still exposes mutable typed shared-pointer getters for migration convenience.
- Inference handlers now enter through the app-owned Dispatcher. Concrete local
  Service snapshots and Worker proxy construction are confined to the default
  endpoint resolver; dispatch deliberately remains outside the core Runtime
  control contract.
- `HLIEDefaultModelInitializationService` owns configured primary/SD3
  selection, default projector validation, LLM launch-parameter preparation,
  and concrete Service initialization. It returns typed Initialized/ZeroLoad/
  Failed outcomes; Workflow only maps the legacy process return code.
- Worker status publication, typed status application, processor snapshots, and
  shutdown ticket handling are owned by `HLIEWorkerControlService`;
  `HLIEWorkerControlApplicationService` owns the legacy GET/POST status and
  shutdown HTTP/JSON mapping. Workflow keeps only local/process observation
  callbacks.
- `HLIEEmbeddedStartupService` owns initialized embedded Service startup,
  device observation, Runtime publication, startup rollback, unpublished
  Service cleanup, and ready callback ordering;
  `HLIEDefaultEmbeddedStartupTargetProvider` is the only adapter that expands
  current ServiceManager slots into startup targets. Lifecycle residual cleanup
  remains only as a final safety net for failures outside this transaction.
- `HLIEDocumentApplicationService` owns PDF request inspection, document
  operation selection, Dispatcher delegation, and the LLM-owned document
  companion stop callback. PDFium/layout/OCR implementation remains in
  `DocumentPipelineService`; successful per-request routing is logged at
  `DEBUG`, not `INFO`.
- `HLIEAsrApplicationService` owns transcription model selection, the
  Qwen3-ASR/ForceAligner word-timestamp pipeline, admission lifetime, realtime
  stream decode, and listener start/stop. `IHLIEAsrRealtimeTransport` keeps
  uWebSockets in the default adapter; normal request routing is `DEBUG`, while
  listener lifecycle transitions are `INFO`.
- `HLIESpeechApplicationService` owns CosyVoice/Qwen3-TTS request validation,
  alias selection, loading wait, generation-bound dispatch target, Qwen
  admission lifetime, backend response header, and speech error mapping.
  Concrete TTS services and Worker proxy behavior remain behind inference
  endpoints; successful request selection is logged at `DEBUG`.
- `HLIEModelListApplicationService` owns `/v1/models` compatibility DTO
  assembly, loaded-record preference, local/Worker instance mapping, alias
  resolution, and `loaded_info` extensions. Controller, ServiceManager, and
  ProcessManager remain behind composition-provided read snapshots.
- `HLIEModelUnloadApplicationService` owns the legacy unload selectors,
  type aliases, and HTTP/error mapping. Controller remains the only eviction
  transaction owner; ServiceManager and Worker publication remain behind
  composition-provided adapters.
- `HLIEStopServiceApplicationService` owns `/hlie/v1/stop` compatibility DTOs,
  local all-model claim/eviction coordination, and ticketed/legacy Worker stop
  reconciliation. Controller remains the lifecycle owner and Supervisor
  remains the process-stop owner behind composition-provided adapters.
- `HLIEObservabilityApplicationService` owns live/ready/health JSON and
  Prometheus rendering. A lightweight model-summary provider serves readiness;
  health and metrics share a neutral immutable snapshot assembled by
  composition from the existing state owners.
- `HLIELlmApplicationService` owns LLM model selection, discovery retry,
  POST-only auto-load adaptation, readiness error mapping, response-lifetime
  admission, document-aware routing, and `/props` + `/slots` compatibility.
  Controller/Readiness/Dispatcher/Document remain the lifecycle, wait, data
  plane, and PDF-operation owners behind composition-provided callbacks.
- `HLIEBootstrapApplicationService` owns main/Worker process-role selection,
  launch-ticket and manifest handling, initial catalog registration, discovery
  baseline ordering, and default initialization result mapping. Concrete
  parsing, Service initialization, Controller registration, and Discovery stay
  behind composition adapters; Runtime start/publication remains a later phase.
- `HLIEApplicationLifecycleService` owns one-shot run, monotonic draining, and
  idempotent concurrent shutdown ordering. It sequences EmbeddedStartup,
  admission, ASR ingress, Reconciler, Discovery, Controller Runtime stop,
  Supervisor shutdown, residual unpublished-slot cleanup, and WorkerControl
  through composition callbacks. ASR ingress is stopped before Whisper/LLM
  Runtime destruction.

These boundaries narrow the remaining P1-7 façade work to residual
compatibility adapters and their failure metrics. Further cleanup must extract
real application services rather than hide the same ownership in a Pimpl or
split the file only; it must not move registry or process logic into the
ServiceManager. The Worker control contract is documented in
[P1-7_WORKER_CONTROL_SERVICE.md](P1-7_WORKER_CONTROL_SERVICE.md),
embedded startup is documented in [P1-7_EMBEDDED_STARTUP_SERVICE.md](P1-7_EMBEDDED_STARTUP_SERVICE.md),
and document request ownership is documented in
[P1-7_DOCUMENT_APPLICATION_SERVICE.md](P1-7_DOCUMENT_APPLICATION_SERVICE.md).
ASR request and realtime ownership is documented in
[P1-7_ASR_APPLICATION_SERVICE.md](P1-7_ASR_APPLICATION_SERVICE.md), and speech
request ownership is documented in
[P1-7_SPEECH_APPLICATION_SERVICE.md](P1-7_SPEECH_APPLICATION_SERVICE.md).
Default Service initialization and the two-stage startup contract are
documented in
[P1-7_DEFAULT_MODEL_INITIALIZATION_SERVICE.md](P1-7_DEFAULT_MODEL_INITIALIZATION_SERVICE.md).
The model-list compatibility view is documented in
[P1-7_MODEL_LIST_APPLICATION_SERVICE.md](P1-7_MODEL_LIST_APPLICATION_SERVICE.md).
The model-unload compatibility and ownership boundary is documented in
[P1-7_MODEL_UNLOAD_APPLICATION_SERVICE.md](P1-7_MODEL_UNLOAD_APPLICATION_SERVICE.md).
The local-batch and Worker-stop compatibility boundary is documented in
[P1-7_STOP_SERVICE_APPLICATION_SERVICE.md](P1-7_STOP_SERVICE_APPLICATION_SERVICE.md).
The health/readiness and metrics projection is documented in
[P1-7_OBSERVABILITY_APPLICATION_SERVICE.md](P1-7_OBSERVABILITY_APPLICATION_SERVICE.md).
The admitted LLM compatibility and auto-load boundary is documented in
[P1-7_LLM_APPLICATION_SERVICE.md](P1-7_LLM_APPLICATION_SERVICE.md).
The main/Worker initialization state machine is documented in
[P1-7_BOOTSTRAP_APPLICATION_SERVICE.md](P1-7_BOOTSTRAP_APPLICATION_SERVICE.md).
The process-level run/drain/stop state machine is documented in
[P1-7_APPLICATION_LIFECYCLE_SERVICE.md](P1-7_APPLICATION_LIFECYCLE_SERVICE.md).

# P0 implementation status

See [P0_IMPLEMENTATION_STATUS.md](P0_IMPLEMENTATION_STATUS.md) for the current implementation and verification state.
