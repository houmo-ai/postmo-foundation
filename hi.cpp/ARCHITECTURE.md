# HLIECpp Architecture

This document describes the current model loading, service lifecycle, registry, and residency architecture.

## Goals

- Keep model service construction centralized.
- Keep local service lifecycle explicit and reusable.
- Keep Workflow focused on orchestration instead of concrete service ownership.
- Support multi-model residency with safe unload, LRU eviction, active-request protection, and pinned models.
- Avoid holding Workflow locks while stopping model services or subprocesses.

## Main Components

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

### HLIEWorkflow

Location:

- `include/hlie_app/hlie_workflow.h`
- `src/hlie_app/hlie_workflow.cpp`

Responsibilities:

- Parse API requests.
- Resolve model metadata and model paths.
- Request device memory from `HLIEDeviceService`.
- Register models into `HLIEModelRegistry`.
- Mark model loading, loaded, unloading, and unloaded states.
- Decide when to start a local service or a subprocess.
- Route requests to local services or subprocess proxies.
- Coordinate master/slave status updates.
- Coordinate model unload and eviction.
- Keep long-running service stop operations outside the Workflow mutex.

Key helper methods:

- `rollback_model_start(...)`
  - Marks a failed start back to unloaded.
  - Notifies waiters.
- `mark_local_model_started(...)`
  - Optionally starts ASR websocket routing.
  - Marks the model loaded.
  - Sends slave status update when needed.

### HLIEModelRegistry

Responsibilities:

- Own model instance metadata.
- Track model status and port.
- Track active request leases.
- Protect loaded models from eviction while they are in use.
- Track last-used timestamps for LRU selection.
- Track eviction state to avoid duplicate unload of the same model.

The registry is the source of truth for model residency state. Service pointers live in `HLIEModelServiceManager`; residency metadata lives in `HLIEModelRegistry`.

### HLIEModelResidencyManager

Responsibilities:

- Read residency policy from config.
- Select unload candidates for:
  - memory pressure
  - `max_loaded_models`
  - idle TTL
- Respect pinned models.
- Respect active request leases.
- Prefer LRU-style eviction when unload is allowed.

The residency manager selects candidates. Workflow performs the actual unload orchestration.

## Start Flow

### Local service start

```text
HTTP /start
  -> Workflow parses request and model info
  -> Workflow validates version and device requirements
  -> Workflow marks model loading in registry
  -> ModelServiceManager.start_*(...)
       -> Factory creates concrete service
       -> service.init(...)
       -> service.run()
  -> Workflow marks model loaded
  -> Workflow updates master status if needed
```

On failure:

```text
service init/start fails
  -> ModelServiceManager cleans up local service pointer
  -> Workflow.rollback_model_start(...)
  -> HTTP error response
```

### Subprocess start

```text
HTTP /start for a model that already has a local base service
  -> Workflow generates model info JSON
  -> Workflow allocates instance id
  -> Workflow starts subprocess through HLIEProcessManager
  -> subprocess initializes its native service from --json_info
```

Subprocess services still use `HLIEModelServiceFactory` and `HLIEModelServiceManager` during initialization, but process orchestration stays in Workflow.

## Unload Flow

Manual unload and eviction follow the same lock principle:

```text
Workflow lock
  -> verify target service exists
  -> mark eviction/busy state in registry
  -> detach service pointer from ModelServiceManager
unlock
  -> stop service outside Workflow lock
  -> mark registry unloaded
  -> notify master if needed
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
  -> HLIEModelResidencyManager selects candidate
  -> Registry rejects candidates that are pinned, busy, active, or already evicting
  -> Workflow detaches local service under lock
  -> Workflow stops service outside lock
  -> Registry marks model unloaded
```

## Locking Rules

- Do not call service `stop()` while holding Workflow mutex.
- Do not call subprocess stop while holding Workflow mutex.
- Use `detach_*` under lock, then stop outside lock.
- Registry eviction state must be aborted on all-or-nothing stop refusal.
- Active request leases must prevent eviction of in-flight models.

## Current Boundaries

Good boundaries:

- Factory owns construction.
- Manager owns local service pointers and local service lifecycle.
- Workflow owns orchestration and system state.
- Registry owns model residency state.
- Residency manager owns eviction candidate selection.

Known transitional areas:

- `HLIEModelServiceManager` still exposes mutable typed shared-pointer getters for migration convenience.
- Workflow still contains model-specific request parsing and subprocess start branches.
- Some status-update logic is still type-expanded in `Workflow::run()`.

These are acceptable for the current phase. Future cleanup should reduce mutable getter usage and continue extracting Workflow-local orchestration helpers without moving registry or process logic into the manager.

