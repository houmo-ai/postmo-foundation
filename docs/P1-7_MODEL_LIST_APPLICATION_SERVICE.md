# P1-7 Model List Application Service

`HLIEModelListApplicationService` owns the OpenAI-compatible `/v1/models`
response view. It merges the legacy controller metadata projection with local
embedded-Service and Worker-process snapshots, while preserving existing
HLIE extensions such as status, instances, device ids, capabilities, roles,
and llama parameter summaries.

This service is an app-layer compatibility view, not the core model catalog.
`IHLIEModelCatalog`/`HLIEModelCatalog` remain transport-neutral model identity
and specification owners; JSON and `server_model_meta` stay in `hlie-app`.

## Class Boundary

```mermaid
classDiagram
    class HLIEWorkflow {
      +get_route_configs()
    }
    class HLIEModelListApplicationService {
      +handle() server_http_res_ptr
    }
    class HLIEModelListApplicationDependencies {
      +models() vector~server_model_meta~
      +local_instances() vector~LocalInstance~
      +process_instances() vector~ProcessSnapshot~
      +resolve_model_key(name) optional~string~
      +local_port() int
    }
    class HLIEModelController
    class HLIEModelServiceManager
    class HLIEProcessManager

    HLIEWorkflow --> HLIEModelListApplicationService : route delegation
    HLIEModelListApplicationService --> HLIEModelListApplicationDependencies
    HLIEModelListApplicationDependencies ..> HLIEModelController : metadata adapter
    HLIEModelListApplicationDependencies ..> HLIEModelServiceManager : locked local snapshot
    HLIEModelListApplicationDependencies ..> HLIEProcessManager : neutral process snapshot
```

The application service has no direct Controller, ServiceManager,
ProcessManager, concrete model Service, DeviceService, Singleton, TCIM, or
GGML dependency. Production adapters are assembled by
`HLIEApplicationCompositionRoot`; transport/OS objects never cross the
snapshot contract.

## Request Sequence

```mermaid
sequenceDiagram
    participant R as ApiRouter
    participant L as ModelListApplicationService
    participant C as Controller Adapter
    participant E as Embedded Snapshot Adapter
    participant P as Process Snapshot Adapter

    R->>L: handle()
    par collect stable read models
        L->>C: models()
        C-->>L: server_model_meta snapshots
    and collect local instances
        L->>E: local_instances()
        E-->>L: key + device ids
    and collect workers
        L->>P: process_instances()
        P-->>L: instance/pid/port/model names/devices
    end
    L->>L: prefer loaded duplicate by public name
    L->>C: resolve aliases to canonical model key
    L->>L: merge instances and loaded_info extensions
    L-->>R: 200 application/json list
```

The implementation currently invokes providers sequentially; the `par`
notation describes independent read ownership, not concurrent execution.

## Compatibility Rules

- Models remain de-duplicated by public `meta.name`, preferring a loaded record.
- Local instance id/process id remain `0`; its port remains the current server
  port and empty Service device ids fall back to `loaded_info.device_ids`.
- Worker ports remain `local_port + instance_id`; aliases are resolved through
  Controller snapshots before matching the public model.
- `model_type`, `service_kind`, `role`, `model_family`, `architecture`,
  `multimodal`, `capabilities`, `llama_presets`, `llama_params_hash`, and
  `llama_params_summary` retain their previous output conditions.
- Listing remains read-only and performs no model file, GGUF, preset, or SDK IO.

No per-request route log is emitted: a normal model listing is high-frequency
read traffic and does not represent a lifecycle transition. Failures in
snapshot providers remain owned and logged by their adapters.

The architecture gate prevents orchestration from returning to Workflow and
prevents the service from acquiring concrete control/runtime dependencies.
Focused tests cover loaded-record preference, local and Worker instance
mapping, alias resolution, extension fields, and device metadata fallback in
both full and SDK-free targets. The focused slice passes 2 cases and 16
assertions in each target; Windows Release server/unit builds pass, full
Release CTest is 24/24, and `HLIE_TESTS_ONLY` CTest is 22/22.
