# P1-7 Model Unload Application Service

`HLIEModelUnloadApplicationService` owns the compatibility boundary for
`POST /hlie/v1/unload_model` and `POST /models/unload`. It parses the legacy
selectors, maps validation and lifecycle outcomes to HTTP/JSON, and delegates
the actual eviction transaction through composition-provided ports.

This extraction does not create another lifecycle owner. `HLIEModelController`
remains the only owner of eviction claims, generation fencing, runtime stop and
detach, reservation release, and Registry state publication.

## Class Boundary

```mermaid
classDiagram
    class HLIEWorkflow {
      +get_route_configs()
    }
    class HLIEModelUnloadApplicationService {
      +handle(request) server_http_res_ptr
    }
    class HLIEModelUnloadApplicationDependencies {
      +replicas_mode_provider() bool
      +resolve_model_key(name) optional~string~
      +find_loaded_model_by_type(type) optional~string~
      +is_supported_model_type(type) bool
      +evict_model(key, reason) bool
      +publish_unloaded(key)
    }
    class HLIEModelController {
      +get_instance_snapshot(name)
      +evict_runtime(key, reason) EvictionResult
    }
    class HLIEModelServiceManager
    class HLIEWorkerControlService
    class HLIEApplicationCompositionRoot

    HLIEWorkflow --> HLIEModelUnloadApplicationService : route delegation
    HLIEModelUnloadApplicationService --> HLIEModelUnloadApplicationDependencies
    HLIEApplicationCompositionRoot --> HLIEModelUnloadApplicationService : constructs
    HLIEModelUnloadApplicationDependencies ..> HLIEModelController : identity and eviction adapters
    HLIEModelUnloadApplicationDependencies ..> HLIEModelServiceManager : locked type snapshot
    HLIEModelUnloadApplicationDependencies ..> HLIEWorkerControlService : unloaded publication
```

The application service does not include or store Controller, Registry,
ServiceManager, ProcessManager, DeviceService, concrete model Service, TCIM, or
GGML types. Those objects remain behind adapters assembled by
`HLIEApplicationCompositionRoot`.

## Request Sequences

Explicit identity is authoritative when either `model_id` or its legacy
fallback `model` is present:

```mermaid
sequenceDiagram
    participant R as ApiRouter
    participant U as ModelUnloadApplicationService
    participant C as Controller Adapter
    participant W as WorkerControl Adapter

    R->>U: handle(request)
    U->>U: reject replica mode
    U->>U: parse model_id, then model fallback
    U->>C: resolve_model_key(requested)
    alt identity not found
        C-->>U: none
        U-->>R: 404 not_found_error
    else canonical identity
        C-->>U: model key
        U->>C: evict_runtime(key, api_unload)
        alt busy, unloaded, missing runtime, or stop unconfirmed
            C-->>U: false
            U-->>R: 503 / ret 80004
        else eviction confirmed
            C-->>U: true
            U->>W: publish_unloaded(key)
            U-->>R: 200 / code 0 / model key
        end
    end
```

When identity is absent, the service preserves the legacy type selector:

```mermaid
sequenceDiagram
    participant R as ApiRouter
    participant U as ModelUnloadApplicationService
    participant S as Service Snapshot Adapter
    participant C as Controller Adapter
    participant W as WorkerControl Adapter

    R->>U: handle(request)
    U->>U: normalize document to llm, tts to cosyvoice
    U->>U: validate supported model type
    U->>S: find_loaded_model_by_type(type)
    alt no loaded service
        S-->>U: none
        U-->>R: 503 / ret 80004
    else loaded model key
        S-->>U: key
        U->>C: evict_runtime(key, api_unload_by_type)
        C-->>U: result
        U->>W: publish_unloaded(key) after confirmed success
        U-->>R: 200 / code 0
    end
```

## Compatibility Matrix

| Input or condition | Resolution | HTTP result |
| --- | --- | --- |
| `model_id` present | Resolve that identity; ignore `model` and `model_type` | `200` with canonical `model`, or `404` |
| `model` present without `model_id` | Legacy identity fallback | Same as `model_id` |
| Neither identity nor type | Default to `llm` | Type-selected unload |
| `model_type=document` | Use the owning LLM lifecycle | Type-selected unload |
| `model_type=tts` | Preserve legacy CosyVoice selection | Type-selected unload |
| Unsupported type | No lifecycle call | `400`, ret `80006` |
| Replica mode | No lookup or lifecycle call | `501`, ret `80005` |
| Busy, unloaded, runtime missing, or unconfirmed stop | Controller refuses eviction | `503`, ret `80004` |

JSON parsing and non-string identity coercion retain the prior behavior. Worker
`UNLOADED` publication happens only after Controller confirms eviction.

## Ownership And Logging

- The facade never marks Registry state or releases Scheduler reservations.
- It never directly stops an embedded Service or Worker process.
- `document` remains a companion of the LLM runtime, not an independently
  unloadable runtime.
- HTTP success mapping is `DEBUG`; Controller records the authoritative
  lifecycle transition at `INFO`.
- Busy or unmatched eviction and a supported type with no loaded Service are
  recoverable `WARN` events.
- Prompt, response body, transcript, and launch token are never logged.

The architecture gate rejects direct runtime/control dependencies and rejects
return of `handle_unload_model` or `api_unload_by_type` orchestration to
Workflow. Focused tests cover explicit identity precedence, legacy `model`
fallback, replica rejection, not-found and busy outcomes, `document`/`tts`
aliases, default LLM selection, invalid types, absent loaded Services, and
Worker unloaded publication. Both full Release and SDK-free slices pass 4 test
cases and 57 assertions; Windows Release CTest passes 26/26 and SDK-free CTest
passes 24/24.
