# P1-7 ASR Application Service

`HLIEAsrApplicationService` owns ASR request orchestration and realtime
listener lifecycle. It preserves the existing Whisper and Qwen3-ASR behavior
while removing model selection, the two-stage word timestamp pipeline, audio
conversion, streaming decode, and WebSocket ownership from `HLIEWorkflow`.

The service is an app-layer boundary for the current HOUMO implementation. It
is not a second inference engine or a multi-SDK provider abstraction.

## Ownership

The service owns:

- direct Whisper versus LLM transcription selection;
- Qwen3-ASR and ForceAligner discovery/selection;
- ASR transcription followed by force alignment for word timestamps;
- LLM admission across the complete two-stage request;
- PCM float-to-WAV adaptation for realtime LLM transcription;
- Qwen streaming response decode and delta delivery;
- realtime listener idempotent start/stop and the historical `7901 -> 7999`
  port mapping.

It does not own Controller state, Runtime publication, request lease
accounting, concrete Whisper/LLM services, uWebSockets, or model loading.
Generation-bound dispatch remains in `HLIEInferenceDispatcher`; model
readiness/auto-load remains behind the existing LLM route callback.

```mermaid
classDiagram
    class HLIEWorkflow {
        +run()
        +stop()
        +initialize_api_router()
    }
    class HLIEAsrApplicationService {
        +handle_transcriptions(request)
        +handle_realtime(request, callback)
        +ensure_realtime()
        +stop_realtime()
    }
    class IHLIEAsrRealtimeTransport {
        <<interface>>
        +start(handler, port)
        +stop()
    }
    class HLIEDefaultAsrRealtimeTransport
    class HLIEWebSocketService
    class HLIEInferenceDispatcher
    class HLIEAdmissionController

    HLIEWorkflow --> HLIEAsrApplicationService : facade calls
    HLIEAsrApplicationService --> HLIEAdmissionController
    HLIEAsrApplicationService *-- IHLIEAsrRealtimeTransport
    HLIEDefaultAsrRealtimeTransport ..|> IHLIEAsrRealtimeTransport
    HLIEDefaultAsrRealtimeTransport *-- HLIEWebSocketService
    HLIEAsrApplicationService ..> HLIEInferenceDispatcher : injected dispatch
```

`WhisperStreamRequest` and language normalization now live in the neutral
`hlie_asr_realtime_contract.h`; neither the ASR application service nor the
Whisper model-service header needs to include the uWebSockets implementation.

## Word Timestamp Sequence

```mermaid
sequenceDiagram
    participant R as ApiRouter
    participant A as AsrApplicationService
    participant C as Model Candidates
    participant L as LLM Route
    participant D as InferenceDispatcher
    participant Q as Qwen3-ASR
    participant F as ForceAligner

    R->>A: handle_transcriptions(request)
    A->>A: acquire one LLM admission permit
    A->>C: resolve ASR and aligner models
    A->>L: route transcription request(stream=false)
    L->>D: generation-bound LlmTranscription
    D->>Q: transcribe audio
    Q-->>A: transcript
    A->>L: route alignment request(text + audio)
    L->>D: generation-bound LlmTranscription
    D->>F: align transcript
    F-->>A: verbose word timestamps
    A-->>R: aligned response with canonical text
```

The same admission permit covers both backend calls. Direct LLM
transcriptions keep the permit until the returned response/stream is destroyed
through `HLIEAdmissionGuardedResponse`.

## Realtime Sequence

```mermaid
sequenceDiagram
    participant W as Workflow lifecycle
    participant A as AsrApplicationService
    participant T as RealtimeTransport
    participant D as InferenceDispatcher
    participant E as InferenceEndpoint

    W->>A: ensure_realtime()
    A->>A: listener policy + model availability
    A->>T: start(handler, local_port + 98)
    T-->>A: WhisperStreamRequest
    alt Whisper model
        A->>D: generation-bound realtime dispatch
        D->>E: dispatch_realtime_transcription()
    else Qwen/LLM model
        A->>A: PCM float -> WAV
        A->>D: route LlmTranscription(stream=true)
        A->>A: decode SSE/native deltas
    end
    W->>A: stop_realtime()
    A->>T: stop()
```

Dynamic Workers do not create the process-global realtime listener. Masters
and replica ports retain the previous listener policy and isolated port
mapping.

## Logging And Verification

- `DEBUG`: successful per-request backend selection, pipeline selection, and
  completed realtime transcription.
- `INFO`: realtime listener start and stop lifecycle transitions.
- `WARN`: admission rejection, unavailable target, or final/delta mismatch.
- `ERROR`: null/failed backend responses, missing stream deltas, or transport
  creation failure.

Failure logs contain status and byte counts, not response bodies or transcript
text. The architecture gate
`hlie.architecture.asr-application-service-dependencies` prevents the service
from depending on Workflow, Controller, ServiceManager, concrete model
services, ProcessManager, DeviceService, WebSocket implementation, Singleton,
TCIM/GGML, or llama server response helpers. It also prevents ASR ownership
from returning to Workflow.

Focused tests cover direct Whisper dispatch, Qwen3-ASR plus ForceAligner
composition, canonical text merge, listener port mapping, idempotent start,
stop, and listener policy rejection.
