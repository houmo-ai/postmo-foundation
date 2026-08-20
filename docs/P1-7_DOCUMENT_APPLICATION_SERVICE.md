# P1-7 Document Application Service

`HLIEDocumentApplicationService` is the request-boundary owner for PDF routing.
It decides whether an otherwise normal LLM request is a document request and
selects `DocumentPreprocess` or `DocumentOcr`. The existing Dispatcher and
embedded endpoint continue to own the request lease and invoke
`DocumentPipelineService` directly, so the local path gains no HTTP hop and no
PDF payload copy.

## Ownership

The service owns:

- PDF inspection through `DocumentInputAdapter`;
- `document_preprocess_only` compatibility parsing;
- selection of `DocumentPreprocess` versus `DocumentOcr`;
- delegation to the generation-fenced inference dispatch callback;
- the lifecycle callback used to stop the LLM-owned document companion.

It does not own PDFium rendering, layout detection, crop policy, OCR prompts,
markdown merge, model readiness, Runtime publication, request leases, or model
service storage. Those responsibilities remain in their existing owners.

```mermaid
classDiagram
    class HLIEWorkflow {
        +stop()
    }
    class HLIELlmApplicationService {
        +handle(request, operation, method)
    }
    class HLIEDocumentApplicationService {
        +operation_for(request, body)
        +dispatch_or_route(model, operation, request, body, method)
        +stop_document_pipeline()
    }
    class DocumentInputAdapter {
        +inspect(request)
    }
    class HLIEInferenceDispatcher {
        +dispatch(model, operation, request, method)
    }
    class DocumentPipelineService {
        +handle_document_request(request)
        +handle_document_ocr_request(request, llm)
    }
    class HLIEModelServiceManager {
        +stop_and_reset_document_pipeline()
    }
    class HLIEApplicationCompositionRoot

    HLIEApplicationCompositionRoot *-- HLIEDocumentApplicationService
    HLIEWorkflow --> HLIELlmApplicationService : route delegation
    HLIELlmApplicationService --> HLIEDocumentApplicationService : PDF-aware dispatch
    HLIEWorkflow --> HLIEDocumentApplicationService : stop companion
    HLIEDocumentApplicationService --> DocumentInputAdapter : inspect
    HLIEDocumentApplicationService ..> HLIEInferenceDispatcher : injected dispatch
    HLIEInferenceDispatcher --> DocumentPipelineService : embedded endpoint
    HLIEDocumentApplicationService ..> HLIEModelServiceManager : injected stop callback
```

## Request Sequence

```mermaid
sequenceDiagram
    participant W as Workflow facade
    participant A as DocumentApplicationService
    participant I as DocumentInputAdapter
    participant D as InferenceDispatcher
    participant E as Embedded Endpoint
    participant P as DocumentPipelineService

    W->>A: dispatch_or_route(model, fallback, request, body)
    A->>I: inspect(request)
    alt no PDF
        A->>D: dispatch(model, fallback, request)
    else PDF preprocess
        A-->>A: select DocumentPreprocess
        A->>D: dispatch(model, DocumentPreprocess, request)
        D->>E: generation-bound endpoint dispatch
        E->>P: handle_document_request(request)
    else PDF OCR
        A-->>A: select DocumentOcr
        A->>D: dispatch(model, DocumentOcr, request)
        D->>E: generation-bound endpoint dispatch
        E->>P: handle_document_ocr_request(request, llm)
    end
```

The Dispatcher still creates and guards the inference session before the
pipeline runs. Streaming/response teardown therefore preserves the P1-6 rule:
response first, endpoint/Runtime next, request lease last.

## Logging And Verification

PDF route observations are emitted through a callback owned by the composition
root and logged at `DEBUG`, because successful per-request routing is not a
lifecycle transition. Pipeline failures keep their existing `ERROR` logs in
`DocumentPipelineService`; no polling or repeated state log was added.

The architecture gate
`hlie.architecture.document-application-service-dependencies` prevents the
service from depending on Workflow, Controller, ServiceManager, concrete model
services, ProcessManager, DeviceService, Singleton, TCIM, GGML, or PDFium. It
also prevents PDF operation selection and document companion stop ownership
from returning to Workflow.

Focused tests cover non-document fallthrough, OCR selection, boolish preprocess
selection, dispatch delegation, and the lifecycle callback. Windows Release
and the SDK-free test slice both compile this production service.
