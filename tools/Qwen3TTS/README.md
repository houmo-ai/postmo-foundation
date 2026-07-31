# Qwen3-TTS GGUF conversion

`convert_qwen3_tts_to_gguf.py` packages a Houmo HMM deployment, its FP16
embedding state dictionaries, and the official Qwen tokenizer/configuration
files into one GGUF bundle.

Like the CosyVoice3 GGUF bundle, embeddings are stored as flat I8 byte
resources containing contiguous FP16 data. Their dtype and logical shape are
recorded in `file_info`. The three resources are `quant_embedding.bin`,
`text_embedding.bin`, and `quant_embedding_code_predictor.bin`; the last one
has logical shape `[num_code_groups - 1, vocab_size, hidden_size]`.

Both the `CustomVoice` and `Base` variants are supported for the configured
0.6B and 1.7B model sizes. Model dimensions come from the selected official
configuration and are checked against every embedding. The converter rejects
a mixed deployment, such as 1.7B HMM files with a 0.6B configuration.

Base deployments additionally require `speaker_encoder.hmm` and
`speech_tokenizer_encoder.hmm`. They are stored under the stable GGUF tensor
names `qwen3_tts.speaker_encoder.hmm` and
`qwen3_tts.speech_tokenizer_encoder.hmm`. Base bundles use
`qwen3_tts.variant=base`; CustomVoice bundles retain
`qwen3_tts.variant=custom_voice`.

Example:

```powershell
python tools/Qwen3TTS/convert_qwen3_tts_to_gguf.py `
  --input E:\models\Qwen3-TTS\hmm_xh2_qwen3-tts-1.7b-customvoice_w8a16_256_2k_b1_1chip_2cores_v1.4.0 `
  --config-dir E:\models\Qwen3-TTS\Qwen3-TTS-12Hz-1.7B-CustomVoice-config `
  --output E:\models\Qwen3-TTS\qwen3-tts-1.7b-customvoice.gguf
```

Base example:

```powershell
python tools/Qwen3TTS/convert_qwen3_tts_to_gguf.py `
  --input E:\models\Qwen3-TTS\hmm_xh2_qwen3-tts-0.6b-base_w8a16h1_sefp_256_2k_b1_1chip_2cores_v1.4.0 `
  --config-dir E:\models\Qwen3-TTS\Qwen3-TTS-12Hz-0.6B-Base-config `
  --output E:\models\Qwen3-TTS\HiModel_xh2_qwen3-tts-0.6b-base-w8a16h1-sefp-256-2k-b1-1chip-2cores-v1.4.0.gguf
```

If PTQ produced a deployment `meta.json`, pass it with `--deployment-meta`.
This preserves Stateful Decoder timing and lookahead values in GGUF. Older
deployments without that file use the current 12 Hz defaults: 1920 samples per
codec frame and four initial lookahead frames.

The configuration directory must contain the official tokenizer files and the
JSON files below. The original `model.safetensors` is not used:

- `config.json`
- `generation_config.json`
- `preprocessor_config.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`
- `speech_tokenizer/config.json`
- `speech_tokenizer/configuration.json`
- `speech_tokenizer/preprocessor_config.json`

The bundle deliberately retains `speech_tokenizer.hmm` for compatibility with
hi.py and future voice-cloning workflows. The native C++ CustomVoice path does
not consume reference audio, so it does not load that module eagerly. This
avoids reserving roughly 519 MB of device memory for an inactive graph while
preserving the original portable bundle layout.

## Dynamic loading

The C++ service uses the explicit dynamic service type `qwen3_tts`. The
generic legacy type `tts` remains an alias for CosyVoice so existing
voice-clone clients are not redirected to Qwen3-TTS. For example:

```json
{
  "model_type": "qwen3_tts",
  "model_path": "E:\\models\\Qwen3-TTS\\qwen3-tts-1.7b-customvoice-w8a16-256-2k-b1-1chip-2cores-v1.4.0.gguf",
  "devices": "0"
}
```

Send the object to `POST /models/load` (or `/hlie/v1/start`). Prefer unloading
by model identity with `POST /models/unload` (or `/hlie/v1/unload_model`):

```json
{"model_id":"qwen3-tts-1.7b-customvoice-w8a16-256-2k-b1-1chip-2cores-v1.4.0"}
```

The legacy `{"model_type":"qwen3_tts"}` form remains supported when the
caller does not have a model id.

## Native inference debugging

After the model reaches `READY`, the native Qwen3-TTS pipeline exposes these
temporary model-specific verification routes:

- `POST /hlie/v1/qwen3_tts/prefill_debug`
- `POST /hlie/v1/qwen3_tts/codec_frames_debug`
- `POST /hlie/v1/qwen3_tts/reference_debug` (Base reference audio frontend)
- `POST /hlie/v1/qwen3_tts/stateful_decoder_debug`
- `POST /hlie/v1/qwen3_tts/speech_debug`

The debug speech and codec-frame routes use deterministic greedy generation for
Python/C++ graph comparison. Base requests must provide both `prompt_text` (the
reference transcript) and `prompt_wav_path` (a local WAV file); the aliases
`reference_text` and `reference_audio_path` are also accepted. The native
frontend resamples the reference to 24 kHz, runs the speech-tokenizer encoder
and speaker encoder, and primes the Stateful Decoder with the reference codec
before decoding clone audio. References longer than the static tokenizer
encoder window are truncated in the same way as `demo_base.py`; the speaker
mel frontend independently trims to the encoder's configured frame count. See
`tests/http/qwen3_tts_client.http` for Base
reference, clone, load, and unload requests.

The synchronous pipeline is also available from the OpenAI-compatible
`POST /v1/audio/speech` API. Select it with `"model":"qwen3_tts"` or the
loaded model key. `input` and `voice` follow the OpenAI request surface. For
Base voice clone, select the Base model and add `prompt_text` and
`prompt_wav_path`; `voice` is ignored because Base has no fixed speaker map.
`language` and `max_frames` are optional HLIE extensions. The current native
backend supports `response_format=wav` and `speed=1.0`. Streaming behavior and
additional encoded formats will be added separately.

Requests without a `model` preserve the historical CosyVoice behavior,
including `prompt_text` and `prompt_wav_path`. Qwen3-TTS requests must select
the Qwen3-TTS model explicitly.

The production route samples Talker with temperature 0.9 and top-k 50, plus
repetition penalty, minimum EOS length, and reserved-token suppression. Talker
cannot be fully greedy because some valid inputs, especially mixed-language
text, can remain on a repetitive path without producing EOS. CustomVoice keeps
CodePredictor greedy to reduce request-to-request timbre variation. Base follows
the official `subtalker_dosample` generation setting stored in its GGUF bundle.

Sampling uses a request-level seed. The default `seed=1024` prevents prior
requests from advancing the random sequence used by later requests and greatly
improves voice and pace stability. Pass another non-negative 32-bit seed for a
different stable rendition, or `seed=-1` to request a fresh random seed. The
effective seed is returned in the `X-Qwen3-TTS-Seed` response header. Small
run-to-run differences can still occur when near-tied NPU logits change order.

Production speech requests require the Talker to emit EOS. If `max_frames` is
too small, the API returns an error instead of presenting truncated audio as a
successful WAV response. The model registry also holds an active-request lease
for the complete synchronous inference, so a concurrent unload cannot destroy
the service while it is in use.
