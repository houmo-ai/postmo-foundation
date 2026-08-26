# Qwen3-TTS HTTP streaming tests

`qwen3_tts_http_stream.py` loads either a Qwen3-TTS Base or CustomVoice GGUF
and verifies that `POST /v1/audio/speech` progressively returns chunked
PCM16LE from one continuous synthesis request.

Base is the default variant. Its request omits the prompt fields and therefore
also checks the built-in `samples/zero_shot_prompt.wav` Clone profile.

Install the shared server-test dependencies and run from the repository root.

PowerShell uses a backtick for line continuation:

```powershell
python -m pip install -r tests/python/server/requirements.txt
python tests/python/server/tts/qwen3_tts_http_stream.py `
  --model-path E:/models/qwen3-tts-base.gguf `
  --devices 0
```

Bash uses a backslash instead. The backslash must be the final character on
the line, with no trailing spaces:

```bash
python -m pip install -r tests/python/server/requirements.txt
python tests/python/server/tts/qwen3_tts_http_stream.py \
  --model-path 'E:/models/qwen3-tts-base.gguf' \
  --devices 0
```

The same Bash command can be written on one line to avoid shell continuation
rules entirely.

To test CustomVoice streaming, select the variant and a fixed speaker:

```bash
python tests/python/server/tts/qwen3_tts_http_stream.py \
  --variant custom_voice \
  --model-path 'E:/models/qwen3-tts-customvoice.gguf' \
  --voice vivian \
  --devices 0
```

The CustomVoice test asserts `X-Qwen3-TTS-Variant: custom_voice` and rejects a
response that unexpectedly selects the Base Clone reference profile. Explicit
`--reference-wav` and `--reference-text` options are Base-only.

When more than one Qwen3-TTS variant is loaded, the public `qwen3_tts` alias
can resolve to the model on the primary process. The test avoids that ambiguity
by deriving the concrete model ID from the GGUF filename. It verifies the owning
worker through `/hlie/v1/status`, but always sends synthesis through the public
`--base-url` (7901 by default). This exercises primary-process routing and
streaming proxy behavior when the selected model is hosted on another port.
With `--skip-load`, provide either `--model-path` for discovery or an explicit
`--model` ID.

Use an already loaded model with `--skip-load`. Add `--unload` to unload the
model after a successful test. Raw PCM and a playable WAV are written under
`artifacts/qwen3_tts_http_stream/`.

The test fails unless at least two non-empty HTTP chunks arrive over a
measurable interval. `--min-chunks` and `--min-stream-span-ms` can make that
assertion stricter for a particular device.
