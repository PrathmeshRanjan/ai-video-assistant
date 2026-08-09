# Speech-to-Text Rest API

> Process short audio files synchronously with immediate response. Instant transcription and translation for quick audio processing with multiple format support.

<h3>
  Synchronous Processing
</h3>

<p>
  Process short audio files with immediate response. Best for quick
  transcriptions and testing with a maximum duration of 30 seconds.
</p>

## Saaras v3: State-of-the-Art Speech Recognition (Recommended)

Saaras v3 is our latest state-of-the-art speech recognition model with flexible output formats. It supports multiple modes for different use cases: transcribe, translate, verbatim, transliterate, and codemix.

**Recommended for new integrations.** Saaras v3 offers improved accuracy and flexible output modes. [Learn more about Saaras v3](/api/getting-started/models/saaras).

### Output Modes

| Mode                   | Description                                     |
| ---------------------- | ----------------------------------------------- |
| `transcribe` (default) | Standard transcription in the original language |
| `translate`            | Translates speech to English                    |
| `verbatim`             | Exact word-for-word transcription               |
| `translit`             | Romanization to Latin script                    |
| `codemix`              | Code-mixed text output                          |

### Code Examples for Saaras v3

#### Python

```python
from sarvamai import SarvamAI

client = SarvamAI(
    api_subscription_key="YOUR_SARVAM_API_KEY",
)

# Transcribe mode (default)
response = client.speech_to_text.transcribe(
    file=open("audio.wav", "rb"),
    model="saaras:v3",
    mode="transcribe"  # or "translate", "verbatim", "translit", "codemix"
)

print(response)
```

#### JavaScript

```javascript
import {SarvamAIClient} from "sarvamai";
import fs from 'fs';

const client = new SarvamAIClient({
    apiSubscriptionKey: "YOUR_SARVAM_API_KEY"
});

const audioFile = fs.createReadStream("recording.wav");

const response = await client.speechToText.transcribe({
    file: audioFile,
    model: "saaras:v3",
    mode: "transcribe"  // or "translate", "verbatim", "translit", "codemix"
});

console.log(response);
```

#### cURL

```bash
curl -X POST https://api.sarvam.ai/speech-to-text \
  -H "api-subscription-key: YOUR_SARVAM_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F model="saaras:v3" \
  -F mode="transcribe" \
  -F file=@file.wav
```

Check out our detailed [API Reference](/api-reference/speech-to-text/transcribe)
to explore all available options.

## Preparing Your Audio

Most failed STT requests are caused by the audio itself, not the API call. Run through this checklist before uploading:

| Check              | Recommendation                                                                                                                                                                                                      |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Duration**       | The sync REST API accepts up to **30 seconds** per request. For longer files, use the [Batch API](/api/api-guides-tutorials/speech-to-text/batch-api) (up to 2 hours per file) or split the audio into ≤30s chunks. |
| **Sample rate**    | **16 kHz** is recommended. **8 kHz telephony audio** (IVR, call recordings) is fully supported — no need to upsample.                                                                                               |
| **Channels**       | Use **mono**. For stereo telephony recordings with one speaker per channel, split the channels and transcribe each separately to keep speakers separated.                                                           |
| **Format**         | WAV, MP3, AAC, FLAC, or OGG. Prefer WAV (16-bit PCM) for best accuracy.                                                                                                                                             |
| **File integrity** | Verify the file exists and is non-empty before uploading (`file.size > 0` in browsers). Pass a **file object**, not a path string — e.g. `file=open("audio.wav", "rb")` in Python.                                  |

Not sure whether to use REST, Batch, or Streaming? See the [Which API to Use](/api/api-guides-tutorials/speech-to-text/which-api-to-use) decision table for a side-by-side comparison of limits, latency, and features.

---

## Legacy Model (Deprecated Soon)

The following model will be deprecated soon. We recommend migrating to **Saaras v3** for new integrations.

### Saaras v2.5: Speech to Text Translation

Saaras v2.5 is available in the Speech-to-Text Translate endpoint for translating speech directly to English.

**Deprecation Notice:** Saaras v2.5 will be deprecated soon. Use [Saaras v3](/api/getting-started/models/saaras) with `mode="translate"` instead — see the [code examples above](#code-examples-for-saaras-v3).

## API Response Format

### Speech to Text Transcription Response

| Field           | Type   | Description                                                                                       |
| --------------- | ------ | ------------------------------------------------------------------------------------------------- |
| `request_id`    | string | Unique identifier for the request                                                                 |
| `transcript`    | string | The transcribed text from the audio file                                                          |
| `language_code` | string | BCP-47 language code of detected language (e.g., `hi-IN`). Returns `null` if no language detected |

```json
{
  "request_id": "20241115_12345678-1234-5678-1234-567812345678",
  "transcript": "नमस्ते, आप कैसे हैं?",
  "language_code": "hi-IN"
}
```

### Speech to Text Translation Response

| Field           | Type   | Description                                 |
| --------------- | ------ | ------------------------------------------- |
| `request_id`    | string | Unique identifier for the request           |
| `transcript`    | string | Translated text in English                  |
| `language_code` | string | BCP-47 code of the detected source language |

**Supported source languages:** `hi-IN`, `bn-IN`, `kn-IN`, `ml-IN`, `mr-IN`, `od-IN`, `pa-IN`, `ta-IN`, `te-IN`, `gu-IN`, `en-IN`

```json
{
  "request_id": "20241115_12345678-1234-5678-1234-567812345678",
  "transcript": "Hello, how are you?",
  "language_code": "hi-IN"
}
```

## Error Responses

All errors return a JSON object with an `error` field (`message`, `code`, `request_id`). The full error-code table, retry guidance, and SDK exception reference live on the central [Errors & Troubleshooting](/api/getting-started/errors-troubleshooting) page.

Errors specific to this endpoint:

| HTTP Status | Error Code                   | When This Happens                                              | What To Do                                                                                                                                 |
| ----------- | ---------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `422`       | `unprocessable_entity_error` | Invalid audio format, file too large, or audio over 30 seconds | Use supported formats (WAV, MP3, AAC, FLAC, OGG); for longer audio use the [Batch API](/api/api-guides-tutorials/speech-to-text/batch-api) |

#### Error Handling Code Example

```python
from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError

client = SarvamAI(api_subscription_key="YOUR_SARVAM_API_KEY")

try:
    response = client.speech_to_text.transcribe(
        file=open("audio.wav", "rb"),
        model="saaras:v3",
        mode="transcribe"
    )
    print(response.transcript)
except ApiError as e:
    if e.status_code == 400:
        print(f"Bad request: {e.body}")
    elif e.status_code == 403:
        print("Invalid API key. Check your credentials.")
    elif e.status_code == 429:
        print("Rate limit exceeded. Wait and retry.")
    elif e.status_code == 503:
        print("Service overloaded. Retry with backoff.")
    else:
        print(f"Error {e.status_code}: {e.body}")
```

## Next Steps

#### Get API Key

Sign up and get your API key from the
[dashboard](https://dashboard.sarvam.ai).

#### Test Integration

Try the API with sample audio files.

#### Go Live

Deploy your integration and monitor usage.

Need help? Contact us on [discord](https://discord.com/invite/5rAsykttcs) for
guidance.