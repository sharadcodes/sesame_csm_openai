# CSM-1B TTS API

An OpenAI-compatible Text-to-Speech API that harnesses the power of Sesame's Conversational Speech Model (CSM-1B). This API allows you to generate high-quality speech from text using a variety of consistent voices, compatible with systems like OpenWebUI, ChatBot UI, and any platform that supports the OpenAI TTS API format.

## Features

- **OpenAI API Compatibility**: Drop-in replacement for OpenAI's TTS API
- **Multiple Voices**: Six distinct voices (alloy, echo, fable, onyx, nova, shimmer)
- **Voice Consistency**: Maintains consistent voice characteristics across multiple requests
- **Conversational Context**: Supports conversational context for improved naturalness
- **Multiple Audio Formats**: Supports MP3, OPUS, AAC, FLAC, and WAV
- **Speed Control**: Adjustable speech speed
- **CUDA Acceleration**: GPU support for faster generation

## Getting Started

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- Hugging Face account with access to `sesame/csm-1b` model

### Installation

1. Clone this repository:

```bash
git clone https://github.com/phildougherty/csm-tts-api
cd csm-tts-api
```

2. Create a `.env` file with your Hugging Face token:

```
HF_TOKEN=your_hugging_face_token_here
```

3. Build and start the container:

```bash
docker compose up -d --build
```

The server will start on port 8000. First startup may take some time as it downloads the model files.

## Hugging Face Configuration

This API requires access to the `sesame/csm-1b` model on Hugging Face:

1. Create a Hugging Face account if you don't have one: [https://huggingface.co/join](https://huggingface.co/join)
2. Accept the model license at [https://huggingface.co/sesame/csm-1b](https://huggingface.co/sesame/csm-1b)
3. Generate an access token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Use this token in your `.env` file or pass it directly when building the container:

```bash
HF_TOKEN=your_token docker compose up -d --build
```

### Required Models

The API uses the following models which are downloaded automatically:

- **CSM-1B**: The main speech generation model from Sesame
- **Mimi**: Audio codec for high-quality audio generation
- **Llama Tokenizer**: Uses the unsloth/Llama-3.2-1B tokenizer for text processing

## How the Voices Work

Unlike traditional TTS systems with pre-trained voice models, CSM-1B works differently:

- The base CSM-1B model is capable of producing a wide variety of voices but doesn't have fixed voice identities
- This API creates consistent voices by using acoustic "seed" samples for each named voice
- When you specify a voice (e.g., "alloy"), the API uses a consistent acoustic seed and speaker ID
- The most recent generated audio becomes the new reference for that voice, maintaining voice consistency
- Each voice has unique tonal qualities:
  - **alloy**: Balanced mid-tones with natural inflection
  - **echo**: Resonant with slight reverberance
  - **fable**: Brighter with higher pitch
  - **onyx**: Deep and resonant
  - **nova**: Warm and smooth
  - **shimmer**: Light and airy with higher frequencies

The voice system can be extended with your own voice samples by modifying the `voice_embeddings.py` file.

## API Usage

### Basic Usage

Generate speech with a POST request to `/v1/audio/speech`:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "csm-1b",
    "input": "Hello, this is a test of the CSM text to speech system.",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

### Available Endpoints

- `GET /v1/audio/models` - List available models
- `GET /v1/audio/voices` - List available voices
- `GET /v1/audio/speech/response-formats` - List available response formats
- `POST /v1/audio/speech` - Generate speech from text
- `POST /api/v1/audio/conversation` - Advanced endpoint for conversational speech

### Request Parameters

| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
| `model` | Model ID to use | string | "csm-1b" |
| `input` | The text to convert to speech | string | Required |
| `voice` | The voice to use | string | "alloy" |
| `response_format` | Audio format | string | "mp3" |
| `speed` | Speech speed multiplier | float | 1.0 |
| `temperature` | Sampling temperature | float | 0.9 |
| `max_audio_length_ms` | Maximum audio length in ms | integer | 90000 |

### Available Voices

- `alloy` - Balanced and natural
- `echo` - Resonant
- `fable` - Bright and higher-pitched
- `onyx` - Deep and resonant
- `nova` - Warm and smooth
- `shimmer` - Light and airy

### Response Formats

- `mp3` - MP3 audio format
- `opus` - Opus audio format
- `aac` - AAC audio format
- `flac` - FLAC audio format
- `wav` - WAV audio format

## Integration with OpenWebUI

OpenWebUI is a popular open-source UI for AI models that supports custom TTS endpoints. Here's how to integrate the CSM-1B TTS API:

1. Access your OpenWebUI settings
2. Navigate to the TTS settings section
3. Select "Custom TTS Endpoint"
4. Enter your CSM-1B TTS API URL: `http://your-server-ip:8000/v1/audio/speech`
5. Use the API Key field to add any authentication if you've configured it (not required by default)
6. Test the connection
7. Save your settings

Once configured, OpenWebUI will use your CSM-1B TTS API for all text-to-speech conversion, producing high-quality speech with the selected voice.

### OpenWebUI Voice Selection

In OpenWebUI, you can select different voices through the UI's voice selector. The voices will map directly to the CSM-1B voices:

- **Alloy** - General purpose voice with balanced tone
- **Echo** - Resonant voice with a deeper quality
- **Fable** - Upbeat, brighter voice for creative content
- **Onyx** - Deep, authoritative voice
- **Nova** - Warm, pleasant midrange voice
- **Shimmer** - Light, higher-pitched voice

## Advanced Usage

### Conversational Context

For more natural-sounding speech in a conversation, you can use the conversation endpoint:

```bash
curl -X POST http://localhost:8000/api/v1/audio/conversation \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Nice to meet you too!",
    "speaker_id": 0,
    "context": [
      {
        "speaker": 1,
        "text": "Hello, nice to meet you.",
        "audio": "BASE64_ENCODED_AUDIO"
      }
    ]
  }' \
  --output response.wav
```

This allows the model to take into account the previous utterances for more contextually appropriate speech.

### Model Parameters

For fine-grained control, you can adjust:

- `temperature` (0.0-1.0): Higher values produce more variation but may be less stable
- `topk` (1-100): Controls diversity of generated speech
- `max_audio_length_ms`: Maximum length of generated audio in milliseconds

## Troubleshooting

### API Returns 503 Service Unavailable

- Verify your Hugging Face token has access to `sesame/csm-1b`
- Check if the model downloaded successfully in the logs
- Ensure you have enough GPU memory (at least 8GB recommended)

### Audio Quality Issues

- Try different voices - some may work better for your specific text
- Adjust temperature (lower for more stable output)
- For longer texts, split into smaller chunks

### Voice Inconsistency

- The API maintains voice consistency across separate requests
- However, very long pauses between requests may result in voice drift
- For critical applications, consider using the same seed audio

## License

This project is released under the MIT License. The CSM-1B model is subject to its own license terms defined by Sesame.

## Acknowledgments

- [Sesame](https://www.sesame.com) for releasing the CSM-1B model
- This project is not affiliated with or endorsed by Sesame or OpenAI

---

Happy speech generating!
