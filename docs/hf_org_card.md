# trunk-reporter

Open-source tools for P25 trunked radio transcription and monitoring.

We build speech recognition models and infrastructure for public safety radio -- law enforcement, fire, EMS, public works, transit, and more. Everything from real-time transcription to edge deployment on a Raspberry Pi.

## Models

### IMBE-ASR -- Speech Recognition from Vocoder Parameters

Skip audio reconstruction entirely. ASR directly from P25 IMBE codec parameters (4.4 kbps). **1.9% WER** on LibriSpeech-IMBE with beam search.

| Model | Params | WER | Use Case |
|-------|--------|-----|----------|
| [imbe-asr-large-1024d](https://huggingface.co/trunk-reporter/imbe-asr-large-1024d) | 290M | 1.9% (beam) | Best accuracy, server/desktop |
| [imbe-asr-base-512d](https://huggingface.co/trunk-reporter/imbe-asr-base-512d) | 48.6M | 18.9% (greedy) | Edge deployment (Pi 5, 15x RT) |
| [imbe-asr-base-512d-p25](https://huggingface.co/trunk-reporter/imbe-asr-base-512d-p25) | 48.6M | -- | Real P25 dispatch fine-tuned |

All models available in SafeTensors, ONNX fp32, and ONNX int8/uint8 for edge deployment. The 290M model runs at 3.6x real-time on a Raspberry Pi 5 via a 70KB C binary.

[View collection](https://huggingface.co/collections/trunk-reporter/imbe-asr-speech-recognition-from-vocoder-parameters-69c0a4f68ef670b5bf68449d) | [GitHub](https://github.com/trunk-reporter/imbe-asr)

### Qwen3-ASR P25 -- Fine-tuned Audio ASR

| Model | Base | Use Case |
|-------|------|----------|
| [qwen3-asr-p25-0.6B](https://huggingface.co/trunk-reporter/qwen3-asr-p25-0.6B) | Qwen3-ASR 0.6B | P25 audio transcription, pseudo-labeling |

Fine-tuned on P25 dispatch audio for conventional audio-based ASR. Used in our pseudo-labeling pipeline alongside Whisper large-v3.

[GitHub](https://github.com/trunk-reporter/qwen3-asr-server)

### Coming Soon

- **IMBENet** -- Neural IMBE vocoder for high-quality P25 audio reconstruction

## Links

- [GitHub](https://github.com/trunk-reporter)
- [trunk-recorder](https://github.com/trunk-reporter/trunk-recorder) (fork with IMBE tap support)
