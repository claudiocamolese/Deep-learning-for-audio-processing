# AnCoGen: Attribute-based Neural Codec for Speech Generation

An interactive educational framework for learning and using the AnCoGen (Attribute-based Neural Codec Generation) model for speech synthesis and manipulation.

## Overview

AnCoGen is a neural model that enables **analysis, representation, and resynthesis** of speech through decomposition into high-level attributes. This Jupyter notebook provides a comprehensive educational interface with interactive quizzes and demonstration functions to explore the model's capabilities.

## Architecture

### Speech Representations (Speech Attributes)

The AnCoGen model decomposes the speech signal into **7 independent high-level attributes**:

| Attribute | Code | Description | Range/Type |
|-----------|------|-------------|------------|
| **Audio** | A0 | Raw acoustic content | Discrete tokens (HuBERT) |
| **Content** | A1 | Semantic/phonetic representation | Quantized HuBERT embeddings |
| **Pitch** | A2 | Fundamental frequency (f0) | Hz (continuous → discretized) |
| **Loudness** | A3 | RMS signal level | dB (continuous → discretized) |
| **SNR** | A4 | Signal-to-Noise Ratio | 0-80 dB |
| **C50** | A5 | Clarity Index (reverberation) | Acoustic clarity index |
| **Identity** | A6 | Speaker identity | Speaker embedding (Resemblyzer) |

### Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING                             │
│  Audio Waveform → Resampling (16kHz) → Normalization            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ HuBERT   │  │ CREPE    │  │ RMS      │  │Resemblyzer│       │
│  │(Content) │  │ (Pitch)  │  │(Loudness)│  │(Identity) │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TOKENIZATION                                │
│  • K-means clustering for continuous attributes                 │
│  • Conversion of continuous features → discrete tokens          │
│  • Vocabulary for each attribute (e.g., 1024 tokens for Content)│
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ANCOGEN MODEL                                 │
│  • Transformer-based architecture                               │
│  • Coupled masking strategy (training)                          │
│  • All-or-nothing masking (inference)                           │
│  • Learning inter/intra-attribute dependencies                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SYNTHESIS (HiFi-GAN)                          │
│  Audio Tokens → Mel-Spectrogram → Waveform (16kHz)             │
└─────────────────────────────────────────────────────────────────┘
```

### Masking Strategy

The model uses two complementary masking strategies:

**1. Coupled Masking (Training)**
- Simultaneous masking across multiple correlated attributes
- Enables learning of **inter-attribute** and **intra-attribute** dependencies
- Example: masking pitch and content together to capture prosody-content relationships

**2. All-or-Nothing Masking (Inference)**
- Entire sequence of an attribute is either masked or fully visible
- Used during generation/manipulation
- Example: completely mask audio (A0) while keeping other attributes for reconstruction

## Main Features

### 1. **Speech Analysis and Synthesis**
```python
# Audio preprocessing
audio = preprocess("input.wav")

# Attribute extraction
attributes = analysis(audio, apply_max=True)

# Signal reconstruction
synthesis(from_attributes=attributes)
```

**What it does:**
- Decomposes an audio file into 7 attributes
- Reconstructs the speech signal from discrete tokens
- Enables inspection and manipulation of extracted attributes

---

### 2. **Speech Enhancement (Denoising)**
```python
snr_control(attributes=attributes, target=30)
```

**What it does:**
- Modifies the SNR of attribute A4
- Resynthesizes the audio with the new noise level
- **Application:** background noise removal, call quality improvement

**How it works:**
1. Extracts the original SNR from the noisy signal
2. Replaces the SNR attribute with the target value (e.g., 30 dB)
3. Resynthesizes while keeping content, pitch, and identity unchanged

---

### 3. **Voice Conversion**
```python
voice_conversion(source_path="speaker_A.wav", 
                 target_path="speaker_B.wav")
```

**What it does:**
- **Speaker identity swapping** (attribute A6)
- Maintains the linguistic content of the source
- Applies the vocal characteristics of the target

**How it works:**
1. Extracts attributes from both audio files
2. Replaces `source_attributes[6]` with `target_attributes[6]`
3. Resynthesizes with: **Content from A** + **Voice from B**

**Applications:**
- Personalized Text-to-Speech
- Dubbing and post-production
- Privacy (voice anonymization)

---

### 4. **Interactive Educational Quizzes**

The notebook includes 3 quiz sections on:
- Speech attribute representation
- Tokenization and quantization
- Masking strategies

---


## Pre-trained Models

The notebook automatically downloads from HuggingFace Hub:

1. **AnCoGen Model** (`samir-sadok/AnCoGen-LibriSpeech`)
   - Checkpoint `2024-4-8.zip`
   - Trained on LibriSpeech dataset

2. **HiFi-GAN Vocoder** (`samir-sadok/AnCoGen-HiFiGAN`)
   - Neural waveform generator
   - Converts mel-spectrogram → audio

## Notebook Structure

1. **Theoretical Section**
   - Quiz 1: Speech Representation (SA attributes)
   - Quiz 2: Token Representations (quantization)
   - Quiz 3: Masking Strategy

2. **Environment Setup**
   - Dependencies installation
   - Pre-trained models download
   - Loading weights and configurations

3. **Core Functions**
   - `preprocess()`: audio preparation
   - `analysis()`: attribute extraction
   - `synthesis()`: audio reconstruction
   - `post_process()`: token → waveform conversion

4. **Demonstration Applications**
   - Basic synthesis (analysis-synthesis loop)
   - Speech enhancement with SNR control
   - Voice conversion between speakers

## Technical Details

### Attribute Tokenization

| Attribute | Quantization Method | Vocabulary |
|-----------|---------------------|------------|
| Content (A1) | K-means on HuBERT embeddings | 1024 tokens |
| Pitch (A2) | Logarithmic binning | ~512 tokens |
| Loudness (A3) | Linear/log binning | ~256 tokens |
| SNR (A4) | Binning (0-80 dB) | ~160 tokens |
| C50 (A5) | Binning | ~256 tokens |
| Identity (A6) | K-means on speaker embeddings | 512 tokens |

### Post-Processing

The `post_process()` function handles:
- **Token decoding** → logits → indices → mel-spectrogram
- **Neural vocoding** with HiFi-GAN
- **3 outputs generation**:
  - `original_signal`: reconstructed original audio
  - `masked_signal`: audio with masking applied
  - `reconstructed_signal`: model-generated audio

## Use Cases
- **Voice Assistants**: voice personalization
- **Accessibility**: intelligibility improvement
- **Broadcasting**: automatic post-production
- **Gaming**: dynamic voice acting


## 🔗 References

- **HuggingFace Hub**: [samir-sadok/AnCoGen-LibriSpeech](https://huggingface.co/samir-sadok/AnCoGen-LibriSpeech)
- **Dataset**: LibriSpeech (English read speech corpus)
- **Vocoder**: HiFi-GAN neural vocoder
