# AdaSpeech Inference Guide

This guide explains how to properly run inference using the AdaSpeech model.

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:Cocii/AdaSpeech.git
   cd AdaSpeech
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Model and Vocoder Checkpoints

Before running inference, you need:

1. AdaSpeech model checkpoint
2. Vocoder checkpoint (default BigVGAN)
3. Configuration files:
   - `preprocess.yaml`
   - `model.yaml` 
   - `train.yaml`

## Running Inference

### Basic Command Structure
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
--language_id <LANG_ID> \
--speaker_id <SPEAKER_ID> \
--reference_audio <REF_AUDIO_PATH> \
--text "$(cat test.txt)" \
-p <PREPROCESS_CONFIG> \
-m <MODEL_CONFIG> \
-t <TRAIN_CONFIG> \
--restore_step <CHECKPOINT_STEP> \
--vocoder_checkpoint <VOCODER_PATH> \
--vocoder_config <VOCODER_CONFIG>
```

### Key Parameters

- `language_id`: Language identifier (0 for English, 1 for Chinese)
- `speaker_id`: Target speaker identifier  
- `reference_audio`: Path to reference audio file for speaker embedding
- `text`: Input text for synthesis (can be read from file)
- `restore_step`: Checkpoint step number to load
- `vocoder_checkpoint`: Path to vocoder model weights
- `vocoder_config`: Path to vocoder configuration file