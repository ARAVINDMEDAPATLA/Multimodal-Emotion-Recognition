# Multimodal Emotion Recognition

A PyTorch-based system for recognizing emotions from speech, text, and multimodal (speech + text) inputs using TESS (Toronto Emotional Speech Set). The project compares speech-only (Bi-LSTM + Attention), text-only (fine-tuned BERT), and cross-modal attention fusion architectures.

## Key Results

| Model | Test Accuracy | Macro F1 |
|-------|:------------:|:--------:|
| Speech-only | **100.0%** | 1.00 |
| Text-only | 14.4% | 0.05 |
| Multimodal (Fusion) | **100.0%** | 1.00 |

**Key insight:** TESS encodes emotion entirely in acoustic features (prosody, pitch, energy), not lexical content — the transcripts are single neutral words identical across all emotions.

## Architecture Overview
Speech: MFCCs (40 + Delta + Delta-Delta) → Bi-LSTM (2x128) + Self-Attention → [256]
Text: BERT WordPiece → Fine-tuned BERT (last 6 layers) → [CLS] + Mean Pooling → [1536]
Fusion: Cross-Modal Attention (Speech→Text + Text→Speech, 4 heads) → LayerNorm → [256]
Output: 2-layer FC (256→7) + Softmax

### Components

| Block | Speech | Text |
|-------|--------|------|
| Preprocessing | librosa (22.05kHz, trim silence, 3s fixed length) | Lowercase, remove special chars, BERT tokenization |
| Features | MFCCs (40) + Delta + Delta-Delta (T×120) | BERT last_hidden_state (tokens×768) |
| Modelling | Bi-LSTM + Self-Attention (output 256) | BERT fine-tuned + [CLS]/Mean Pool (output 1536) |
| Fusion | Cross-modal attention (4 heads each direction) → 256-d → Classifier | |

## Dataset

**TESS (Toronto Emotional Speech Set)**
- 5,600 samples, 7 emotions (balanced, 800 per emotion)
- Emotions: angry, disgust, fear, happy, neutral, pleasant surprise, sad
- Two female actors speaking semantically neutral single words
- Split: 80% train / 20% test (stratified)

## Project Structure
project/
├── models/
│ ├── speech_pipeline/
| | |── train.py
| | |── test.py
├ ├── text_pipeline/
| | |── train.py
| | |── test.py
│ └── fusion_pipeline/
| |── train.py
| |── test.py
│
├── Results/
| |── All 3 model variants accuracy tables
| └── plots
├── README.md
└── requirements.txt


## Setup

### Requirements

pip install torch==2.12.0 librosa transformers scikit-learn matplotlib seaborn numpy pandas
Dataset
Download TESS from Kaggle and place in ./data/TESS/

Usage
Train Models

# Train speech-only model
python train.py --modality speech --epochs 50 --batch_size 32

# Train text-only model
python train.py --modality text --epochs 10 --batch_size 16

# Train multimodal fusion model
python train.py --modality fusion --epochs 50 --batch_size 32
Evaluate

python evaluate.py --modality speech --checkpoint checkpoints/speech_best.pt
python evaluate.py --modality text --checkpoint checkpoints/text_best.pt
python evaluate.py --modality fusion --checkpoint checkpoints/fusion_best.pt


**Visualizations**

python visualize.py --modality speech --type tsne  # t-SNE of learned representations
python visualize.py --modality speech --type confusion
python visualize.py --modality all --type comparison  # Model comparison bar chart
Results in Detail
Speech-Only (Bi-LSTM + Attention)
Perfect classification on all 7 emotions:

text
              precision    recall  f1-score   support
       angry       1.00      1.00      1.00       160
     disgust       1.00      1.00      1.00       160
        fear       1.00      1.00      1.00       160
       happy       1.00      1.00      1.00       160
     neutral       1.00      1.00      1.00       160
          ps       1.00      1.00      1.00       160
         sad       1.00      1.00      1.00       160
    accuracy                           1.00      1120
Text-Only (Fine-tuned BERT)
Model collapsed to predicting "disgust" for most samples (14.4% accuracy ≈ random):

text
              precision    recall  f1-score   support
       angry       0.00      0.00      0.00       160
     disgust       0.15      0.95      0.25       160
        ...        ...       ...       ...       ...
    accuracy                           0.14      1120
Multimodal (Cross-Modal Attention)
Perfect classification (100% accuracy, macro F1 = 1.00)
**
t-SNE Visualizations**
Block	Separability	Interpretation
Temporal (Speech Bi-LSTM)	✅ Clear clusters	Acoustic features are highly discriminative
Contextual (BERT)	❌ Heavy overlap	Neutral words carry no emotional signal
Fusion	✅ Clear clusters	Speech dominates; attention learns to rely on audio
Error Analysis: Text Model Failures
Input Text	True Emotion	Predicted
"the speaker said home out loud"	fear	disgust
"the speaker said mess out loud"	pleasant surprise	disgust
"the speaker said mill out loud"	angry	disgust
"the speaker said tough out loud"	angry	disgust
"the speaker said page out loud"	happy	disgust
Why text fails: The same neutral word (e.g., "back") appears across all 7 emotions. BERT cannot infer emotion from context-free single words.

**When Does Fusion Help?**
Not needed for TESS (speech alone = 100% accuracy). However, the cross-modal attention architecture is designed for:

Ambiguous speech (sarcasm, irony — tone conflicts with words)

Degraded modalities (noisy audio, unclear speech)

Subtle emotions requiring both prosodic and semantic cues

Emotionally meaningful text datasets (IEMOCAP, MELD)

**Technical Specifications**
Component	Detail
Framework	PyTorch 2.12.0
Hardware	CPU (Intel)
Random seed	42
Speech features	40 MFCC + Delta + Delta-Delta
LSTM	2 layers, 128 hidden dims, bidirectional
BERT	bert-base-uncased, last 6 layers fine-tuned
Cross-attention	4 heads per direction, 256-d fusion space
Dropout	0.3 (classifier only)
Training time	Speech: ~50 min, Text: ~3 hr, Fusion: ~5 hr

**References**

TESS Dataset: Toronto Emotional Speech Set(Kaggle)

BERT: Devlin et al. (2019) - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

Cross-modal attention: Based on multimodal fusion literature for affective computing

