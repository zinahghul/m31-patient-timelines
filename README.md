# M31 Patient Timelines

A lightweight **Transformer-based longitudinal EHR model** for predicting newly diagnosed future clinical conditions from patient timelines.

## Approach

Patient histories are converted into chronological clinical-event sequences with explicit temporal-gap tokens.

**EHRTransformer:**

* 128-dimensional embeddings
* 4 attention heads
* 3 Transformer layers
* 512-token maximum sequence length
* 40-condition multi-label prediction

Timelines are truncated at an anchor date **five years before the patient's final recorded encounter** to prevent data leakage.

The model uses `conditions`, `encounters`, `allergies`, and `careplans` data with a custom **209-token vocabulary** and seven categorical temporal intervals.

## Training

The model is trained from scratch using:

* AdamW (`1e-4` learning rate)
* Focal Loss with BCE-with-logits
* Dynamic positive-class weighting
* Batch size: 32
* Early stopping based on validation macro-AUROC
* Automatic mixed precision and memory optimization

No external datasets or pretrained weights were used.

## Validation Results

Test labels are withheld for independent scoring. The following results are from the held-out validation cohort at the optimal early-stopping checkpoint (Epoch 20).

| Metric      | Validation |
| ----------- | ---------: |
| Macro AUROC | **0.6642** |
| mAP         | **0.1344** |
| Macro F1    | **0.1380** |
| Brier Score | **0.1953** |

## Repository Structure

```text
m31-patient-timelines/
├── src/
│   ├── build_labels.py
│   ├── build_sequences.py
│   ├── build_test_sequences.py
│   ├── build_vocab.py
│   ├── dataset.py
│   ├── filter_timeline.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
├── predictions.csv
├── upload_model.py
└── README.md
```

### Pipeline

* `filter_timeline.py` — leak-free timeline filtering
* `build_labels.py` — future-condition label generation
* `build_vocab.py` — clinical event vocabulary
* `build_sequences.py` — training/validation sequence construction
* `build_test_sequences.py` — test sequence construction
* `dataset.py` — PyTorch dataset and attention masks
* `model.py` — EHRTransformer architecture
* `train.py` — model training and validation
* `predict.py` — test-set probability prediction
* `upload_model.py` — model upload to Hugging Face

## Test Predictions

`predictions.csv` contains probability predictions for all test patients across the 40 target conditions using the required condition-code headers.

## Trained Model

**Hugging Face:** https://huggingface.co/zinahghulam/m31-patient-timelines

## Experiment Tracking

Training was monitored using Weights & Biases.

**Run:** `2kcd1uk0`

## AI-Assisted Development

AI coding assistants were used for prototyping, debugging, code review, and identifying implementation issues involving label generation, attention masking, and datetime parsing.
