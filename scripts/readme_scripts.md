# Scripts Directory

Auxiliary tool scripts for training monitoring, data inspection, generation/inference, and configuration.

## Training Monitoring

### check_training.py
Real-time training status viewer - displays current Loss and Learning Rate.

```bash
python scripts/check_training.py
```

### analyze_loss.py
Loss trend analyzer - compares Loss across stages (initial, early, mid, recent) to help diagnose training health.

```bash
python scripts/analyze_loss.py
```

### diagnose_dynamics.py
Exports training dynamics: learning rate changes, loss curve, gradient norm, and Top-K logits distribution.

```bash
python scripts/diagnose_dynamics.py
```

---

## Data Inspection

### check_data.py
Three-phase checker: raw data quality → tokenizer verification → DataLoader test.

```bash
python scripts/check_data.py
```

### generate_delete_candidates.py
Scans data for problematic pairs (too short, too long, duplicates, etc.) and generates a cleanup candidate list.

```bash
python scripts/generate_delete_candidates.py
```

---

## Generation / Inference

### generate_samples.py
Greedy decoding - picks the highest-probability token at each step. Fast but low diversity.

```bash
python scripts/generate_samples.py --checkpoint checkpoints/best_model.pt
```

### generate_beam.py
Beam Search - keeps multiple candidate paths, producing higher-quality translations.

```bash
python scripts/generate_beam.py --checkpoint checkpoints/best_model.pt --beam 5
```

### generate_sampling.py
Sampling-based generation (Temperature + Top-K). Controls randomness; useful for diversity analysis.

```bash
python scripts/generate_sampling.py --checkpoint checkpoints/best_model.pt --temperature 0.8 --top_k 50
```

### analyze_predictions.py
Analyzes prediction quality metrics: repetition rate, top token/bigram distributions, etc.

```bash
python scripts/analyze_predictions.py --file predictions.txt
```

---

## Configuration

### print_config.py
Prints all current hyperparameters (epochs, batch_size, d_model, etc.).

```bash
python scripts/print_config.py
```

### train_tokenizer_run.py
SentencePiece BPE tokenizer training entry point with customizable parameters.

```bash
python scripts/train_tokenizer_run.py --zh data/wmt_processed/train.zh --en data/wmt_processed/train.en
```

---

## File Overview

| File | Size | Purpose |
|------|------|---------|
| check_training.py | 1.4 KB | Training status monitor |
| analyze_loss.py | 2.6 KB | Loss trend analysis |
| diagnose_dynamics.py | 3.7 KB | Training dynamics diagnostics |
| check_data.py | 6.1 KB | Data quality check |
| generate_delete_candidates.py | 2.0 KB | Data cleanup candidates |
| generate_samples.py | 3.4 KB | Greedy decoding |
| generate_beam.py | 5.8 KB | Beam Search generation |
| generate_sampling.py | 5.4 KB | Sampling-based generation |
| analyze_predictions.py | 1.8 KB | Prediction analysis |
| print_config.py | 2.0 KB | Hyperparameter display |
| train_tokenizer_run.py | 0.9 KB | Tokenizer training |
