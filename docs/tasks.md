# Four task pipelines

All maintained task entry points are ordinary Python scripts under `tasks/`. They
consume tokenized T-Matrix JSON or JSONL and save `model.pt` plus `metrics.json`.
Run `--dry-run` first to validate labels, token lengths and deterministic splits
without importing PyTorch.

## Task 1: unsupervised anomaly detection

```bash
python tasks/task1_anomaly_detection.py \
  --dataset train.json \
  --benign-label 0 \
  --mask-ratio 0.4 \
  --threshold-percentile 95
```

The backbone first learns benign context through masked feature prediction. A
two-layer symmetric autoencoder then reconstructs benign T-Attent embeddings.
The threshold is selected from held-out benign reconstruction scores; anomalous
samples are never used to fit it. Output metrics include ROC-AUC, TPR and FPR.

## Task 2: attack identification

```bash
python tasks/task2_attack_identification.py --dataset attacks.json
```

This supports binary or multiclass labels with the paper's two-layer classification
head. It reports accuracy, macro precision, macro recall and macro F1. Split by
capture day, host, or campaign before T-Matrix conversion when those groups could
otherwise leak across random splits.

## Task 3: IoT device identification

```bash
python tasks/task3_iot_device_identification.py --dataset devices.json
```

This is the complete fourth repo component that was missing from the original code
snapshot. Use endpoint sessions and labels derived from an external inventory. Raw
MAC/IP identity must not be included as a model feature. Report per-device results
in addition to macro metrics for imbalanced deployments.

## Task 4: encrypted website fingerprinting

```bash
python tasks/task4_website_fingerprinting.py \
  --dataset websites.json \
  --unknown-label unknown
```

Without `--unknown-label`, this is closed-world multiclass classification. With an
unknown class, it additionally reports monitored-vs-unknown TPR and FPR. Unknown
sites must remain disjoint between training and test; do not tune the threshold or
class mapping on the final test set.

## Configuration and overrides

Each task has a JSON config under `configs/tasks/`:

```bash
python tasks/task3_iot_device_identification.py \
  --config configs/tasks/task3_iot.json \
  --epochs 5
```

Command-line values override config defaults. Random seeds, split sizes, model
dimensions, optimizer settings, device selection and output directories are exposed
through `--help`.
