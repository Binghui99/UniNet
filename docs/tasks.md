# UniNet task recipes

The shared T-Attent backbone produces an embedding; task behavior comes from the
head and the sample granularity. These recipes describe the paper's orchestration,
not downloadable benchmark splits.

## 1. Unsupervised anomaly detection

Create session samples, pretrain T-Attent on benign traffic with masked feature
prediction, remove the MFP head, and train the symmetric embedding autoencoder on
benign embeddings. Select the reconstruction-error percentile threshold on benign
validation data. Report ROC-AUC and TPR at operationally low FPR values.

## 2. Attack identification

Create one sample per bidirectional flow (flow features followed by its packets) and
use the two-layer classification head with cross-entropy. Freeze a class mapping and
split by capture/day or host where possible to avoid near-duplicate leakage. Report
macro-F1 alongside accuracy and low-FPR detection behavior.

## 3. IoT device identification

Use MAC/device metadata only to assign labels before anonymization; do not feed raw
identifiers to the model. Construct endpoint sessions and preserve incomplete flows
created by time-window boundaries. Evaluate macro metrics and per-device results,
especially for minority devices.

## 4. Encrypted website fingerprinting

Do not inspect decrypted content. Combine aggregate session behavior with packet
metadata such as sizes, directions, and IATs. In open-world evaluation, keep unknown
sites disjoint between training and test and report TPR at fixed FPR. Never tune the
decision threshold on the final test set.

The maintained model heads live in `src/uninet/model.py`. Dataset-specific splits and
labels remain the researcher's responsibility because the original datasets have
separate distribution terms.
