# UniNet

**A Unified Multi-Granular Traffic Modeling Framework for Network Security**

[![Paper](https://img.shields.io/badge/IEEE_TCCN-10.1109%2FTCCN.2025.3585170-blue)](https://doi.org/10.1109/TCCN.2025.3585170)
[![Tests](https://github.com/Binghui99/UniNet/actions/workflows/test.yml/badge.svg)](https://github.com/Binghui99/UniNet/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

UniNet represents traffic at three granularities - session, bidirectional flow,
and packet - and learns a shared embedding with a lightweight hierarchical
attention model. This repository provides an installable reference implementation,
a reproducible **PCAP to T-Matrix** command, deterministic synthetic captures, and
the original experiment notebooks.

> The smoke data is synthetic and validates the software path only. It does not
> reproduce the paper's benchmark numbers. See [Reproducibility](docs/reproducibility.md).

<p align="center">
  <img src="materials/Idea_of_architecture.png" width="100%" alt="UniNet framework">
</p>

## Quick start

No third-party package is required for preprocessing standard classic PCAP files.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# Generate three labeled synthetic captures and tokenized T-Matrices.
uninet smoke --output-dir data/smoke

# Inspect the result.
uninet inspect data/smoke/tmatrix-smoke.json

# Convert your own capture. The raw file is useful for auditing features.
uninet pcap2tmatrix capture.pcap \
  --output output/tmatrix.json \
  --raw-output output/tmatrix-raw.json \
  --tokenizer-out output/tokenizer.json
```

Run the complete smoke test:

```bash
python -m unittest discover -s tests -v
```

## PCAP to T-Matrix

The conversion pipeline follows the paper's semantic structure:

1. Decode Ethernet/SLL/raw-IP packet metadata without using payload contents as features.
2. Choose a contextual endpoint (an internal IP by default).
3. Group packets into static 15-minute sessions.
4. Canonicalize both directions of a 5-tuple into one flow, expiring it after
   60 seconds of inactivity.
5. Extract session features, eight flow features, and six packet features.
6. Equal-frequency-bin continuous fields and emit fixed-length model inputs.

```text
session features (segment 2)
  flow features (segment 1)
    packet features (segment 0)
    packet features (segment 0)
  flow features (segment 1)
    ...
```

Every tokenized sample contains the five paper-defined fields:
`input`, `true_value`, `mask_index`, `segment_label`, and `sequence_label`.
Metadata such as feature names and the original sequence length is retained for
auditing. Defaults are a 2,000-token sequence and vocabulary IDs 0-1041, with
`[MASK]=1040` and `[PAD]=1041`.

For large corpora, use a `.jsonl` output path (or `--format jsonl`) so each sample
is an independent line:

```bash
uninet pcap2tmatrix captures/*.pcap -o train.jsonl \
  --format jsonl --tokenizer-out train-tokenizer.json
```

Useful capture modes:

```bash
# Observe one known endpoint only.
uninet pcap2tmatrix traffic.pcap -o host.json \
  --context-mode key-ip --key-ip 192.168.1.42

# Declare site-specific internal ranges (repeat the option as needed).
uninet pcap2tmatrix traffic.pcap -o site.json \
  --internal-network 10.20.0.0/16 --internal-network 2001:db8:1234::/48

# Pretraining-style masked-feature input.
uninet pcap2tmatrix traffic.pcap -o masked.json --mask-ratio 0.4 --seed 7

# Reuse training-set quantiles for a held-out capture (important).
uninet pcap2tmatrix test.pcap -o test.json --tokenizer-in train-tokenizer.json
```

The built-in reader supports classic PCAP with Ethernet (including VLAN), Linux
cooked v1, or raw IPv4/IPv6 packets. For PCAPNG, use Wireshark's `editcap` first:

```bash
editcap -F pcap input.pcapng output.pcap
```

Full design decisions and limitations are in [T-Matrix tool](docs/tmatrix.md).
Task-specific orchestration is summarized in [Task recipes](docs/tasks.md).

## T-Attent

Install the optional PyTorch dependency to use the model:

```bash
python -m pip install -e '.[model]'
```

```python
import torch
from uninet.model import (
    ClassificationHead,
    EmbeddingAutoencoder,
    MaskedFeatureHead,
    TAttent,
)

backbone = TAttent()  # paper defaults: d=10, 10 heads, 2 encoder layers
tokens = torch.randint(0, 1042, (4, 2000))
segments = torch.zeros_like(tokens)
embedding = backbone(tokens, segments)
logits = ClassificationHead(embedding_dim=10, num_classes=5)(embedding)

# MFP predicts vocabulary IDs at masked positions from per-token hidden states.
mfp_logits = MaskedFeatureHead(10)(backbone.encode(tokens, segments))

# Anomaly detection scores reconstruction error in the learned embedding.
scores = EmbeddingAutoencoder(10, bottleneck_dim=4).anomaly_score(embedding)
```

The package intentionally separates preprocessing from the model so researchers
can use T-Matrix with a different backbone or use T-Attent with an adapted schema.

## Repository map

```text
src/uninet/                  maintained library and CLI
  pcap.py                    streaming capture decoder
  tmatrix.py                 grouping and multi-granular features
  tokenizer.py               quantile vocabulary and five-field samples
  model.py                   optional PyTorch T-Attent reference
scripts/                     direct utility entry points
configs/                     documented paper/smoke defaults
data/smoke/                  deterministic generated fixtures
tests/                       dependency-free unit and integration tests
docs/                        data, design, and reproduction guides
Task1_Anomaly detection/     original research notebooks (legacy)
Task2_Attack_identification/ original research notebooks (legacy)
Task4-Website-fingerprinting/original research scripts (legacy)
```

The legacy artifacts are retained for provenance. They contain environment-specific
paths and are not the supported quick-start interface.

## Paper and citation

B. Wu, D. M. Divakaran, and M. Gurusamy, "UniNet: A Unified Multi-Granular
Traffic Modeling Framework for Network Security," *IEEE Transactions on Cognitive
Communications and Networking*, vol. 12, pp. 2424-2438, 2026,
doi: [10.1109/TCCN.2025.3585170](https://doi.org/10.1109/TCCN.2025.3585170).

```bibtex
@article{wu2026uninet,
  author  = {Wu, Binghui and Divakaran, Dinil Mon and Gurusamy, Mohan},
  title   = {UniNet: A Unified Multi-Granular Traffic Modeling Framework for Network Security},
  journal = {IEEE Transactions on Cognitive Communications and Networking},
  volume  = {12},
  pages   = {2424--2438},
  year    = {2026},
  doi     = {10.1109/TCCN.2025.3585170}
}
```

## Responsible use

PCAPs can contain personal data, credentials, internal addresses, and third-party
communications. Obtain authorization, minimize collection, and anonymize captures
before sharing them. Synthetic fixtures in this repository contain documentation-only
addresses and no real traffic. Please report vulnerabilities via [SECURITY.md](SECURITY.md).

## License

Code is released under the [MIT License](LICENSE). Dataset providers retain their
own terms; this license does not grant redistribution rights for external datasets.
