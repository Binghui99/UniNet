# UniNet : A Unified Multi-granular Traffic Modeling Framework for Network Security

The whole code coming soon

# UniNet 🕸️🎛️  
*A Unified Multi-Granular Traffic-Modelling Framework for Network Security*

[![paper](https://img.shields.io/badge/paper-IEEE%20TCCN%2025-blue)](https://doi.org/10.1109/TCCN.2025.3585170) 
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE) 
![python](https://img.shields.io/badge/python-3.9%2B-blue) 
![pytorch](https://img.shields.io/badge/pytorch-2.0+-red)

> **UniNet** combines **T-Matrix** (a session + flow + packet representation) with **T-Attent** (a lightweight hierarchical transformer) and a set of plug-and-play heads, delivering state-of-the-art results on four classical security tasks:  
> 1. Unsupervised anomaly detection  
> 2. (Binary / multi-class) attack identification  
> 3. IoT device fingerprinting  
> 4. Encrypted website fingerprinting

---

## ✨ Key features
| What | Why it matters |
|------|----------------|
| **Multi-granular input** | Session, flow *and* packet features are encoded together, capturing local and global context.|
| **Lightweight transformer** | Only 2 encoder blocks, 10 heads, 15 k parameters – fast (< 1 μs inference on RTX 4080) yet accurate. |
| **Unified heads** | Masked-Feature-Prediction (MFP), auto-encoder, and MLP heads cover unsupervised, semi- and fully-supervised scenarios. |
| **Reproducible results** | Scripts to reproduce all benchmarks on CIC-IDS-2018, UNSW-IoT-2018, and DoQ+QUIC-2024. |
| **Explainable tokens** | Every traffic feature is a token, enabling fine-grained attention heat-maps.|

---


## Framework

<p align="center">
  <img src="./materials/Idea_of_architecture.png" width="100%" alt="UniNet framework diagram">
</p>

**Input**: raw network packet capture (`.pcap`)  
**Output**: task-specific prediction (score, label or embedding)

UniNet’s pipeline is a single **encode-once, predict-any** flow:

1. **Feature extractor** – converts the pcap into session-, flow- and packet-level features (the **T-Matrix**).  
2. **Tokenizer** – maps every feature to a discrete token ID and positional encoding.  
3. **T-Attent transformer** – two lightweight encoder blocks merge local (packet) and global (session) context.  
4. **Head layer** – one of the plug-and-play heads below turns the shared embedding into the final output.

| # | Down-stream task | Head name | Typical metric (default) | Entry-point script |
|---|------------------|-----------|--------------------------|--------------------|
| 1 | **Unsupervised anomaly detection** | Masked-Feature-Prediction (MFP) auto-encoder | AUROC / AUPRC | `scripts/train_anomaly.py` |
| 2 | **Attack identification** (binary / multi-class) | MLP classifier | Accuracy / macro-F1 | `scripts/train_attack.py` |
| 3 | **IoT device fingerprinting** | Contrastive classification head | Macro-F1 | `scripts/train_iot.py` |
| 4 | **Encrypted website fingerprinting** | Triplet-ranking head | Top-1 / Top-5 accuracy | `scripts/train_ewf.py` |

> *Need a new task?* Add a custom head in `uninet/models/heads/`, declare it in your YAML, and the rest of the stack stays unchanged.







## Citation

If you find our work useful in your research, please consider citing our paper by:

```bibtex
@ARTICLE{11063437,
  author={Wu, Binghui and Divakaran, Dinil Mon and Gurusamy, Mohan},
  journal={IEEE Transactions on Cognitive Communications and Networking}, 
  title={UniNet: A Unified Multi-Granular Traffic Modeling Framework for Network Security}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCCN.2025.3585170}}
```
