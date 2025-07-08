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
> 4. Encrypted website fingerprinting :contentReference[oaicite:0]{index=0}

---

## ✨ Key features
| What | Why it matters |
|------|----------------|
| **Multi-granular input** | Session, flow *and* packet features are encoded together, capturing local and global context. :contentReference[oaicite:1]{index=1} |
| **Lightweight transformer** | Only 2 encoder blocks, 10 heads, 15 k parameters – fast (< 1 μs inference on RTX 4080) yet accurate. :contentReference[oaicite:2]{index=2} |
| **Unified heads** | Masked-Feature-Prediction (MFP), auto-encoder, and MLP heads cover unsupervised, semi- and fully-supervised scenarios. :contentReference[oaicite:3]{index=3} |
| **Reproducible results** | Scripts to reproduce all benchmarks on CIC-IDS-2018, UNSW-IoT-2018, and DoQ+QUIC-2024. :contentReference[oaicite:4]{index=4} |
| **Explainable tokens** | Every traffic feature is a token, enabling fine-grained attention heat-maps. :contentReference[oaicite:5]{index=5} |

---

## 🗄️ Repository layout


## Framework

<img src="./materials/Idea_of_architecture.png" style="width:2000px;height:350px"/>

`Input`: pcap of a traffic `Output`: Task specific output 

There are four tasks 






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
