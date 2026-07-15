# Data preparation

## Included synthetic fixtures

Run `uninet smoke --output-dir data/smoke` to deterministically create:

| Capture | Label | Intended software check |
|---|---:|---|
| `benign_browse.pcap` | 0 | DNS plus a bidirectional TLS-like TCP exchange |
| `dns_burst.pcap` | 1 | Repeated outbound DNS metadata |
| `syn_scan.pcap` | 1 | SYN packets across multiple service ports |

These traces contain no real payload or user data. Labels are illustrative and are
not evidence that this tiny dataset can train or evaluate a security detector.

`data/smoke/tasks/` additionally contains four deterministic tokenized datasets,
one for each Python task entry. Regenerate them with
`python scripts/generate_task_smoke_data.py`. They validate loaders, splits and
one-epoch CI training only; their metrics have no scientific meaning.

## External research datasets

The paper evaluates CIC-IDS-2018, UNSW-IoT-2018, and DoQ-2024. They are not bundled:
their size, hosting, and licenses are controlled by their providers. Obtain them from
the official sources, preserve the original archives, and record checksums locally.

Recommended layout (ignored by Git):

```text
data/external/<dataset>/raw/
data/processed/<dataset>/<preprocessing-version>/
```

Never commit sensitive PCAPs. When sharing a processed artifact, publish the exact
split manifest, T-Matrix configuration, fitted training tokenizer, class mapping,
source checksums, package version, and random seed.
