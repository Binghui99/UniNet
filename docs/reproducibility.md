# Reproducibility guide

## Levels of reproduction

1. **Software smoke test** - included and dependency-free. It verifies PCAP parsing,
   multi-granular extraction, tokenization, masking, and fixed-length serialization.
2. **Method reproduction** - use your authorized PCAPs, freeze split manifests, fit
   quantiles only on training data, and train one of the four Python task scripts.
3. **Paper benchmark reproduction** - requires the paper datasets and their original
   preprocessing/splits. The retained legacy notebooks document research history but
   are not the maintained execution path.

## Minimum experiment record

Record the Git commit, Python and package versions, hardware, dataset checksum,
train/validation/test IDs, config file, tokenizer JSON, random seeds, optimizer and
schedule, stopping rule, chosen checkpoint, and every reported metric definition.

For low-FPR security evaluation, report TPR at fixed FPR values in addition to broad
metrics. Fit thresholds on validation data; never choose them on the test set.

## Deterministic verification

```bash
python -m pip install -e .
uninet smoke --output-dir data/smoke
python -m unittest discover -s tests -v
uninet inspect data/smoke/tmatrix-smoke.json
```

Optional model import check:

```bash
python -m pip install -e '.[model]'
python -c 'from uninet.model import TAttent; print(TAttent())'
```
