"""Shared, reproducible training runners for the four UniNet paper tasks."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .tokenizer import MASK_TOKEN_ID, PAD_TOKEN_ID, VOCAB_SIZE


TASKS = {
    "anomaly": {
        "title": "Task 1 - Unsupervised anomaly detection",
        "mode": "anomaly",
    },
    "attack": {
        "title": "Task 2 - Attack identification",
        "mode": "classification",
    },
    "iot": {
        "title": "Task 3 - IoT device identification",
        "mode": "classification",
    },
    "website": {
        "title": "Task 4 - Encrypted website fingerprinting",
        "mode": "classification",
    },
}


def label_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def read_tokenized_dataset(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        metadata: Dict[str, Any] = {}
        samples = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") == "metadata":
                metadata = row.get("metadata", {})
            elif row.get("record_type") == "sample":
                samples.append(row["sample"])
            else:
                raise ValueError(f"invalid JSONL record_type on line {line_number}")
        payload = metadata
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        samples = payload.get("samples", [])
    if payload.get("format") not in {None, "uninet-tmatrix-tokenized"}:
        raise ValueError("task input must be tokenized T-Matrix JSON/JSONL")
    validate_samples(samples)
    return payload, samples


def validate_samples(samples: Sequence[Dict[str, Any]]) -> None:
    if not samples:
        raise ValueError("dataset contains no samples")
    expected_length = None
    required = ("input", "segment_label", "sequence_label")
    for index, sample in enumerate(samples):
        missing = [key for key in required if key not in sample]
        if missing:
            raise ValueError(f"sample {index} is missing {', '.join(missing)}")
        length = len(sample["input"])
        if length == 0 or len(sample["segment_label"]) != length:
            raise ValueError(f"sample {index} has inconsistent or empty token arrays")
        if expected_length is None:
            expected_length = length
        if length != expected_length:
            raise ValueError("all samples must use the same max_tokens")
        if any(not isinstance(value, int) or not 0 <= value < VOCAB_SIZE for value in sample["input"]):
            raise ValueError(f"sample {index} contains an invalid token ID")
        if any(value not in (0, 1, 2) for value in sample["segment_label"]):
            raise ValueError(f"sample {index} contains an invalid segment label")


def summarize(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter(label_key(sample["sequence_label"]) for sample in samples)
    return {
        "samples": len(samples),
        "sequence_length": len(samples[0]["input"]),
        "labels": dict(sorted(counts.items())),
    }


def stratified_split(
    samples: Sequence[Dict[str, Any]], val_ratio: float, test_ratio: float, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[label_key(sample["sequence_label"])].append(index)
    rng = random.Random(seed)
    train, validation, test = [], [], []
    for indices in groups.values():
        rng.shuffle(indices)
        count = len(indices)
        test_count = max(1, round(count * test_ratio)) if count >= 2 and test_ratio > 0 else 0
        val_count = max(1, round(count * val_ratio)) if count >= 3 and val_ratio > 0 else 0
        while test_count + val_count >= count:
            if val_count:
                val_count -= 1
            elif test_count:
                test_count -= 1
        test.extend(indices[:test_count])
        validation.extend(indices[test_count : test_count + val_count])
        train.extend(indices[test_count + val_count :])
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test


def _classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], classes: int) -> Dict[str, float]:
    accuracy = sum(a == b for a, b in zip(y_true, y_pred)) / max(1, len(y_true))
    precisions, recalls, f1s = [], [], []
    for target in range(classes):
        tp = sum(a == target and b == target for a, b in zip(y_true, y_pred))
        fp = sum(a != target and b == target for a, b in zip(y_true, y_pred))
        fn = sum(a == target and b != target for a, b in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": accuracy,
        "macro_precision": sum(precisions) / classes,
        "macro_recall": sum(recalls) / classes,
        "macro_f1": sum(f1s) / classes,
    }


def _device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _tensor_batch(torch: Any, samples: Sequence[Dict[str, Any]], indices: Sequence[int], labels: Dict[str, int]):
    tokens = torch.tensor([samples[index]["input"] for index in indices], dtype=torch.long)
    segments = torch.tensor([samples[index]["segment_label"] for index in indices], dtype=torch.long)
    targets = torch.tensor(
        [labels[label_key(samples[index]["sequence_label"])] for index in indices],
        dtype=torch.long,
    )
    return torch.utils.data.TensorDataset(tokens, segments, targets)


def _loader(torch: Any, dataset: Any, batch_size: int, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def train_classifier(args: argparse.Namespace, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("install training dependencies with: pip install -e '.[model]'") from exc
    from .model import ClassificationHead, TAttent

    torch.manual_seed(args.seed)
    label_values = sorted({label_key(sample["sequence_label"]) for sample in samples})
    if len(label_values) < 2:
        raise ValueError("classification requires at least two labels")
    labels = {value: index for index, value in enumerate(label_values)}
    train_ids, val_ids, test_ids = stratified_split(samples, args.val_ratio, args.test_ratio, args.seed)
    if not train_ids:
        raise ValueError("training split is empty")
    device = _device(torch, args.device)
    model_args = {
        "max_tokens": len(samples[0]["input"]),
        "embedding_dim": args.embedding_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "feedforward_dim": args.feedforward_dim,
        "dropout": args.dropout,
    }
    backbone = TAttent(**model_args).to(device)
    head = ClassificationHead(args.embedding_dim, len(labels)).to(device)
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(head.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    train_loader = _loader(
        torch,
        _tensor_batch(torch, samples, train_ids, labels),
        args.batch_size,
        True,
        args.seed,
    )
    history = []
    for epoch in range(args.epochs):
        backbone.train()
        head.train()
        total_loss, total = 0.0, 0
        for tokens, segments, targets in train_loader:
            tokens, segments, targets = tokens.to(device), segments.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(head(backbone(tokens, segments)), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
            total += len(targets)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / max(1, total)})

    evaluation_ids = test_ids or val_ids or train_ids
    evaluation = _tensor_batch(torch, samples, evaluation_ids, labels)
    evaluation_loader = _loader(torch, evaluation, args.batch_size, False, args.seed)
    y_true, y_pred = [], []
    backbone.eval()
    head.eval()
    with torch.no_grad():
        for tokens, segments, targets in evaluation_loader:
            logits = head(backbone(tokens.to(device), segments.to(device)))
            y_true.extend(targets.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = _classification_metrics(y_true, y_pred, len(labels))
    if args.task == "website" and args.unknown_label is not None:
        unknown_key = label_key(_parse_json_value(args.unknown_label))
        if unknown_key not in labels:
            raise ValueError(f"unknown label {args.unknown_label!r} is not present in the dataset")
        unknown_id = labels[unknown_key]
        monitored_true = [value != unknown_id for value in y_true]
        monitored_pred = [value != unknown_id for value in y_pred]
        tp = sum(a and b for a, b in zip(monitored_true, monitored_pred))
        fn = sum(a and not b for a, b in zip(monitored_true, monitored_pred))
        fp = sum(not a and b for a, b in zip(monitored_true, monitored_pred))
        tn = sum(not a and not b for a, b in zip(monitored_true, monitored_pred))
        metrics["open_world_tpr"] = tp / (tp + fn) if tp + fn else 0.0
        metrics["open_world_fpr"] = fp / (fp + tn) if fp + tn else 0.0
    result = {
        "task": args.task,
        "mode": "classification",
        "device": str(device),
        "split": {"train": len(train_ids), "validation": len(val_ids), "test": len(test_ids)},
        "label_mapping": {value: index for value, index in labels.items()},
        "metrics": metrics,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": backbone.state_dict(),
            "head": head.state_dict(),
            "model_config": model_args,
            "label_mapping": labels,
            "task": args.task,
        },
        args.output_dir / "model.pt",
    )
    _write_result(args.output_dir, result)
    return result


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate threshold from an empty validation set")
    position = (len(ordered) - 1) * percentile / 100.0
    left, right = math.floor(position), math.ceil(position)
    fraction = position - left
    return ordered[left] * (1 - fraction) + ordered[right] * fraction


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return float("nan")
    wins = sum(p > n for p in positives for n in negatives)
    ties = sum(p == n for p in positives for n in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def train_anomaly(args: argparse.Namespace, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("install training dependencies with: pip install -e '.[model]'") from exc
    from .model import EmbeddingAutoencoder, MaskedFeatureHead, TAttent

    benign_key = label_key(_parse_json_value(args.benign_label))
    benign = [index for index, sample in enumerate(samples) if label_key(sample["sequence_label"]) == benign_key]
    anomalous = [index for index in range(len(samples)) if index not in benign]
    if len(benign) < 3 or not anomalous:
        raise ValueError("anomaly training requires at least 3 benign and 1 anomalous sample")
    rng = random.Random(args.seed)
    rng.shuffle(benign)
    threshold_count = max(1, round(len(benign) * args.val_ratio))
    test_count = max(1, round(len(benign) * args.test_ratio))
    while threshold_count + test_count >= len(benign):
        if test_count > 1:
            test_count -= 1
        elif threshold_count > 1:
            threshold_count -= 1
        else:
            break
    threshold_ids = benign[:threshold_count]
    benign_test_ids = benign[threshold_count : threshold_count + test_count]
    train_ids = benign[threshold_count + test_count :]
    evaluation_ids = benign_test_ids + anomalous
    torch.manual_seed(args.seed)
    device = _device(torch, args.device)
    model_args = {
        "max_tokens": len(samples[0]["input"]),
        "embedding_dim": args.embedding_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "feedforward_dim": args.feedforward_dim,
        "dropout": args.dropout,
    }
    backbone = TAttent(**model_args).to(device)
    mfp = MaskedFeatureHead(args.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(mfp.parameters()), lr=args.learning_rate
    )
    labels = {benign_key: 0}
    dataset = _tensor_batch(torch, samples, train_ids, labels)
    train_loader = _loader(torch, dataset, args.batch_size, True, args.seed)
    mfp_history = []
    for epoch in range(args.pretrain_epochs):
        backbone.train()
        mfp.train()
        total_loss, steps = 0.0, 0
        for tokens, segments, _ in train_loader:
            tokens, segments = tokens.to(device), segments.to(device)
            valid = tokens.ne(PAD_TOKEN_ID)
            mask = torch.rand(tokens.shape, device=device).lt(args.mask_ratio) & valid
            if not mask.any():
                mask.view(-1)[valid.view(-1).nonzero()[0]] = True
            masked_tokens = tokens.clone()
            masked_tokens[mask] = MASK_TOKEN_ID
            optimizer.zero_grad()
            logits = mfp(backbone.encode(masked_tokens, segments))
            loss = nn.functional.cross_entropy(logits[mask], tokens[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        mfp_history.append({"epoch": epoch + 1, "loss": total_loss / max(1, steps)})

    def embeddings(indices: Sequence[int]):
        loader = _loader(
            torch,
            _tensor_batch(
                torch,
                samples,
                indices,
                {
                    benign_key: 0,
                    **{
                        label_key(samples[index]["sequence_label"]): 1
                        for index in anomalous
                    },
                },
            ),
            args.batch_size,
            False,
            args.seed,
        )
        rows = []
        backbone.eval()
        with torch.no_grad():
            for tokens, segments, _ in loader:
                rows.append(backbone(tokens.to(device), segments.to(device)).cpu())
        return torch.cat(rows)

    train_embeddings = embeddings(train_ids).to(device)
    autoencoder = EmbeddingAutoencoder(args.embedding_dim, args.bottleneck_dim).to(device)
    ae_optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=args.learning_rate)
    ae_history = []
    for epoch in range(args.autoencoder_epochs):
        autoencoder.train()
        ae_optimizer.zero_grad()
        reconstruction = autoencoder(train_embeddings)
        loss = nn.functional.mse_loss(reconstruction, train_embeddings)
        loss.backward()
        ae_optimizer.step()
        ae_history.append({"epoch": epoch + 1, "loss": loss.item()})

    autoencoder.eval()
    validation_embeddings = embeddings(threshold_ids).to(device)
    with torch.no_grad():
        benign_scores = autoencoder.anomaly_score(validation_embeddings).cpu().tolist()
        evaluation_scores = autoencoder.anomaly_score(embeddings(evaluation_ids).to(device)).cpu().tolist()
    threshold = _percentile(benign_scores, args.threshold_percentile)
    truth = [0] * len(benign_test_ids) + [1] * len(anomalous)
    predicted = [int(score > threshold) for score in evaluation_scores]
    tp = sum(a == 1 and b == 1 for a, b in zip(truth, predicted))
    fp = sum(a == 0 and b == 1 for a, b in zip(truth, predicted))
    tn = sum(a == 0 and b == 0 for a, b in zip(truth, predicted))
    fn = sum(a == 1 and b == 0 for a, b in zip(truth, predicted))
    metrics = {
        "roc_auc": _auc(truth, evaluation_scores),
        "threshold": threshold,
        "tpr": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }
    result = {
        "task": args.task,
        "mode": "mfp-plus-autoencoder",
        "device": str(device),
        "split": {
            "benign_train": len(train_ids),
            "benign_threshold": len(threshold_ids),
            "benign_test": len(benign_test_ids),
            "anomalous_test": len(anomalous),
        },
        "metrics": metrics,
        "mfp_history": mfp_history,
        "autoencoder_history": ae_history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backbone": backbone.state_dict(),
            "mfp_head": mfp.state_dict(),
            "autoencoder": autoencoder.state_dict(),
            "model_config": model_args,
            "benign_label": benign_key,
            "threshold": threshold,
        },
        args.output_dir / "model.pt",
    )
    _write_result(args.output_dir, result)
    return result


def _parse_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _write_result(output_dir: Path, result: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def build_parser(task: str) -> argparse.ArgumentParser:
    spec = TASKS[task]
    parser = argparse.ArgumentParser(description=spec["title"])
    parser.add_argument("--config", type=Path, help="JSON file providing CLI defaults")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output") / task)
    parser.add_argument("--dry-run", action="store_true", help="Validate data and print the split without PyTorch")
    if spec["mode"] == "classification":
        parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embedding-dim", type=int, default=10)
    parser.add_argument("--num-heads", type=int, default=10)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=40)
    parser.add_argument("--dropout", type=float, default=0.1)
    if spec["mode"] == "anomaly":
        parser.add_argument("--benign-label", default="0")
        parser.add_argument("--pretrain-epochs", type=int, default=20)
        parser.add_argument("--autoencoder-epochs", type=int, default=30)
        parser.add_argument("--mask-ratio", type=float, default=0.4)
        parser.add_argument("--bottleneck-dim", type=int, default=4)
        parser.add_argument("--threshold-percentile", type=float, default=95.0)
    if task == "website":
        parser.add_argument("--unknown-label", help="Optional open-world unknown label for reporting")
    return parser


def main_for_task(task: str, argv: Optional[Sequence[str]] = None) -> int:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path)
    known, _ = probe.parse_known_args(argv)
    parser = build_parser(task)
    if known.config:
        try:
            config = json.loads(known.config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"{task}: error: cannot read config: {exc}", file=sys.stderr)
            return 2
        valid = {action.dest for action in parser._actions}
        unknown = sorted(set(config) - valid)
        if unknown:
            print(f"{task}: error: unknown config keys: {', '.join(unknown)}", file=sys.stderr)
            return 2
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    args.task = task
    if args.dataset is None:
        parser.error("--dataset is required (directly or through --config)")
    args.dataset = Path(args.dataset)
    args.output_dir = Path(args.output_dir)
    try:
        if not 0 <= args.val_ratio < 1 or not 0 <= args.test_ratio < 1:
            raise ValueError("val_ratio and test_ratio must be in [0, 1)")
        if args.val_ratio + args.test_ratio >= 1:
            raise ValueError("val_ratio + test_ratio must be less than 1")
        if args.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if args.embedding_dim <= 0 or args.num_heads <= 0:
            raise ValueError("embedding_dim and num_heads must be positive")
        if args.embedding_dim % args.num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if TASKS[task]["mode"] == "anomaly":
            if not 0 < args.mask_ratio <= 1:
                raise ValueError("mask_ratio must be in (0, 1]")
            if not 0 < args.threshold_percentile < 100:
                raise ValueError("threshold_percentile must be in (0, 100)")
        _, samples = read_tokenized_dataset(args.dataset)
        summary = summarize(samples)
        if TASKS[task]["mode"] == "classification":
            train_ids, val_ids, test_ids = stratified_split(
                samples, args.val_ratio, args.test_ratio, args.seed
            )
            summary["split"] = {
                "train": len(train_ids),
                "validation": len(val_ids),
                "test": len(test_ids),
            }
        else:
            benign = label_key(_parse_json_value(args.benign_label))
            summary["benign_samples"] = sum(
                label_key(sample["sequence_label"]) == benign for sample in samples
            )
        if args.dry_run:
            print(json.dumps({"task": task, **summary}, indent=2))
            return 0
        result = (
            train_anomaly(args, samples)
            if TASKS[task]["mode"] == "anomaly"
            else train_classifier(args, samples)
        )
        print(json.dumps(result["metrics"], indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"{task}: error: {exc}", file=sys.stderr)
        return 2
