"""Paper-aligned discrete tokenizer for T-Matrix features."""

from __future__ import annotations

import bisect
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union

from .schema import Feature, TMatrix
from .tmatrix import flatten_matrix

VOCAB_SIZE = 1042
MASK_TOKEN_ID = 1040
PAD_TOKEN_ID = 1041
MAX_DATA_TOKEN_ID = 1039


class TMatrixTokenizer:
    """Equal-frequency tokenizer with the 1,042-token vocabulary from the paper.

    Quantiles are learned independently per named feature type. Fitted boundaries
    are serializable, so train/validation/test sets can share exactly one mapping.
    """

    def __init__(self, bins: int = 1040):
        if not 2 <= bins <= 1040:
            raise ValueError("bins must be between 2 and 1040")
        self.bins = bins
        self.boundaries: Dict[str, List[float]] = {}

    def fit(self, matrices: Iterable[TMatrix]) -> "TMatrixTokenizer":
        values: Dict[str, List[float]] = {}
        for matrix in matrices:
            for feature in flatten_matrix(matrix):
                if feature.kind == "continuous" and math.isfinite(feature.value):
                    base_name = feature.name.split(".")[-1]
                    values.setdefault(base_name, []).append(float(feature.value))
        self.boundaries = {
            name: _quantile_boundaries(feature_values, self.bins)
            for name, feature_values in values.items()
        }
        return self

    def encode_feature(self, feature: Feature) -> int:
        if feature.kind == "categorical":
            return max(0, min(MAX_DATA_TOKEN_ID, int(feature.value)))
        name = feature.name.split(".")[-1]
        return min(MAX_DATA_TOKEN_ID, bisect.bisect_right(self.boundaries.get(name, []), feature.value))

    def transform(
        self,
        matrix: TMatrix,
        max_tokens: int = 2000,
        mask_ratio: float = 0.0,
        seed: int = 0,
    ) -> Dict[str, object]:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be in [0, 1]")
        features = flatten_matrix(matrix)[:max_tokens]
        token_ids = [self.encode_feature(feature) for feature in features]
        segments = [feature.segment for feature in features]
        feature_names = [feature.name for feature in features]
        real_length = len(token_ids)
        padding = max_tokens - real_length
        token_ids.extend([PAD_TOKEN_ID] * padding)
        segments.extend([0] * padding)

        candidate_count = int(real_length * mask_ratio)
        rng = random.Random(seed)
        masked = sorted(rng.sample(range(real_length), candidate_count)) if candidate_count else []
        true_values = [0] * max_tokens
        mask_index = [0] * max_tokens
        for index in masked:
            true_values[index] = token_ids[index]
            mask_index[index] = 1
            token_ids[index] = MASK_TOKEN_ID
        return {
            "input": token_ids,
            "true_value": true_values,
            "mask_index": mask_index,
            "segment_label": segments,
            "sequence_label": matrix.session.label,
            "metadata": {
                "session_id": matrix.session.session_id,
                "context_ip": matrix.session.context_ip,
                "source": matrix.source,
                "real_length": real_length,
                "truncated": len(flatten_matrix(matrix)) > max_tokens,
                "feature_name": feature_names,
            },
        }

    def fit_transform(
        self, matrices: Sequence[TMatrix], max_tokens: int = 2000, mask_ratio: float = 0.0, seed: int = 0
    ) -> List[Dict[str, object]]:
        self.fit(matrices)
        return [
            self.transform(matrix, max_tokens=max_tokens, mask_ratio=mask_ratio, seed=seed + index)
            for index, matrix in enumerate(matrices)
        ]

    def to_dict(self) -> Dict[str, object]:
        return {"vocab_size": VOCAB_SIZE, "bins": self.bins, "boundaries": self.boundaries}

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TMatrixTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls(bins=int(payload["bins"]))
        tokenizer.boundaries = {
            name: [float(value) for value in values]
            for name, values in payload["boundaries"].items()
        }
        return tokenizer


def _quantile_boundaries(values: Sequence[float], bins: int) -> List[float]:
    ordered = sorted(values)
    if len(ordered) < 2 or ordered[0] == ordered[-1]:
        return []
    # Never manufacture more empirical bins than there are observations. This
    # keeps tiny smoke datasets small and gives each learned bin real support.
    effective_bins = min(bins, len(ordered))
    boundaries = []
    for index in range(1, effective_bins):
        position = index * (len(ordered) - 1) / effective_bins
        left = int(math.floor(position))
        right = int(math.ceil(position))
        fraction = position - left
        value = ordered[left] * (1.0 - fraction) + ordered[right] * fraction
        if not boundaries or value > boundaries[-1]:
            boundaries.append(value)
    return boundaries
