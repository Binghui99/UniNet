"""T-Attent reference implementation (requires the optional ``model`` extra)."""

from __future__ import annotations

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - exercised only without optional extra
    raise ImportError("Install model dependencies with: pip install -e '.[model]'") from exc

from .tokenizer import PAD_TOKEN_ID, VOCAB_SIZE


class TAttent(nn.Module):
    """Lightweight hierarchical transformer encoder from the UniNet design.

    Token, granularity-segment, and learnable positional embeddings are summed
    before two Transformer encoder blocks. Mean pooling excludes PAD tokens.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        max_tokens: int = 2000,
        embedding_dim: int = 10,
        num_heads: int = 10,
        num_layers: int = 2,
        feedforward_dim: int = 40,
        dropout: float = 0.1,
    ):
        super().__init__()
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_TOKEN_ID)
        self.segment_embedding = nn.Embedding(3, embedding_dim)
        self.position_embedding = nn.Embedding(max_tokens, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)

    def encode(self, tokens: Tensor, segments: Tensor) -> Tensor:
        if tokens.shape != segments.shape:
            raise ValueError("tokens and segments must have identical [batch, sequence] shapes")
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        hidden = (
            self.token_embedding(tokens)
            + self.segment_embedding(segments)
            + self.position_embedding(positions)
        )
        pad_mask = tokens.eq(PAD_TOKEN_ID)
        return self.norm(self.encoder(hidden, src_key_padding_mask=pad_mask))

    def forward(self, tokens: Tensor, segments: Tensor) -> Tensor:
        hidden = self.encode(tokens, segments)
        pad_mask = tokens.eq(PAD_TOKEN_ID)
        valid = (~pad_mask).unsqueeze(-1)
        return (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        return self.layers(embedding)


class MaskedFeatureHead(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, vocab_size)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.projection(hidden)


class EmbeddingAutoencoder(nn.Module):
    """Two-layer symmetric head used for embedding-space anomaly detection."""

    def __init__(self, embedding_dim: int = 10, bottleneck_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        return self.decoder(self.encoder(embedding))

    def anomaly_score(self, embedding: Tensor) -> Tensor:
        return (embedding - self(embedding)).pow(2).mean(dim=-1)
