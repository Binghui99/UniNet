import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class ModelTest(unittest.TestCase):
    def test_backbone_and_all_heads(self):
        from uninet.model import (
            ClassificationHead,
            EmbeddingAutoencoder,
            MaskedFeatureHead,
            TAttent,
        )

        backbone = TAttent(max_tokens=32, dropout=0.0)
        tokens = torch.randint(0, 1040, (2, 32))
        tokens[:, -4:] = 1041
        segments = torch.randint(0, 3, (2, 32))

        hidden = backbone.encode(tokens, segments)
        embedding = backbone(tokens, segments)
        self.assertEqual(tuple(hidden.shape), (2, 32, 10))
        self.assertEqual(tuple(embedding.shape), (2, 10))
        self.assertEqual(tuple(MaskedFeatureHead(10)(hidden).shape), (2, 32, 1042))
        self.assertEqual(tuple(ClassificationHead(10, 5)(embedding).shape), (2, 5))
        self.assertEqual(tuple(EmbeddingAutoencoder(10, 4).anomaly_score(embedding).shape), (2,))


if __name__ == "__main__":
    unittest.main()

