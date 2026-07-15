import tempfile
import unittest
from pathlib import Path

from uninet.schema import Packet
from uninet.tmatrix import TMatrixExtractor
from uninet.tokenizer import MASK_TOKEN_ID, PAD_TOKEN_ID, TMatrixTokenizer, VOCAB_SIZE


class TokenizerTest(unittest.TestCase):
    def setUp(self):
        packets = [
            Packet(1.0, "10.0.0.1", "8.8.8.8", 53000, 53, 17, 74),
            Packet(1.1, "8.8.8.8", "10.0.0.1", 53, 53000, 17, 90),
        ]
        self.matrix = TMatrixExtractor().from_packets(packets, label="benign")[0]

    def test_fixed_shape_and_five_paper_fields(self):
        tokenizer = TMatrixTokenizer().fit([self.matrix])
        sample = tokenizer.transform(self.matrix, max_tokens=64, mask_ratio=0.25, seed=4)
        for key in ("input", "true_value", "mask_index", "segment_label", "sequence_label"):
            self.assertIn(key, sample)
        self.assertEqual(len(sample["input"]), 64)
        self.assertIn(MASK_TOKEN_ID, sample["input"])
        self.assertIn(PAD_TOKEN_ID, sample["input"])
        self.assertEqual(VOCAB_SIZE, 1042)
        self.assertTrue(all(0 <= token < VOCAB_SIZE for token in sample["input"]))

    def test_tokenizer_round_trip(self):
        tokenizer = TMatrixTokenizer().fit([self.matrix])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            restored = TMatrixTokenizer.load(path)
        expected = tokenizer.transform(self.matrix, max_tokens=40)
        actual = restored.transform(self.matrix, max_tokens=40)
        self.assertEqual(expected["input"], actual["input"])


if __name__ == "__main__":
    unittest.main()
