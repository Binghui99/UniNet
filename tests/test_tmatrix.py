import tempfile
import unittest
from pathlib import Path

from uninet.synthetic import benign_browse_packets, write_pcap
from uninet.tmatrix import ExtractionConfig, TMatrixExtractor, flatten_matrix


class TMatrixTest(unittest.TestCase):
    def test_builds_paper_granularities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traffic.pcap"
            write_pcap(path, benign_browse_packets())
            matrices, stats = TMatrixExtractor().from_pcap(path, label=0)
        self.assertEqual(stats.decoded, 8)
        self.assertEqual(len(matrices), 1)
        matrix = matrices[0]
        self.assertEqual(matrix.session.context_ip, "10.0.0.10")
        self.assertEqual(matrix.session_features["flow_count"], 2.0)
        self.assertEqual(len(matrix.flow_features), 2)
        self.assertEqual([len(rows) for rows in matrix.packet_features], [2, 6])
        segments = {feature.segment for feature in flatten_matrix(matrix)}
        self.assertEqual(segments, {0, 1, 2})

    def test_key_ip_filters_unrelated_packets(self):
        config = ExtractionConfig(context_mode="key-ip", key_ip="10.0.0.99")
        matrices = TMatrixExtractor(config).from_packets([])
        self.assertEqual(matrices, [])


if __name__ == "__main__":
    unittest.main()
