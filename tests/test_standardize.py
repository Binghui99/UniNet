import json
import tempfile
import unittest
from pathlib import Path

from uninet.standardize import main


class StandardizeTest(unittest.TestCase):
    def test_pcap_auto_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pcap.json"
            code = main([
                "data/smoke/benign_browse.pcap",
                "-o",
                str(output),
                "--max-tokens",
                "128",
            ])
            payload = json.loads(output.read_text())
        self.assertEqual(code, 0)
        self.assertEqual(payload["inputs"][0]["kind"], "pcap")
        self.assertEqual(len(payload["samples"]), 1)

    def test_packet_and_flow_tables_have_distinct_granularity(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            packet_output = directory / "packet-raw.json"
            flow_output = directory / "flow-raw.json"
            self.assertEqual(
                main([
                    "data/examples/packets.csv",
                    "-o",
                    str(packet_output),
                    "--representation",
                    "raw",
                ]),
                0,
            )
            self.assertEqual(
                main([
                    "data/examples/flows.csv",
                    "-o",
                    str(flow_output),
                    "--representation",
                    "raw",
                ]),
                0,
            )
            packet = json.loads(packet_output.read_text())["samples"][0]
            flow = json.loads(flow_output.read_text())["samples"][0]
        self.assertTrue(any(packet["packet_features"]))
        self.assertEqual(flow["packet_features"], [[], []])
        self.assertEqual(flow["session_features"]["flow_count"], 2.0)

    def test_raw_tmatrix_round_trip_to_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw = directory / "raw.json"
            tokens = directory / "tokens.json"
            main(["data/examples/packets.csv", "-o", str(raw), "--representation", "raw"])
            code = main([
                str(raw),
                "-o",
                str(tokens),
                "--input-kind",
                "tmatrix",
                "--max-tokens",
                "128",
            ])
            payload = json.loads(tokens.read_text())
        self.assertEqual(code, 0)
        self.assertEqual(payload["format"], "uninet-tmatrix-tokenized")
        self.assertEqual(len(payload["samples"][0]["input"]), 128)

    def test_packet_json_adapter(self):
        records = [
            {
                "time": "2026-01-01T00:00:00Z",
                "src": "10.0.0.8",
                "dst": "1.1.1.1",
                "sport": 50000,
                "dport": 53,
                "proto": "UDP",
                "length": 72,
                "class": "dns",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "packets.json"
            output = directory / "tokens.json"
            source.write_text(json.dumps({"packets": records}))
            code = main([
                str(source),
                "-o",
                str(output),
                "--input-kind",
                "packet",
                "--label-column",
                "class",
                "--max-tokens",
                "64",
            ])
            payload = json.loads(output.read_text())
        self.assertEqual(code, 0)
        self.assertEqual(payload["samples"][0]["sequence_label"], "dns")


if __name__ == "__main__":
    unittest.main()
