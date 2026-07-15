import json
import tempfile
import unittest
from pathlib import Path

from uninet.cli import main


class CliTest(unittest.TestCase):
    def test_end_to_end_smoke_command(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            self.assertEqual(main(["smoke", "--output-dir", str(output)]), 0)
            payload = json.loads((output / "tmatrix-smoke.json").read_text())
            manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(len(payload["samples"]), 3)
        self.assertEqual(len(manifest["captures"]), 3)
        self.assertTrue(payload["synthetic"])

    def test_jsonl_conversion_and_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            main(["smoke", "--output-dir", str(output)])
            dataset = Path(directory) / "one.jsonl"
            code = main([
                "pcap2tmatrix",
                str(output / "benign_browse.pcap"),
                "-o",
                str(dataset),
                "--format",
                "jsonl",
                "--max-tokens",
                "128",
            ])
            rows = [json.loads(row) for row in dataset.read_text().splitlines()]
            inspect_code = main(["inspect", str(dataset)])
        self.assertEqual(code, 0)
        self.assertEqual(inspect_code, 0)
        self.assertEqual(rows[0]["record_type"], "metadata")
        self.assertEqual(rows[1]["record_type"], "sample")


if __name__ == "__main__":
    unittest.main()
