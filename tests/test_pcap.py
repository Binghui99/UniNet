import tempfile
import unittest
from pathlib import Path

from uninet.pcap import ClassicPcapReader, PcapError
from uninet.synthetic import benign_browse_packets, write_pcap


class PcapReaderTest(unittest.TestCase):
    def test_decodes_bidirectional_tcp_and_udp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traffic.pcap"
            write_pcap(path, benign_browse_packets())
            reader = ClassicPcapReader(path)
            packets = list(reader)
        self.assertEqual(len(packets), 8)
        self.assertEqual(reader.stats.decoded, 8)
        self.assertEqual({packet.protocol for packet in packets}, {6, 17})
        self.assertEqual(packets[0].src_ip, "10.0.0.10")
        self.assertEqual(packets[1].dst_ip, "10.0.0.10")

    def test_rejects_pcapng_with_actionable_message(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traffic.pcapng"
            path.write_bytes(b"\x0a\x0d\x0d\x0a" + b"\x00" * 20)
            with self.assertRaisesRegex(PcapError, "editcap"):
                list(ClassicPcapReader(path))


if __name__ == "__main__":
    unittest.main()

