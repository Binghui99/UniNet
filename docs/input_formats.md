# T-Matrix input formats

The standalone `tmatrix.py` command maps several common traffic formats to one
versioned UniNet representation. Use `--input-kind` when automatic detection is
ambiguous.

## Packet tables

Supported containers are CSV, TSV, JSON arrays, `{ "packets": [...] }`, and JSONL.
Required semantic fields and accepted aliases are:

| Semantic field | Accepted examples | Required |
|---|---|---|
| Timestamp | `timestamp`, `time`, `ts`; Unix seconds or ISO-8601 | Yes |
| Source IP | `src_ip`, `source_ip`, `src`, `saddr` | Yes |
| Destination IP | `dst_ip`, `destination_ip`, `dst`, `daddr` | Yes |
| Source/destination port | `src_port`/`dst_port`, `sport`/`dport` | No; defaults to -1 |
| IP protocol | `protocol`, `proto`; number or TCP/UDP/ICMP | Yes |
| Wire size | `size`, `length`, `packet_size`, `frame_len` | Yes |
| TCP flags | Numeric/hex mask or `SYN|ACK` style text | No |
| Sequence label | Configurable using `--label-column` | No |

Packet rows are grouped into static sessions and bidirectional 5-tuple flows before
the three granularities are extracted.

## Flow tables

Flow rows require timestamp, endpoint addresses and protocol. They may additionally
provide `duration`, `mean_iat`, `std_iat`, `packet_count`, `byte_count`,
`mean_packet_size`, `std_packet_size`, and `tcp_flag_code` (common aliases are also
accepted). Missing aggregates become zero, except mean packet size, which is derived
from bytes/count when possible.

A flow-only source has no packet observations. Its standardized T-Matrix therefore
contains session and flow segments with an empty packet segment. The converter never
creates artificial packets to make the shape look complete.

## JSON and JSONL

JSON tables can be a top-level list or an object containing `records`, `packets`, or
`flows`. JSONL expects one input record per line. Tokenized T-Matrix JSONL is already
standardized and should be consumed directly by the task scripts.

## Raw and tokenized output

- `--representation raw` preserves human-readable feature dictionaries.
- `--representation tokenized` produces fixed-length T-Attent input.
- `--representation both` writes tokenized output to `--output` and raw output to
  `--raw-output` (or `<output-stem>.raw.json`).

Fit the equal-frequency tokenizer only on training data. Save it with
`--tokenizer-out` and apply it to validation/test inputs using `--tokenizer-in`.

