# T-Matrix preprocessing design

This document makes the reference implementation's assumptions inspectable. The
paper deliberately permits users to add or remove semantic features; consequently,
dataset-specific choices should be recorded together with every trained model.

The schema below is a general paper-aligned default, not a claim that every task in
the paper used identical fields. For example, the website-fingerprinting experiment
uses task-specific inbound/outbound session aggregates described in Section V-E.

## Granularities and ordering

A sample is one contextual endpoint in one static time window. It is flattened as:

1. eight session values (`segment_label=2`);
2. for each flow in first-packet order, eight flow values (`segment_label=1`);
3. each flow's packets in arrival order, six packet values (`segment_label=0`).

The sequence is truncated to `max_tokens` and right-padded with token 1041. The
default is 2,000, matching the experiments in the paper.

### Session features

| Feature | Definition |
|---|---|
| `flow_count` | Bidirectional flows in the session |
| `unique_src_ips`, `unique_dst_ips` | Endpoint diversity |
| `unique_service_ports` | Distinct ports 1-1024, 3306, or 8080 |
| `mean_flow_duration`, `std_flow_duration` | Population statistics in seconds |
| `mean_flow_bytes`, `std_flow_bytes` | Population statistics over wire lengths |

### Flow features

Duration; mean and population standard deviation of IAT; packet and byte counts;
mean and population standard deviation of packet wire size; compact TCP-flag code.
Both directions share one canonical 5-tuple. A later packet after the inactivity
timeout starts a new flow.

TCP flag codes follow the paper: individual ACK/SYN/FIN/PSH/URG/RST/ECE/CWR/NS
flags use 1-9; SYN+ACK, PSH+ACK, URG+ACK, FIN+ACK, and RST+ACK use 10-14;
uncommon combinations use 15; non-TCP/no-flags uses 0.

### Packet features

Source port representation, destination port representation, IP protocol number,
direction, flow-relative IAT in seconds, and packet wire size. Direction 0 means
the packet originates at the contextual IP; 1 means it is incoming.

Ports 1-1024 keep their number. 8080 maps to 1025, 3306 to 1026, and all other
or missing ports map to 1027, following Table III of the paper.

## Context selection

`internal` is the practical default. For a packet crossing the boundary, its RFC1918,
loopback, link-local, or ULA endpoint becomes the context. When both endpoints are
internal, the source is selected; in asymmetric monitoring, `key-ip` is preferable.

Modes `src`, `dst`, and `pair` are also provided. Context selection affects session
semantics, so record the CLI and internal CIDRs in experiment metadata.

## Tokenizer fitting and leakage

Continuous fields are encoded with per-feature equal-frequency boundaries. Fit the
tokenizer on the training split only, save it using `--tokenizer-out`, then reuse it
for validation and test sets with `--tokenizer-in`. Fitting on all splits leaks test
distribution information.

Categorical values retain their compact numeric token IDs. Continuous bins occupy
0-1039; 1040 and 1041 are reserved for `[MASK]` and `[PAD]`.

## Masked feature prediction

At ratio eta, a deterministic random subset of real (non-padding) input positions is
replaced by 1040. `true_value` stores original IDs only at masked positions and is 0
elsewhere; `mask_index` is the corresponding binary vector. Use eta=0 for supervised
classification. The paper reports exploring 0.15-0.60 and selecting 0.40 for its
unsupervised experiment.

## Parser boundaries

The dependency-free reader supports classic PCAP (microsecond or nanosecond stamps),
Ethernet with stacked VLAN tags, Linux SLL v1, and raw IPv4/IPv6. It extracts TCP and
UDP ports/flags and retains other IP protocols with ports -1. Non-IP frames and
non-initial fragments are skipped and counted. It does not perform IP reassembly.

PCAPNG must currently be converted to classic PCAP with `editcap`. This narrow parser
surface makes CI deterministic; a future optional Scapy backend can expand link-layer
coverage without changing the T-Matrix schema.

The reference converter currently materializes decoded sessions while fitting
quantiles. For multi-gigabyte captures, split files on session-window boundaries or
convert shards separately while reusing one training tokenizer.
