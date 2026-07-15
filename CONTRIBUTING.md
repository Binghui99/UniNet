# Contributing

Contributions should preserve the PCAP-to-T-Matrix contract or explicitly version it.
Before opening a pull request, run `python -m unittest discover -s tests -v` and add a
regression test for behavioral changes. Do not commit third-party or sensitive PCAPs.

For a new feature, document its unit, type (continuous/categorical), segment, missing
value behavior, and compatibility impact. For benchmark claims, include the split
manifest, seed, metric implementation, and tokenizer fitted only on training data.

