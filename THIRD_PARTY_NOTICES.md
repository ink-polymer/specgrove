# Third-Party Code and Models

The package retains the original LICENSE files and notices under each
`code/*/third_party/` snapshot. Public upstream authorship is not removed
for submission anonymity.

- `ddtree_pinned/`: pinned DDTree implementation, MIT License, copyright
  Liran Ringel. Its `SOURCE_SHA256.json` records the upstream commit and
  file fingerprints, including the documented final-newline normalization.
- `ddtree_official/`: recorded official DDTree source, distributed with its
  original MIT LICENSE and README.
- `dflash_official/`: recorded DFlash source, distributed with its original
  MIT LICENSE, README, and package metadata.
- PyTorch, Transformers, NumPy, Hugging Face datasets/hub, and the math
  verifier are external dependencies and retain their respective licenses.
- Qwen and DFlash model weights are not redistributed. Use the public
  checkpoint IDs and exact revisions in `code/*/configs/adaptive_block_qwen3_*.json` and
  comply with the corresponding model licenses.
- Dataset files are not redistributed in this package. The preparation code
  names the public dataset sources; their licenses apply to downloaded data.

This notice does not assign a new license to the authors' research code.
No private repository URL, credentials, or model download cache is needed
for inspecting the included research code. Recorded results are distributed separately.
