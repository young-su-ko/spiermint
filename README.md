# spiermint

Scoring Protein IntERactions with MINT.

`spiermint` is a small package for scoring target-conditioned binder pseudo-log-likelihoods with MINT.

Install from GitHub:

```bash
pip install "spiermint @ git+https://github.com/young-su-ko/spiermint.git"
```

`spiermint` vendors the minimal MINT inference code it needs, so users do not need to install MINT separately. On first use, it downloads the MINT checkpoint from Hugging Face and caches it locally.

```python
from spiermint import mint

mint.load()
score = mint.score(target=target_sequence, binder=binder_sequence, approx=False)
approx_score = mint.score(target=target_sequence, binder=binder_sequence, approx=True)
```

For repeated scoring, load the model once:

```python
from spiermint import MINTScorer

scorer = MINTScorer(device="cuda:0")
score = scorer.score(target=target_sequence, binder=binder_sequence)
details = scorer.score(target=target_sequence, binder=binder_sequence, approx=True, return_details=True)
```

CLI usage:

```bash
spiermint --target "$TARGET" --binder "$BINDER"
spiermint --input-csv pairs.csv --target-col target_sequence --binder-col binder_sequence --approx
```

By default, the checkpoint is cached under `~/.cache/spiermint/mint.ckpt`, or under `$XDG_CACHE_HOME/spiermint/mint.ckpt` when `XDG_CACHE_HOME` is set. Set `SPIERMINT_CACHE_DIR` to choose a different cache directory.

Runtime dependencies are intentionally minimal: `torch` for inference and `typer` for the CLI.
