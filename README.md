# spiermint

Scoring Protein IntERactions with MINT.

`spiermint` is a small package for scoring target-conditioned binder pseudo-log-likelihoods with MINT. It vendors the minimal MINT inference code it needs, so users do not need to install the original MINT repository.

## Install

Install directly from GitHub:

```bash
pip install "spiermint @ git+https://github.com/young-su-ko/spiermint.git"
```

For local development:

```bash
git clone https://github.com/young-su-ko/spiermint.git
cd spiermint
uv venv --python 3.10
uv sync
```

Runtime dependencies are intentionally small: `torch` for inference and `typer` for the CLI.

## Checkpoint

The MINT checkpoint is downloaded on first model load from:

```text
https://huggingface.co/varunullanat2012/mint/resolve/main/mint.ckpt
```

The checkpoint is cached as `mint.ckpt` in the first applicable location:

1. `$SPIERMINT_CACHE_DIR/mint.ckpt`
2. `$XDG_CACHE_HOME/spiermint/mint.ckpt`
3. `~/.cache/spiermint/mint.ckpt`

Set `SPIERMINT_CACHE_DIR` to put the checkpoint somewhere else. The package uses that directory directly; it does not append an extra `spiermint/` subdirectory.

## Python API

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

`approx=False` computes the exact masked binder PLL. `approx=True` uses the Gordon-style approximation from the original MINT benchmark code.

## CLI

```bash
spiermint --target "$TARGET" --binder "$BINDER"
spiermint --input-csv pairs.csv --target-col target_sequence --binder-col binder_sequence --approx
```

Single-pair scoring prints `score`, `pll`, `length`, and `mode`. CSV scoring writes a new column named `spiermint_score` by default and writes to `<input>_spiermint.csv` unless `--output-csv` is provided.
