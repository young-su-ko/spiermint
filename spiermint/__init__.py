"""Programmatic scoring API for target-conditioned MINT binder PLLs."""

__all__ = ["MINTScorer", "ScoreResult", "mint", "score"]


def __getattr__(name: str):
    if name in __all__:
        from . import scoring

        return getattr(scoring, name)
    raise AttributeError(f"module 'spiermint' has no attribute {name!r}")
