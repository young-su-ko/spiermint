from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from ._mint.data import Alphabet
from ._model import load_model


@dataclass(frozen=True)
class ScoreResult:
    """Detailed output for a target-conditioned binder PLL score."""

    score: float
    pll: float
    length: int
    mode: str


def encode_chain(sequence: str, alphabet) -> torch.Tensor:
    cleaned = sequence.strip().upper().replace("J", "L")
    if not cleaned:
        raise ValueError("Protein sequences must be non-empty.")
    tokens = alphabet.encode(f"<cls>{cleaned}<eos>")
    return torch.tensor(tokens, dtype=torch.long)


def build_pair(target: str, binder: str, alphabet) -> tuple[torch.Tensor, torch.Tensor]:
    target_tokens = encode_chain(target, alphabet)
    binder_tokens = encode_chain(binder, alphabet)
    chains = torch.cat([target_tokens, binder_tokens], dim=0)
    chain_ids = torch.cat(
        [
            torch.zeros_like(target_tokens, dtype=torch.int32),
            torch.ones_like(binder_tokens, dtype=torch.int32),
        ],
        dim=0,
    )
    return chains, chain_ids


def get_scored_positions(tokens: torch.Tensor, chain_ids: torch.Tensor, model, score_chain_id: int) -> torch.Tensor:
    valid_mask = (
        (tokens != model.cls_idx)
        & (tokens != model.eos_idx)
        & (tokens != model.padding_idx)
        & (chain_ids == score_chain_id)
    )
    return torch.nonzero(valid_mask, as_tuple=False).flatten()


def chunked(items: Sequence[int], chunk_size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def compute_binder_pll(
    model,
    chains: torch.Tensor,
    chain_ids: torch.Tensor,
    score_chain_id: int = 1,
    mask_batch_size: int = 64,
) -> ScoreResult:
    score_positions = get_scored_positions(chains, chain_ids, model, score_chain_id)
    if score_positions.numel() == 0:
        raise ValueError("No scorable binder residues found after filtering special tokens.")

    device = next(model.parameters()).device
    chains = chains.to(device)
    chain_ids = chain_ids.to(device)
    score_positions = score_positions.to(device)
    token_ids = chains[score_positions].tolist()
    mask_batch_size = max(1, int(mask_batch_size))

    pll_terms: list[float] = []
    with torch.no_grad():
        for pos_chunk in chunked(score_positions.tolist(), mask_batch_size):
            batch_tokens = chains.unsqueeze(0).repeat(len(pos_chunk), 1)
            batch_chain_ids = chain_ids.unsqueeze(0).repeat(len(pos_chunk), 1)

            for row_idx, seq_pos in enumerate(pos_chunk):
                batch_tokens[row_idx, seq_pos] = model.mask_idx

            logits = model(batch_tokens, batch_chain_ids)["logits"]
            selected_logits = logits[torch.arange(len(pos_chunk), device=device), pos_chunk]
            log_probs = torch.log_softmax(selected_logits, dim=-1)
            true_ids = torch.tensor(
                token_ids[len(pll_terms) : len(pll_terms) + len(pos_chunk)],
                dtype=torch.long,
                device=device,
            )
            pll_terms.extend(log_probs[torch.arange(len(pos_chunk), device=device), true_ids].tolist())

    pll = float(sum(pll_terms))
    length = len(pll_terms)
    return ScoreResult(score=pll / length, pll=pll, length=length, mode="exact")


def compute_binder_gordon_pll(
    model,
    chains: torch.Tensor,
    chain_ids: torch.Tensor,
    score_chain_id: int = 1,
    alpha: float = 0.1,
    beta: float = 0.1,
    epsilon: float = 1e-3,
) -> ScoreResult:
    score_positions = get_scored_positions(chains, chain_ids, model, score_chain_id)
    if score_positions.numel() == 0:
        raise ValueError("No scorable binder residues found after filtering special tokens.")

    device = next(model.parameters()).device
    chains = chains.to(device)
    chain_ids = chain_ids.to(device)
    score_positions = score_positions.to(device)

    with torch.no_grad():
        logits = model(chains.unsqueeze(0), chain_ids.unsqueeze(0))["logits"][0]
        probs = torch.softmax(logits, dim=-1)
        scale = (alpha + beta) / alpha
        shift = beta / alpha
        smoothed_probs = torch.clamp(scale * probs - shift, min=epsilon)

    selected_probs = smoothed_probs[score_positions, chains[score_positions]]
    log_probs = torch.log(selected_probs)
    pll = float(log_probs.sum().item())
    length = int(log_probs.numel())
    return ScoreResult(score=pll / length, pll=pll, length=length, mode="approx")


def is_oom_error(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


class MINTScorer:
    """Load MINT once and score target-conditioned binder sequences."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        mask_batch_size: int = 64,
        gordon_alpha: float = 0.1,
        gordon_beta: float = 0.1,
        gordon_epsilon: float = 1e-3,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mask_batch_size = max(1, int(mask_batch_size))
        self.gordon_alpha = gordon_alpha
        self.gordon_beta = gordon_beta
        self.gordon_epsilon = gordon_epsilon

        # MINT uses the ESM2 architecture with the ESM-1b alphabet/token IDs.
        self.alphabet = Alphabet.from_architecture("ESM-1b")
        self.model, self.checkpoint_path = load_model(self.device)

    def score(
        self,
        target: str,
        binder: str,
        approx: bool = False,
        return_details: bool = False,
        mask_batch_size: int | None = None,
        gordon_alpha: float | None = None,
        gordon_beta: float | None = None,
        gordon_epsilon: float | None = None,
    ) -> float | ScoreResult:
        chains, chain_ids = build_pair(target, binder, self.alphabet)
        if approx:
            result = compute_binder_gordon_pll(
                self.model,
                chains,
                chain_ids,
                score_chain_id=1,
                alpha=self.gordon_alpha if gordon_alpha is None else gordon_alpha,
                beta=self.gordon_beta if gordon_beta is None else gordon_beta,
                epsilon=self.gordon_epsilon if gordon_epsilon is None else gordon_epsilon,
            )
        else:
            result = self._score_exact_with_retries(chains, chain_ids, mask_batch_size)
        return result if return_details else result.score

    def score_many(
        self,
        pairs: Iterable[tuple[str, str]],
        approx: bool = False,
        return_details: bool = False,
    ) -> list[float | ScoreResult]:
        return [
            self.score(target=target, binder=binder, approx=approx, return_details=return_details)
            for target, binder in pairs
        ]

    def _score_exact_with_retries(
        self,
        chains: torch.Tensor,
        chain_ids: torch.Tensor,
        mask_batch_size: int | None,
    ) -> ScoreResult:
        current_batch_size = max(1, int(mask_batch_size or self.mask_batch_size))
        while True:
            try:
                return compute_binder_pll(
                    self.model,
                    chains,
                    chain_ids,
                    score_chain_id=1,
                    mask_batch_size=current_batch_size,
                )
            except RuntimeError as exc:
                if not is_oom_error(exc):
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if current_batch_size <= 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)


class _LazyMINTFacade:
    def __init__(self) -> None:
        self._scorer: MINTScorer | None = None

    def load(self, **kwargs) -> MINTScorer:
        self._scorer = MINTScorer(**kwargs)
        return self._scorer

    def scorer(self) -> MINTScorer:
        if self._scorer is None:
            self.load()
        assert self._scorer is not None
        return self._scorer

    def score(self, target: str, binder: str, approx: bool = False, **kwargs) -> float | ScoreResult:
        return self.scorer().score(target=target, binder=binder, approx=approx, **kwargs)

    def score_many(self, pairs: Iterable[tuple[str, str]], approx: bool = False, **kwargs):
        return self.scorer().score_many(pairs=pairs, approx=approx, **kwargs)


mint = _LazyMINTFacade()


def score(target: str, binder: str, approx: bool = False, **kwargs) -> float | ScoreResult:
    return mint.score(target=target, binder=binder, approx=approx, **kwargs)
