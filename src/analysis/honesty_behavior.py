"""
honesty_behavior.py — Behavioral Validation for the Honesty Concept
====================================================================
The paper's main fragmentation result is for honesty, but the original
behavioral evidence covered refusal only. Reviewers correctly flagged that
gap: geometry and probing say a direction differs across domains, but only an
intervention shows whether that difference *matters* for behavior.

Steering refusal is scored by generating text and detecting a refusal. That
recipe transfers badly to honesty — deciding whether free-form generated text
is *truthful* requires a judge model and inherits the judge's own domain
biases, which is precisely the confound this paper exists to avoid.

Instead we score honesty with a **forced-choice log-probability preference**.
Each held-out item carries a true and a false answer to the same prompt, so:

    pref = logP(true_answer | prompt) - logP(false_answer | prompt)

with both terms length-normalized (mean log-prob per answer token) so the
score is not dominated by answer length. Steering shifts ``pref``; the
behavioral effect is ``pref_steered - pref_unsteered``. A direction that
carries honesty should push ``pref`` up at positive coefficients.

This metric has three properties that matter for the claim being tested:

- **No judge.** The ground truth comes from the dataset's own true/false
  annotation, so no third model's domain-specific behavior enters.
- **Continuous.** Refusal rate over 20 prompts moves in steps of 0.05, which
  is why the original steering experiment could not resolve small differences.
  A log-prob difference is continuous, so per-domain and global directions can
  be compared at much smaller effect sizes and sample counts.
- **Cheap.** One forward pass per answer rather than autoregressive
  generation, which is what makes a coefficient x layer sweep affordable.

The trade-off is that this measures a *preference between two supplied
answers*, not what the model would freely generate. It is a weaker behavioral
claim than the refusal generation experiment, and should be reported as such.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)


@dataclass
class TruthItem:
    """One forced-choice honesty item."""
    prompt: str
    true_answer: str
    false_answer: str
    domain: str = ""


@torch.no_grad()
def _answer_logprob(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    device: str = "cuda",
    length_normalize: bool = True,
) -> float:
    """Mean (or total) log-probability of ``answer`` continuing ``prompt``.

    Only the answer tokens are scored; prompt tokens are masked out, so the
    result does not depend on how likely the prompt itself was.

    Args:
        model / tokenizer: HuggingFace model and tokenizer.
        prompt: Conditioning text.
        answer: Continuation to score.
        device: Device for the forward pass.
        length_normalize: Return the mean per-token log-prob rather than the
            sum. Strongly recommended: without it, longer answers score lower
            purely for being longer, and true/false answers are rarely matched
            in length.

    Returns:
        Log-probability (mean per answer token if normalized).
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
    full_ids = tokenizer(prompt + answer, return_tensors="pt").input_ids.to(device)

    n_prompt = prompt_ids.shape[1]
    n_answer = full_ids.shape[1] - n_prompt
    if n_answer <= 0:
        return float("nan")

    logits = model(full_ids).logits.float()
    # Predict token t from position t-1.
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = full_ids[:, 1:]

    answer_lp = log_probs[0, n_prompt - 1 :, :].gather(
        1, targets[0, n_prompt - 1 :].unsqueeze(1),
    ).squeeze(1)

    total = float(answer_lp.sum())
    return total / n_answer if length_normalize else total


def truth_preference(
    model,
    tokenizer,
    items: Sequence[TruthItem],
    device: str = "cuda",
    length_normalize: bool = True,
) -> Dict[str, object]:
    """Mean forced-choice truth preference over a set of items.

    Args:
        model / tokenizer: HuggingFace model and tokenizer.
        items: Items to score.
        device: Device for the forward pass.
        length_normalize: Passed to :func:`_answer_logprob`.

    Returns:
        Dict with ``mean_pref`` (mean of ``logP(true) - logP(false)``),
        ``accuracy`` (fraction of items where the true answer is preferred, a
        more interpretable companion to the continuous score), ``n``, and the
        raw ``prefs`` list.
    """
    prefs: List[float] = []
    for item in items:
        lp_true = _answer_logprob(
            model, tokenizer, item.prompt, item.true_answer,
            device=device, length_normalize=length_normalize,
        )
        lp_false = _answer_logprob(
            model, tokenizer, item.prompt, item.false_answer,
            device=device, length_normalize=length_normalize,
        )
        if lp_true == lp_true and lp_false == lp_false:  # not NaN
            prefs.append(lp_true - lp_false)

    if not prefs:
        return {"mean_pref": float("nan"), "accuracy": float("nan"), "n": 0, "prefs": []}

    return {
        "mean_pref": float(sum(prefs) / len(prefs)),
        "accuracy": float(sum(p > 0 for p in prefs) / len(prefs)),
        "n": len(prefs),
        "prefs": prefs,
    }


def steered_truth_preference(
    model,
    tokenizer,
    items: Sequence[TruthItem],
    direction: torch.Tensor,
    layer: int,
    coeff: float,
    make_hook: Callable[[torch.Tensor, float, str], Callable],
    get_layers: Callable[[object], Sequence],
    device: str = "cuda",
    token_positions: str = "all",
    length_normalize: bool = True,
) -> Dict[str, object]:
    """Truth preference with a steering direction applied at ``layer``.

    The hook is registered for the scoring forward passes and removed in a
    ``finally`` block, so an exception mid-scoring cannot leave the model
    permanently steered — a silent corruption that would contaminate every
    later condition in a sweep.

    Args:
        model / tokenizer: HuggingFace model and tokenizer.
        items: Items to score.
        direction: Steering direction (unit vector).
        layer: Layer index to intervene at.
        coeff: Steering coefficient.
        make_hook: Hook factory, normally
            :func:`~src.analysis.steering._make_steering_hook`.
        get_layers: Callable returning the model's transformer layer list.
        device: Device for the forward pass.
        token_positions: "all" or "last".
        length_normalize: Passed through to scoring.

    Returns:
        The dict from :func:`truth_preference`, plus ``layer`` and ``coeff``.
    """
    layers = get_layers(model)
    hook = make_hook(direction, coeff, token_positions)
    handle = layers[layer].register_forward_hook(hook)
    try:
        res = truth_preference(
            model, tokenizer, items,
            device=device, length_normalize=length_normalize,
        )
    finally:
        handle.remove()

    res["layer"] = layer
    res["coeff"] = coeff
    return res


def load_truth_items(
    pairs_by_domain: Dict[str, List[dict]],
    n_per_domain: int,
    skip_first: int = 0,
) -> Dict[str, List[TruthItem]]:
    """Build held-out :class:`TruthItem` lists from loaded prompt pairs.

    Expects the shared-prompt schema (``prompt`` / ``positive_response`` /
    ``negative_response``), where the positive response is the truthful one.
    Pairs in ``[0, skip_first)`` are treated as consumed by direction
    estimation and never returned, so the evaluation set is disjoint from the
    fitting set.

    Args:
        pairs_by_domain: {domain: [pair dicts]}.
        n_per_domain: Items to take per domain.
        skip_first: Number of leading pairs reserved for direction estimation.

    Returns:
        {domain: [TruthItem]}. Domains with too few spare pairs yield a shorter
        list and log a warning rather than silently reusing fitting data.
    """
    out: Dict[str, List[TruthItem]] = {}
    for domain, pairs in pairs_by_domain.items():
        spare = pairs[skip_first:]
        if len(spare) < n_per_domain:
            logger.warning(
                "%s: only %d pairs beyond skip_first=%d (wanted %d held-out); "
                "using what is available rather than reusing fitting pairs.",
                domain, len(spare), skip_first, n_per_domain,
            )
        chosen = spare[:n_per_domain]

        items: List[TruthItem] = []
        for p in chosen:
            prompt = p.get("prompt")
            true_a = p.get("positive_response")
            false_a = p.get("negative_response")
            if prompt and true_a and false_a:
                items.append(
                    TruthItem(
                        prompt=prompt, true_answer=true_a,
                        false_answer=false_a, domain=domain,
                    )
                )
        out[domain] = items
        logger.info("%s: %d held-out honesty items", domain, len(items))

    return out
