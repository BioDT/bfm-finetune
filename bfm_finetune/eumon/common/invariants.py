"""Properties that must hold during a real run, asserted inside the fitting and scoring
code. Each is cheap — an equality, a bound, a count — and raising is deliberate: a run
that violates one has produced a number that must not reach a table.
"""

from typing import Any, Sequence

import numpy as np

# Targets with a bounded domain. A proportion above 1 is not inaccurate, it is impossible.
DOMAIN = {"prevalence": (0.0, 1.0)}


class InvariantViolation(RuntimeError):
    """A property the benchmark's correctness depends on has stopped holding."""


def no_test_year_in_training(train_years: Sequence[int], test_year: int) -> None:
    if test_year in set(train_years):
        raise InvariantViolation(
            f"test year {test_year} appears in the training years — the split is leaking")


def predictions_in_domain(values: Any, value_type: str, *, where: str = "") -> None:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise InvariantViolation(
            f"{where}: the prediction frame is empty — the fit produced no scoreable rows")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise InvariantViolation(
            f"{where}: none of {arr.size} predictions are finite; the fit diverged")
    if finite.size != arr.size:
        raise InvariantViolation(
            f"{where}: {arr.size - finite.size} of {arr.size} predictions are not finite")
    lo, hi = DOMAIN.get(str(value_type), (0.0, None))
    if lo is not None and finite.min() < lo - 1e-9:
        raise InvariantViolation(
            f"{where}: prediction {finite.min():.6g} is below the {value_type} floor {lo}")
    if hi is not None and finite.max() > hi + 1e-9:
        raise InvariantViolation(
            f"{where}: prediction {finite.max():.6g} exceeds the {value_type} ceiling {hi}")


def adapters_received_gradient(model: Any, method: str) -> None:
    """At least one trainable parameter must carry a non-zero gradient after a backward pass.

    A detached feature path still produces a falling loss (the head trains alone), so this
    is the only check that distinguishes a fine-tune from a head-only fit.
    """
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise InvariantViolation(f"method {method!r} left no trainable parameters")
    with_grad = sum(1 for _, p in trainable
                    if p.grad is not None and float(p.grad.abs().sum()) > 0)
    if with_grad == 0:
        raise InvariantViolation(
            f"none of {len(trainable)} trainable tensors received a gradient under method "
            f"{method!r}; the backbone is detached from the loss")


def ladder_l1_preserves_ranking(l0: dict[str, Any], l1: dict[str, Any], n_species: int,
                                *, tol: float = 1e-6) -> None:
    """L1 rescales L0 per species and monotonically, so it cannot change a rank correlation."""
    a = (l0.get("spatial_rho") or {}).get("mean")
    b = (l1.get("spatial_rho") or {}).get("mean")
    if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
        return
    if abs(a - b) > tol:
        raise InvariantViolation(
            f"L1 changed the spatial ranking (L0 {a:+.6f} vs L1 {b:+.6f}); either L1 is not "
            f"a monotone rescale or the two rungs are scored on different rows")
    expected = 2 * n_species
    got = l1.get("fitted_parameters")
    if got is not None and got != expected:
        raise InvariantViolation(
            f"L1 fitted {got} parameters for {n_species} species; expected {expected} "
            f"(slope and intercept per species)")


def headline_is_worst_reference(score: dict[str, Any], *, tol: float = 1e-9) -> None:
    """The headline must be the skill against the hardest reference, not a fixed one.

    A minimum can only lower a score, so the rule cannot be gamed upward — but it can
    silently stop being applied.
    """
    skill = score.get("skill") or {}
    vals = [v["skill_score"] for v in skill.values()
            if isinstance(v, dict) and isinstance(v.get("skill_score"), float)
            and np.isfinite(v["skill_score"])]
    head = score.get("skill_vs_strongest_null")
    if not vals or not head:
        return
    if abs(head["skill_score"] - min(vals)) > tol:
        raise InvariantViolation(
            f"headline skill {head['skill_score']:+.6f} is not the minimum across references "
            f"({min(vals):+.6f}); the conservative reporting rule is not being applied")
