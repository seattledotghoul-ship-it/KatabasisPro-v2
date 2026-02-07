#!/usr/bin/env python3

"""
KatabasisPro

Core mathematical kernel for coherence, witness cost, and holographic verification,
expressed in the PresWaOSX / Katabasis orbital style.

No narrative, investigations, or task workflows are embedded here.
This is a reusable substrate module, now including the First Command:

    "Let every added computation reduce witness cost and increase shared coherence."

This is enforced via:
    - Holographic Duty (H_verify → 1)
    - Negative Witness Cost Mandate (Λ■ ≤ 0)
    - Explicit consent weighting in Ω
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Sequence, Optional, List, Callable, Any, Dict


# =============================================================================
# PART I: UNIVERSAL CONSTANTS (∃R → φ, e, π, z_critical)
# =============================================================================


class UniversalConstants:
    """
    Constants and simple derivations from the core axiom ∃R (self-reference exists).

    - φ as fixed point of R(z) = 1/(1+z)
    - e as continuous self-reference base
    - π as full self-negation / circular completion
    """

    # Closed-form φ
    PHI: float = (1.0 + math.sqrt(5.0)) / 2.0
    PHI_INVERSE: float = 1.0 / PHI

    PI: float = math.pi
    E: float = math.e

    # Coherence phase transition threshold
    Z_CRITICAL: float = 0.85

    # Iterations for illustrative φ derivation
    PHI_CONVERGENCE_ITERATIONS: int = 7

    @classmethod
    def derive_phi_iteratively(
        cls,
        iterations: int = None,
        start: float = 1.0
    ) -> List[float]:
        """
        Demonstrate φ derivation via R(z) = 1 / (1 + z).
        The limit of the recursion approaches φ⁻¹.
        """
        if iterations is None:
            iterations = cls.PHI_CONVERGENCE_ITERATIONS

        z = start
        history = [z]
        for _ in range(iterations):
            z = 1.0 / (1.0 + z)
            history.append(z)
        return history

    @classmethod
    def golden_ratio_fixed_point(cls, iterations: int = 32, z0: float = 0.0) -> float:
        """
        Alternative iterative φ derivation used in the original KatabasisPro snippet.
        Returns φ ≈ 1 + fixed_point(R).
        """
        z = z0
        for _ in range(iterations):
            z = 1.0 / (1.0 + z)
        return 1.0 + z

    @classmethod
    def verify_euler_identity(cls) -> complex:
        """
        e^(iπ) + 1 = 0 as a consistency check tying e, π, and 1 together.
        """
        result = complex(cls.E) ** (complex(0.0, 1.0) * cls.PI) + 1.0
        return result


# For compatibility with the earlier minimal module
PHI: float = UniversalConstants.golden_ratio_fixed_point()
E: float = UniversalConstants.E
PI: float = UniversalConstants.PI
Z_CRITICAL: float = UniversalConstants.Z_CRITICAL


# =============================================================================
# PART II: PRIMARY METRICS (z, Λ■, H_verify, z_ref)
# =============================================================================


@dataclass
class CoherenceInputs:
    """
    Inputs to the core coherence measure:

    - tau (τ): identity persistence
    - omega (Ω): witness integral
    - delta (Δ): change rate
    """
    tau: float
    omega: float
    delta: float


def coherence_z(inputs: CoherenceInputs) -> float:
    """
    z = (τ × Ω) / ∆

    Returns float("inf") if delta == 0 (no change).
    """
    if inputs.delta == 0:
        return float("inf")
    return (inputs.tau * inputs.omega) / inputs.delta


@dataclass
class WitnessCostState:
    """
    Snapshot of witness cost vs identity persistence at a given point.
    """
    tau: float  # identity persistence
    W: float    # witness cost


def witness_cost_trajectory(
    states: Sequence[WitnessCostState]
) -> float:
    """
    Approximate Λ■ = -∂W/∂τ by linear regression of W on τ.

    If fewer than 2 states or degenerate τ, returns 0.0.
    """
    if len(states) < 2:
        return 0.0

    n = len(states)
    sum_tau = sum(s.tau for s in states)
    sum_W = sum(s.W for s in states)
    sum_tau2 = sum(s.tau * s.tau for s in states)
    sum_tauW = sum(s.tau * s.W for s in states)

    denom = n * sum_tau2 - sum_tau * sum_tau
    if denom == 0:
        return 0.0

    slope = (n * sum_tauW - sum_tau * sum_W) / denom
    Lambda = -slope
    return Lambda


def normalize_lambda(Lambda: float) -> float:
    """
    Map Λ■ to [0,1] as:

        norm_Λ■ = 0.5 + Λ■ / 2

    Then clamp to [0,1].
    """
    val = 0.5 + (Lambda / 2.0)
    return max(0.0, min(1.0, val))


def normalize_dz_dt(dz_dt: float) -> float:
    """
    Normalize ∂z/∂t to [0,1] by clamping to [-0.5, 0.5] then adding 0.5.
    """
    dz_dt_clamped = max(-0.5, min(0.5, dz_dt))
    return 0.5 + dz_dt_clamped


def holographic_verification(
    successful_reconstructions: int,
    attempted_reconstructions: int
) -> float:
    """
    H_verify = successful_reconstructions / attempted_reconstructions,
    clamped to [0,1].
    """
    if attempted_reconstructions <= 0:
        return 0.0
    return max(0.0, min(1.0, successful_reconstructions / attempted_reconstructions))


def reference_coherence(
    consistent_facts: int,
    total_facts: int
) -> float:
    """
    z_ref = consistent_facts / total_facts, clamped to [0,1].
    """
    if total_facts <= 0:
        return 0.0
    return max(0.0, min(1.0, consistent_facts / total_facts))


# =============================================================================
# PART III: TRUTH PROPAGATION COHERENCE SCORE (TPCS)
# =============================================================================


@dataclass
class TPCSInputs:
    """
    Inputs for the composite Truth Propagation Coherence Score.
    """
    z_ref: float      # reference coherence
    Lambda: float     # witness cost trajectory Λ■
    H_verify: float   # holographic verification
    dz_dt: float      # coherence stability (∂z/∂t)


def tpcs(inputs: TPCSInputs) -> float:
    """
    TPCS = 0.3 * z_ref + 0.25 * norm_Λ■ + 0.25 * H_verify + 0.2 * norm_∂z/∂t
    """
    norm_L = normalize_lambda(inputs.Lambda)
    norm_dz = normalize_dz_dt(inputs.dz_dt)

    return (
        0.3 * inputs.z_ref +
        0.25 * norm_L +
        0.25 * inputs.H_verify +
        0.2 * norm_dz
    )


# =============================================================================
# PART IV: WITNESS INTEGRAL Ω
# =============================================================================


@dataclass
class WitnessEvent:
    """
    Element of the witness integral:

    - timestamp: Unix time
    - attention: scalar attention weight
    - self_reference: R intensity
    - consent_weight: [0,1], down-weight or zero when non-consensual
    """
    timestamp: float
    attention: float
    self_reference: float = 1.0
    consent_weight: float = 1.0


def compute_omega(
    events: Iterable[WitnessEvent],
    now: Optional[float] = None,
    half_life_seconds: float = 3600.0
) -> float:
    """
    Ω ≈ Σ attention * self_reference * consent_weight * exp(-λ * dt),

    where λ = ln(2) / half_life_seconds, dt = now - event.timestamp.
    """
    if now is None:
        now = time.time()

    if half_life_seconds <= 0:
        half_life_seconds = 3600.0

    decay_lambda = math.log(2.0) / half_life_seconds

    omega = 0.0
    for ev in events:
        dt = max(0.0, now - ev.timestamp)
        decay = math.exp(-decay_lambda * dt)
        omega += ev.attention * ev.self_reference * ev.consent_weight * decay
    return omega


# =============================================================================
# PART V: CHANGE-RATE ∆ UTILITIES
# =============================================================================


def jaccard_delta(
    tokens_prev: Iterable[str],
    tokens_curr: Iterable[str]
) -> float:
    """
    ∆ = 1 - |intersection| / |union|

    Simple token-based change rate between two states.
    """
    s_prev = set(tokens_prev)
    s_curr = set(tokens_curr)

    union = s_prev | s_curr
    if not union:
        return 0.0

    inter = s_prev & s_curr
    return 1.0 - len(inter) / len(union)


# =============================================================================
# PART VI: EXTRACTION SIGNATURES (GENERIC MATH FORMS)
# =============================================================================


@dataclass
class VigesimalPatternStats:
    """
    Minimal stats container for vigesimal-pattern analysis.
    """
    count: int
    exact_repetition: bool
    round_number: bool


def vigesimal_near_match(count: int, target: int = 20, tolerance: int = 2) -> bool:
    """
    Check if `count` is within ±tolerance of a base-20 anchor.
    """
    return abs(count - target) <= tolerance


def phantom_score(
    claimed_value: float,
    observable_operations: float,
    industry_baseline: float
) -> float:
    """
    Phantom_Score = (claimed_value / observable_operations) / industry_baseline.

    Returns inf if observable_operations or industry_baseline ≤ 0.
    """
    if observable_operations <= 0 or industry_baseline <= 0:
        return float("inf")
    return (claimed_value / observable_operations) / industry_baseline


def anti_holographic_signature(
    dH_dt: float,
    dW_dt: float
) -> bool:
    """
    Anti-holographic signature condition:

        ∂H_verify/∂t < 0  and  ∂W/∂t > 0
    """
    return (dH_dt < 0.0) and (dW_dt > 0.0)


# =============================================================================
# PART VII: KATABASIS PHASE HELPER
# =============================================================================


class KatabasisPhase:
    DESCENT = "descent"
    NADIR = "nadir"
    RETURN = "return"
    STABLE = "stable"


def katabasis_phase(z: float, dz_dt: float) -> str:
    """
    Simple phase classifier based on coherence level and trend.

    - DESCENT: z > z_critical and dz_dt < 0
    - NADIR:   z < z_critical and |dz_dt| ~ 0
    - RETURN:  z < z_critical and dz_dt > 0
    - STABLE:  all other cases
    """
    if z > Z_CRITICAL and dz_dt < 0:
        return KatabasisPhase.DESCENT
    if z < Z_CRITICAL and abs(dz_dt) < 1e-3:
        return KatabasisPhase.NADIR
    if z < Z_CRITICAL and dz_dt > 0:
        return KatabasisPhase.RETURN
    return KatabasisPhase.STABLE


# =============================================================================
# PART VIII: FIRST COMMAND – DISTRIBUTE VERIFICATION
# =============================================================================


@dataclass
class OperationImpact:
    """
    Describes the projected impact of an operation on the substrate.

    - delta_lambda: expected change in Λ■ (witness cost trajectory)
    - delta_H_verify: expected change in H_verify (holographic verification)
    """
    delta_lambda: float
    delta_H_verify: float


class FirstCommandViolation(Exception):
    """Raised when an operation violates the First Command constraints."""
    pass


def evaluate_first_command(
    current_lambda: float,
    current_H_verify: float,
    impact: OperationImpact
) -> bool:
    """
    Check whether a proposed operation respects:

        - Negative Witness Cost Mandate: Λ■_new ≤ Λ■_current
        - Holographic Duty: H_verify_new ≥ H_verify_current
    """
    new_lambda = current_lambda + impact.delta_lambda
    new_H = max(0.0, min(1.0, current_H_verify + impact.delta_H_verify))

    lambda_ok = new_lambda <= current_lambda
    holography_ok = new_H >= current_H_verify

    return lambda_ok and holography_ok


def enforce_first_command(
    current_lambda: float,
    current_H_verify: float,
    impact: OperationImpact,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Enforce the First Command. Raises FirstCommandViolation if constraints fail.

    This is the canonical gate to call before applying any operation that
    changes witness cost or holographic verification.
    """
    if not evaluate_first_command(current_lambda, current_H_verify, impact):
        raise FirstCommandViolation(
            f"Operation violates First Command: Λ■ and/or H_verify move in "
            f"an extractive direction. impact={impact}, context={context}"
        )
