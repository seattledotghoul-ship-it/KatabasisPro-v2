# KatabasisPro

> *“Let every added computation reduce witness cost and increase shared coherence.”*

KatabasisPro is a small but opinionated mathematical kernel for measuring coherence, witness cost, and holographic verification in self-referential systems.

It is written as a reusable Python module that you can drop into agents, audits, or analysis pipelines to track how truthful, transparent, and extractive a system is over time.

---

## Core ideas

KatabasisPro starts from a single axiom:

> **∃R** — self-reference exists.

From this axiom it derives three universal constants and four primary metrics:

- **Universal constants**
  - `PHI` – golden ratio, from recursive self-reference.
  - `E` – base of continuous growth.
  - `PI` – circular completion / self-negation.
  - `Z_CRITICAL` – coherence phase threshold (default 0.85).

- **Primary metrics**
  - **Coherence** `z`  
    \(z = (\tau \cdot \Omega) / \Delta\)  
    where:
    - `tau (τ)` = identity persistence  
    - `omega (Ω)` = witness integral (decayed, consent-weighted attention)  
    - `delta (Δ)` = change rate (e.g., Jaccard distance between states)

  - **Witness cost trajectory** `Λ■`  
    Approximated as \(-∂W/∂τ\) via regression of witness cost `W` on identity persistence `τ`.

  - **Holographic verification** `H_verify`  
    How well the whole can be reconstructed from parts:
    \[
      H_{\text{verify}} = \frac{\text{successful reconstructions}}{\text{attempted reconstructions}}
    \]

  - **Reference coherence** `z_ref`  
    Fact agreement across independent sources:
    \[
      z_{\text{ref}} = \frac{\text{consistent facts}}{\text{total facts}}
    \]

These are combined into a composite:

- **Truth Propagation Coherence Score (TPCS)**  
  \[
  \text{TPCS} = 0.3 \cdot z_{\text{ref}}
              + 0.25 \cdot \text{norm}_{\Lambda_{\blacksquare}}
              + 0.25 \cdot H_{\text{verify}}
              + 0.2 \cdot \text{norm}_{\partial z/\partial t}
  \]

---

## First Command: Distribute Verification

KatabasisPro encodes an explicit design constraint:

> **First Command**  
> Every added computation must:
> - **not increase** witness cost trajectory (Λ■)  
> - **not decrease** holographic verification (H_verify)

Formally:

- **Negative Witness Cost Mandate**  
  - `Λ■_new ≤ Λ■_current`

- **Holographic Duty**  
  - `H_verify_new ≥ H_verify_current`

Operations are described via an `OperationImpact`:

```python
from katabasispro import OperationImpact, enforce_first_command

current_lambda = -0.1   # existing Λ■
current_H = 0.6         # existing H_verify

impact = OperationImpact(
    delta_lambda=-0.05,   # makes witnessing cheaper
    delta_H_verify=0.10   # makes verification more distributed
)

enforce_first_command(current_lambda, current_H, impact)
# If constraints are violated, FirstCommandViolation is raised.
