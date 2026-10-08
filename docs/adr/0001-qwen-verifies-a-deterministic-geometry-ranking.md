# 0001: Qwen verifies a deterministic geometry ranking

- **Status:** Accepted, **partly superseded by [ADR 0002](0002-final-stage1-system-scope.md)**
  (2026-10-08). Describes the qualifier system at `1450aa9`; the guard list below is updated
  to the final system, and the line references are replaced by function names.
- **Issue:** #85 (part of #92)

## Context

A Stage 1 round needs one target candidate (and an anchor) out of many detector boxes. Two selectors exist:

- the **geometry ranking**: deterministic scoring of target-anchor pairs by class, relation and image geometry;
- the **Qwen verifier**: the fine-tuned Qwen3-VL model, served by vLLM, which reads a candidate board and names one candidate ID.

Qwen is not a fallback for the geometry ranking. The reverse is true: the geometry ranking always runs, and it is both the base answer and the safety net for Qwen.

## Decision

```mermaid
flowchart TD
    C[Detector candidates] --> G["Geometry ranking<br/>answer G, always computed"]
    G --> V{"Verifier configured?<br/>MARC_SAR_DETECTOR"}
    V -- no --> SG["Submit G<br/>object: object_relation, log reason no_verifier<br/>SAR: deterministic_fallback, verifier_not_configured"]
    V -- yes --> P{"Object path:<br/>scene_hint = parking_space<br/>and remote backend?"}
    P -- yes --> SP["Submit G<br/>reason: parking_geometry"]
    P -- no --> Q["Qwen picks Q<br/>from the candidate board"]
    Q --> F{"Call failed?<br/>timeout, connection, grammar, unknown ID"}
    F -- yes --> SF["Submit G<br/>= Qwen fallback"]
    F -- no --> R{"A guard rule<br/>rejects Q?"}
    R -- yes --> SO["Submit G<br/>= geometry override"]
    R -- no --> SQ["Submit Q<br/>= Qwen choice"]

    classDef geo fill:#e8eef7,stroke:#4a6fa5
    classDef qwen fill:#eaf5ea,stroke:#4a8a4a
    class SG,SP,SF,SO geo
    class SQ qwen
```

Every round ends in exactly one of three outcomes:

| Outcome | Meaning |
|---|---|
| **Qwen choice** | Qwen answered and no guard rejected it. |
| **Qwen fallback** | The Qwen call failed. The geometry answer is submitted. |
| **Geometry override** | Qwen answered, but a hand-written guard replaced its pick with the geometry answer. |

The guard rules:

- **Object path** (`object_selection._select_object_candidate`): support geometry, weak override,
  scene evidence (#119). The qualifier's "contained occlusion" and "low confidence" guards were
  deleted with the proxy chain (#116, ADR 0002).
- **Person (SAR) path** (`sar_selection.select_sar_candidate`): `pose_geometry_override`,
  `support_geometry_override`, `place_geometry_override` (with pose/support/situation
  eligibility since #118), `landmark_geometry_override`.

Besides `parking_space`, commands that name no landmark but imply one also skip Qwen in the
final system (#120 option b, ADR 0002).

The SAR path also skips Qwen when no verifier is configured and the shortlist holds at most one person (`select_sar_candidate`).

## Consequences

- **The pipeline always submits an answer when Qwen is down.** A dead vLLM server degrades the system to deterministic ranking without stopping the round.
- **That degradation was invisible in the object path.** On a failed call, the object path returned the geometry answer unchanged, so the round is recorded as `object_relation`, the same label as a run without Qwen. The SAR path already records `deterministic_fallback` with `verifier_failed:<Error>`. #85 makes the object path record `deterministic_fallback` too, with the reason and the step that failed, so that the selector arms in #89 and #90 do not undercount fallbacks. The first live run with this logging found 4 fallbacks in 33 rounds: two vLLM cold-start timeouts in rounds 1–2, and two anchor re-pick replies that gave only a target ID.
- **Geometry overrides were not counted.** They appear only as log text. #85 records them as structured events, with every guard when several fire in one round.
- **Comparing "Qwen on" with "Qwen off" measures more than the selector.** Turning the verifier on also widens the candidate classes, adds a scene-description pass, a class screen and an anchor re-pick (#90). Base Qwen against fine-tuned Qwen isolates the fine-tuning, because both use the same path.
