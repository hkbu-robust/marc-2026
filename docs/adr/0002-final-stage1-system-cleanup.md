# ADR 0002: Final Stage 1 System Scope & Anti-Overfitting Cleanup

- **Status:** Accepted — frozen as tag `final-system-v1` (commit `d75d080` on `main`).
- **Partly supersedes:** [ADR 0001](0001-stage1-vlm-selector-and-geometry-guards.md) (updates active guard list and geometry-only routes).
- *Note: References to earlier qualifier iterations refer to the team's internal development history prior to final submission.*

## Summary

Before final competition submission and paper publication, we audited our Stage 1 perception pipeline for any heuristics overfitted to the 33 public practice rounds and **permanently deleted all such rules**—replacing them with general, principled mechanisms evaluated on held-out synthetic scenes and blind paraphrases.

**Both our IEEE MetaCom 2026 paper (`87.25 ± 0.71` across 10 cold runs of the 33 public rounds) and our MARC 2026 Highest Stage 1 Score / Excellence Award come strictly from this cleaned `final-system-v1` system.**

## Context

An internal pipeline audit of an earlier qualifier prototype revealed that part of its public-practice score stemmed from heuristics tied to the public `marc2026_chungmu` practice rounds rather than general visual grounding:

- **Proxy substitutions:** submitting another visible object's bounding box when the true target was occluded.
- **Hard-coded scene/class priors:** bonuses or defaults never stated in the user's command (e.g., favoring `police_car`, mapping `park` → `beside`, or `road` → `taxi`).
- **Per-class fitted thresholds:** class-specific pixel-area gates (e.g., a `400 px²` umbrella threshold) and narrow single-class detection passes.
- **Fail-closed command parsing:** skipping submission whenever phrasing fell outside hand-written grammar rules.

Because competition evaluation may run on unseen layouts and camera placements, relying on practice-fitted rules compromises both scientific validity and real competition generalization.

## Decision

We executed a strict cleanup across the Stage 1 stack:

1. **Delete, do not hide:** Every rule encoding knowledge of public practice answers was deleted from the codebase rather than hidden behind a flag.
2. **Replace with principled general mechanisms:** Missing or noisy signals are handled via calibrated camera/ground geometry, schema-constrained VLM command parsing, data-derived small-object refinement, and the organizers' official `R03` SAR pose × support matrix.
3. **Disclose residual losses transparently:** Unobservable targets (e.g., objects completely occluded beneath a lying person) and underdetermined phrasing conventions are accepted as honest losses and itemized in the paper's error ledger (Table IV: `12.75` pts lost per round, of which `9.09` pts come from invisible targets and `2.06` pts from underdetermined phrasing).

### Final Stage 1 Pipeline (Per Round)

```text
run_stage1_round(command)
  read_command             Qwen3-VL-8B under strict JSON schema (temp 0, seed 0);
                           lenient rule parser fallback (never raises)
  detect                   34-class YOLO26x full-frame (1920 px) across all 6 CCTV views
  refine_detection_round   2x zoomed pass for 10 data-derived small classes (MARC_TILED_REFINE=1)
                           + 960 px landmark crop pass (conf 0.05, all classes)
  select
    objects:               rank target-anchor pairs in world metres
                           -> parking_space or implied-anchor command: geometry only
                           -> else LoRA Qwen3-VL-8B finite-ID pick + guards
                              (support_geometry, weak_override, scene_evidence)
    people (SAR):          LoRA Qwen3-VL-8B pick + pose/support/place/landmark guards
  project                  camera ray (K, R, C) -> pose-gated kNN ground surface (k=12)
                           when camera pose matches calibrated site, else flat ground plane;
                           class landmark footprint prior fitted on even held-out scenes
  situation (SAR)          organizers' R03 pose x support matrix on selected person's pose
  submit                   always submits (fallback grounding if no valid projection)
```

### Removed Heuristics (Deleted Permanently)

- `police_car` anchor bonus and parking-lot police-car pair preference.
- Hard-coded scene mappings (`park` → `beside`, `road` → taxi camera, taxi exclusion from parked vehicles).
- Static-scene cache enabled by default and post-selection YOLO refresh.
- All occlusion proxy chains (`landmark_context` box swaps, `OCCLUSION_PROXY_MAX_CONF`, `occluded_geometry`, contained-occlusion `≤ 400 px²` umbrella/trash-can guard, `hidden_umbrella` / `clipped_target` / `occluded_target` swaps).
- Umbrella/pencil-case-only high-res detection passes and per-class crop tuning.
- Per-landmark-class parser tables, end-of-sentence `behind`/`there` heuristics, parser-derived SAR situations, and fail-closed parser aborts.

### Principled Additions & Held-Out Evidence

| Mechanism | Flag (Default) | Validation / Held-Out Evidence |
|---|---|---|
| Pose-gated kNN calibrated ground surface (`k=12`) + landmark footprint prior (fitted on even held-out scenes) | `MARC_GROUND_CALIBRATION` (`1`), `MARC_LANDMARK_FOOTPRINT` (`1`) | Held-out odd-scene target median error `0.45 m → 0.07 m`; anchor credit `0.793 → 0.862`; public score `84.90 → 87.25` (Paper Table VI) |
| Schema-constrained Qwen3-VL-8B command reader + lenient rule fallback | `MARC_COMMAND_PARSER` (`qwen`) | Blind 60-paraphrase set: no-submission rounds `33/60 → 0`; target `55/56`, landmark `54/56`, relation `50/55` (vs. `48/56`, `44/56`, `41/55` rule fallback) |
| Data-derived 2× small-object zoom pass (10 classes `< 5×10⁻⁴` frame area) | `MARC_TILED_REFINE` (`1`) | Held-out `< 8 px` object recall `70% → 80%` (`147/210 → 169/210`); overall small-class recall `85% → 90%` (Paper Table II) |
| Organizers' `R03` pose × support SAR situation table | always on | Directly aligns SAR situation labels with detected pose and support class |
| SAR place-guard eligibility (pose + support + situation) & object scene-evidence guard | always on | Eliminates invalid cross-pose and cross-camera overrides |
| Candidate mechanisms evaluated & left **off** after failing held-out go/no-go thresholds | `MARC_OCCLUSION_HYPOTHESIS=0`, `MARC_CLASS_FIRST=0`, `MARC_CROSS_CLASS_GUARD=0`, `MARC_PAIR_LIKELIHOOD=0` | Retained behind off-by-default flags only for ablation reproducibility |

## Consequences & Known Limitations

- **Reproducible paper headline:** `87.25 ± 0.71` across 10 cold starts of the 33 public rounds (median `87.54`, 8/10 runs within `0.01`, sim-time p95 `4.00 s` vs. `30 s` limit).
- **Safe transfer to unseen sites:** The calibrated ground surface activates only when a camera's pose matches the calibrated site within `1 cm` / `0.1°`; on unseen layouts, projection falls back automatically to the flat ground plane plus the general landmark footprint prior.
- **Honest limitations:** Fully occluded targets under a lying person cannot be seen in CCTV (`9.09` of `12.75` lost points); toppled bicycles and raised anchors remain harder to project from monocular views without depth; generalization to wholly unseen campus geometries awaits multi-scene training data.
