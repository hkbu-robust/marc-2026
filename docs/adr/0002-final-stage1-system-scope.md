# 0002: The final Stage 1 system: what was removed, what replaced it, and why

- **Status:** Accepted. Describes `final-system` after #110-#121, #133 (2026-10-08).
- **Issue:** #125 (part of #108). **Partly supersedes:** ADR 0001 (see the end).

## Context

A review of the frozen qualifier system (`~/marc-2026-work/experiments/ablations/09-pipeline-review/report.html`,
2026-10-07) found that part of its public score came from rules fitted to the 33 public
practice rounds, which are also the rounds the paper reports:

- **Proxies:** five of 19 object rounds submitted another object's position as the target.
- **Hard-coded answers:** fields the command never states, e.g. `police_car`, park ⇒ `beside`, road ⇒ taxi.
- **Fitted constants and tables:** per-class "next to" tables, a 400 px² umbrella threshold, umbrella/pencil-case-only detection passes.
- **Fail-closed parser:** an unparsed command submitted nothing.

Most of these arrived in one commit (`6d3691d`, PR #73), during which the practice score went
from 61 to 96. The organisers also warn that the scored scenario may use a different background
(`faq.md`: "do not hard-code assumptions tied to the practice scene's layout").

## Decision

Option C (#108): a scoped clean-up. The paper describes only the final system and its
ablations. Rules that encode knowledge of the public answers are deleted, not flagged off.
Fields the scorer needs but the command omits are inferred from the scene or given a generic
default, and the source is logged. Answer-key conventions that no general rule reproduces
are accepted as losses and disclosed.

**Explicitly out:**
- **lf03:** its tumbler is hidden under a lying person; the only rule that scores it does so
  through a false positive.
- **Accepting the bare-ID anchor re-pick reply:** it kept a wrong Qwen target on hf07.
- Retraining and new data.

### The final pipeline (one round)

```text
run_stage1_round(command)
  read_command                       Qwen under a JSON schema; lenient rules on any failure (#114)
  detect                             YOLO full frame, all cameras
  refine_detection_round             zoomed full-frame pass for small target classes (#115, flag on)
                                     + landmark-crop pass (960 px, conf 0.05, every class)
  add_hypotheses                     occlusion hypothesis (#117, flag OFF)
  select
    objects: rank_object_candidates (geometry, world metres)
             -> parking_space or implied-anchor command: geometry only, no Qwen (#120 b)
             -> else Qwen pick, then guards: support_geometry_guard, weak_override_guard,
                scene_evidence_guard (#119); infer_anchor for commands naming no landmark (#113)
    people:  select_sar_candidate: Qwen pick, then pose/support/place/landmark geometry
             overrides; place guard with pose/support/situation eligibility (#118)
  project                            calibrated ground surface when the camera's pose fingerprint
                                     matches, else the flat plane (#112); landmark footprint prior
                                     (#112); people and raised placements on the flat plane;
                                     SAR anchor by ground distance when no landmark is named (#121)
  situation (SAR)                    R03(detected pose of the selected person, anchor class) (#114)
  submit                             always: fallback_grounding when no grounding (#114)
```

Qwen is warmed before the agent registers, including one schema-constrained parse (#133,
#114). The static-scene cache is off by default (#111).

### Removed (no flag)

| Rule | Ticket |
|---|---|
| police_car +0.5 anchor bonus; parking "prefer the police-car pair" | #113 |
| park ⇒ `beside`; road ⇒ taxi camera; taxi excluded from parked vehicles | #113 |
| Static-scene cache on by default; post-select YOLO refresh; dead point hook; `PROSTRATE_CONTACT_INSET` | #111 |
| Landmark anchors projected from the 75 %-height box pixel onto one flat plane per camera (replaced by calibrated ground + footprint prior) | #112 |
| SAR anchor without a named landmark = the pixel-nearest landmark of any class (replaced by ground distance) | #121 |
| SAR place guard choosing among all people in the named place, ignoring pose, support and situation | #118 |
| Umbrella/pencil-case-only full-frame pass (1600/1280 px, conf 0.005) and per-class crop settings | #115 |
| Landmark-context proxy (target box := landmark box, conf 0.0025) | #116 |
| `OCCLUSION_PROXY_MAX_CONF` forcing of ≤ 0.003 candidates | #116 |
| `occluded_geometry` + contained-occlusion guard (umbrella + trash can, ≤ 400 px²) | #116 |
| Low-confidence guard (geometry wins when its target is < 0.02) | #116 |
| `hidden_umbrella`, `clipped_target`, `occluded_target` projection box swaps | #116 |
| Parser: per-landmark-class "next to" and "propped/leaning" tables; end-of-sentence "behind"/"there" rule; implicit `on` for placements at seating | #114 |
| Parser-derived SAR situation in the submission; fail-closed parsing ("skip submit") | #114 |

### Added, with flags (default) and evidence

| Mechanism | Flag (default) | Evidence |
|---|---|---|
| Calibrated ground per camera, gated by a pose fingerprint | `MARC_GROUND_CALIBRATION` (on) | replay +1.13 on #88 runs; held-out odd-scene anchor credit 0.793 → 0.862 (#112) |
| Landmark footprint prior (fitted on held-out even scenes) | `MARC_LANDMARK_FOOTPRINT` (on) | part of the +1.13; the calibration-free share (footprint only) is +0.13 (#112) |
| Implied anchor from the scene (nearest implied-class object, world metres) | — | #113 |
| SAR place-guard eligibility (pose, support, situation) | — | +0.36 replay (#118) |
| Object scene-evidence camera guard | — | hf08 (#119) |
| SAR anchor by ground distance | — | sar06 anchor 0.27 → 0.10 m (#121) |
| Zoomed full-frame pass for small target classes (data-derived list) | `MARC_TILED_REFINE` (on) | held-out end to end neutral (−0.0002 ± 0.007; +0.047 on detector misses) (#115) |
| Occlusion hypothesis for hidden targets | `MARC_OCCLUSION_HYPOTHESIS` (**off**) | failed its go/no-go: 4.5 % false fires (limit 2 %) (#117) |
| Implied-anchor commands decided by geometry | `MARC_IMPLIED_ANCHOR_GEOMETRY` (on) | hf07 35 → 75.7 (#120 option b) |
| Qwen command reader under a JSON schema, lenient rule fallback | `MARC_COMMAND_PARSER` (qwen) | blind test set: no-submission 33/60 → 0; target 23 → 55/56 (#114) |
| SAR situation from R03 on the detected pose | — | 138/140 on #88 SAR rounds (#114) |
| Class-first ranking: rank only the class the command names while any box of it exists | `MARC_CLASS_FIRST` (**off**) | failed its go/no-go: end to end on the 433 held-out cases with Qwen, no relation other than `on`/`above` gains a gold pick (+0, credit −0.001 [−0.011, +0.009]); the +18 gold picks are the `on`/`above` label artefact and carry no credit (#135) |
| Cross-class guard: Qwen's board and pick stay in the named class when one exists | `MARC_CROSS_CLASS_GUARD` (**off**) | same arm; 61/433 Qwen picks were cross-class, 5 within 1 m (#135) |
| Stated person relation is submitted, never filled in | `MARC_PERSON_RELATION` (on) | the scorer scores landmark and relation on any person problem that carries a relation; public SAR problems carry none, so 0 points there (#135) |
| Bounded pair likelihood for directional and near/beside ranking | `MARC_PAIR_LIKELIHOOD` (**off**) | failed its go/no-go on the parsed relation: all 433 +0.0004 [−0.012, +0.013]; right +0.052, behind −0.041, front +0.018, left +0.004 (#135) |
| Organisers' situation table and pose fallback, with their support names | — | their `situation_rules.yaml` and `people_poses.yaml`; the public labels do not change (#135) |
| Robustness: registration retry, `restart: on-failure`, drop an answer once a later round has started, any camera frame name in the TF filter, zoom/crop passes skipped under 4 s left, the named landmark as the anchor of a sentinel round | — | insurance against rare total-loss modes (#135) |

### Decisions informed by public rounds (disclosed)

- **#119 scene-evidence guard:** motivated by hf08.
- **#120 option (b), geometry for implied-anchor commands:** motivated by hf07. No held-out
  evidence exists, because every held-out command names its landmark.
- **#115 zoomed pass on by default:** held-out evidence was neutral; the team turned it on after
  a public smoke run (lf08/hf08 recovered, no round lost).
- **#114 phrase conventions:** where a phrase's label is not obvious, follow the organisers'
  labelling in their published problems: "next to" and "leaning against" → `near`,
  "propped against" → `beside`, no relation word → `near`. One rule per phrase, never per class.

## Consequences

- **Public score:** 94.92 ± 1.18 (frozen system, #88) → 87.55 in the last smoke run of the final
  system (one run; #128 measures 10). The losses that remain, and why:
  - **Hidden targets:** lf06, hf06 (umbrella under a lying person), lf07, hf02 (pencil case
    under a person on a table), lf03 (tumbler). Nothing visible is there; each keeps type,
    landmark and relation (35).
  - **Answer-key conventions:** lf09 and hf07 anchor to the nearest car (the SUV); the key
    wants the police car. hf04 has no relation word; the key wants `beside`. hf01 "next to";
    the key wants `beside` there but `near` in 4 of 5 public uses.
- **Per-site calibration:** the ground model applies only to cameras whose pose and ground
  height match the calibrated ones. On a different background only the footprint prior
  (+0.13) remains.
- **People and raised placements** stay on the flat plane: calibrating people flipped 7 public
  SAR rounds, and the support heights were set against the plane (#112).
- **Known limitations:**
  - The anchor re-pick (`TARGET_ID|ANCHOR_ID`) is a format the LoRA never saw. It is reachable
    only with `MARC_IMPLIED_ANCHOR_GEOMETRY=0` (#120).
  - Toppled-bicycle anchors are worse under the footprint prior (#112).
  - Low-confidence detections still vary between runs (#123).
  - On a shared development GPU, CUDA out-of-memory skips the zoomed pass. It now retries
    once, logs the skip, and the regression gate fails such runs.
  - The late-answer drop keys on a newer mission, not on the local clock, because the round limit
    is simulation time and the clock is wall time (#135).
  - Wall-clock round time is 18-21 s (simulation time p95 4 s, limit 30 s simulation). The
    Qwen budget uses wall clock, so on a slower judging GPU Qwen may be skipped late in a
    round (geometry answers); the 3.5 s camera-discovery settle per round is the first thing
    to trim if that shows up.
  - Stage 2 shares the Stage 1 grounding; an unreadable *bring* command still skips the
    Stage 2 submission (no sentinel navigation target).
- **The paper must change** where it contradicts this record:
  - Guards override Qwen, so it is not true that Qwen alone selects with guards as fallback.
  - Scene cues infer anchors; they do not just break ties.
  - Projection now uses calibrated ground and a footprint prior.
  - There are no proxies.
  - The headline is a repeated-run mean, not best-of-14.
  - The SAR situation matrix is the organisers' R03, not team-authored.

## Relation to ADR 0001

ADR 0001 (Qwen verifies a deterministic geometry ranking) still holds in structure: geometry
always runs and is both base answer and safety net. It is superseded in these parts:

- **Object guards:** "contained occlusion" and "low confidence" no longer exist (#116). The
  remaining object guards are support geometry, weak override and scene evidence (#119).
- **Geometry-only routes:** commands with an implied anchor join `parking_space` (#120 b).
- **Parsing:** never fails closed, so a round always submits (#114).
- **Line references:** ADR 0001's line numbers were for `1450aa9`; it now names functions instead.
