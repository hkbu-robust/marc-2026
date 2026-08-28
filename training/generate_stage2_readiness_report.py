"""Generate the offline Stage 2 readiness report and traceable evidence cards."""

from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from training.evaluate_cctv_detector import _frames, _iou


COLORS = {"match": "#27c47d", "false_candidate": "#ff5964", "miss": "#ffbd59"}
LOW_CONFIDENCE = 0.05


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _collect(checkpoint: Path, names: list[str], frames: list[dict], settings: dict) -> dict:
    from ultralytics import YOLO

    model = YOLO(checkpoint)
    model_names = dict(model.names)
    class_ids = {str(name): int(class_id) for class_id, name in model_names.items()}
    missing = [name for name in names if name not in class_ids]
    if missing:
        raise ValueError(f"checkpoint lacks report classes: {missing}")
    output = {frame["image"].name: {} for frame in frames}
    paths = [str(frame["image"]) for frame in frames]
    for target in names:
        for offset in range(0, len(paths), 4):
            results = model.predict(
                paths[offset:offset + 4], batch=4, device="0", imgsz=settings["imgsz"],
                conf=0.001, iou=settings["nms_iou"], max_det=20,
                end2end=settings["end2end"], classes=[class_ids[target]],
                quantize=settings["quantize"], verbose=False,
            )
            for frame, result in zip(frames[offset:offset + 4], results):
                output[frame["image"].name][target] = [
                    {
                        "class_name": target,
                        "confidence": float(confidence),
                        "xyxy": [float(value) for value in box],
                    }
                    for box, confidence in zip(result.boxes.xyxy.cpu(), result.boxes.conf.cpu())
                ]
    del model
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def _gt(frame: dict, class_id: int) -> list[list[float]]:
    height, width = frame["shape"]
    return [
        [(cx - bw / 2) * width, (cy - bh / 2) * height,
         (cx + bw / 2) * width, (cy + bh / 2) * height]
        for label, cx, cy, bw, bh in frame["labels"] if label == class_id
    ]


def _view(candidates: list[dict], gt: list[list[float]], confidence: float, limit: int) -> dict:
    selected = sorted(
        (item for item in candidates if item["confidence"] >= confidence),
        key=lambda item: item["confidence"], reverse=True,
    )[:limit]
    unmatched = set(range(len(gt)))
    rendered = []
    for rank, candidate in enumerate(selected, 1):
        choices = [(index, _iou(candidate["xyxy"], gt[index])) for index in unmatched]
        index, overlap = max(choices, key=lambda item: item[1], default=(-1, 0.0))
        status = "match" if overlap >= 0.5 else "false_candidate"
        if status == "match":
            unmatched.remove(index)
        rendered.append({**candidate, "rank": rank, "status": status, "iou": overlap})
    return {
        "candidates": rendered,
        "matched": len(gt) - len(unmatched),
        "matched_gt": [gt[index] for index in range(len(gt)) if index not in unmatched],
        "false_candidates": sum(item["status"] == "false_candidate" for item in rendered),
        "missed_gt": [gt[index] for index in sorted(unmatched)],
    }


def _records(names: list[str], frames: list[dict], stage1: dict, stage2: dict, settings: dict) -> list[dict]:
    records = []
    for frame in frames:
        for class_id, target in enumerate(names):
            gt = _gt(frame, class_id)
            records.append({
                "frame": frame,
                "target": target,
                "class_id": class_id,
                "gt": gt,
                "stage1": _view(stage1[frame["image"].name][target], gt, settings["confidence"], settings["candidate_limit"]),
                "stage2": _view(stage2[frame["image"].name][target], gt, settings["confidence"], settings["candidate_limit"]),
                "stage2_low": _view(stage2[frame["image"].name][target], gt, LOW_CONFIDENCE, settings["candidate_limit"]),
            })
    return records


def _short_side(box: list[float]) -> float:
    return min(box[2] - box[0], box[3] - box[1])


def _pick(records: list[dict], names: list[str]) -> tuple[list[dict], dict]:
    selected: dict[tuple[str, str], dict] = {}
    rules = {}

    def add(rule: str, candidates: list[dict], key):
        choice = max(candidates, key=key) if candidates else None
        rules[rule] = None if choice is None else {"image": choice["frame"]["image"].name, "target": choice["target"]}
        if choice:
            sample = selected.setdefault((choice["frame"]["image"].name, choice["target"]), choice)
            sample.setdefault("reasons", []).append(rule)

    for name in names:
        add(
            f"class:{name}",
            [item for item in records if item["target"] == name and item["stage2"]["matched"]],
            lambda item: (-item["stage2"]["false_candidates"], item["stage2"]["matched"],
                          max((box["confidence"] for box in item["stage2"]["candidates"]), default=0)),
        )
    add("crowded tiny-object success", [
        item for item in records
        if any(_short_side(box) < 16 for box in item["stage2"]["matched_gt"])
    ], lambda item: len(item["frame"]["labels"]))
    add("persistent tiny-object miss", [
        item for item in records
        if any(_short_side(box) < 16 for box in item["stage2_low"]["missed_gt"])
    ], lambda item: -min(_short_side(box) for box in item["stage2_low"]["missed_gt"]))
    add("recovered only below selected confidence", [
        item for item in records if item["gt"] and not item["stage2"]["matched"]
        and item["stage2_low"]["matched"]
    ], lambda item: item["stage2_low"]["matched"])
    add("highest false-candidate load", records,
        lambda item: item["stage2"]["false_candidates"])
    add("edge-of-frame target", [
        item for item in records if any(
            box[0] <= 3 or box[1] <= 3 or box[2] >= 637 or box[3] >= 477
            for box in item["stage2"]["matched_gt"]
        )
    ], lambda item: item["stage2"]["matched"])
    add("empty processed-label frame", [
        item for item in records if not item["frame"]["labels"]
    ], lambda item: item["stage2"]["false_candidates"])
    add("clean success", [
        item for item in records if item["stage2"]["matched"] and not item["stage2"]["false_candidates"]
    ], lambda item: item["stage2"]["matched"])
    add("Stage 2 improvement over Stage 1 seed", [
        item for item in records if item["stage2"]["matched"] > item["stage1"]["matched"]
    ], lambda item: item["stage2"]["matched"] - item["stage1"]["matched"])
    add("Stage 2 regression versus Stage 1 seed", [
        item for item in records if item["stage1"]["matched"] > item["stage2"]["matched"]
    ], lambda item: item["stage1"]["matched"] - item["stage2"]["matched"])
    return list(selected.values()), rules


def _pick_blind(records: list[dict], names: list[str], excluded_images: set[str]) -> tuple[list[dict], dict]:
    """Pick a disjoint per-class set, mixing risk-heavy and clean examples."""
    samples, rules = [], {}
    for index, name in enumerate(names):
        candidates = [
            item for item in records
            if item["target"] == name
            and item["frame"]["image"].name not in excluded_images
        ]
        if index < 5:
            choice = max(candidates, key=lambda item: (
                item["stage2"]["false_candidates"] + len(item["stage2"]["missed_gt"]),
                len(item["gt"]),
            ))
            rule = "disjoint per-class risk-oriented card"
        else:
            clean = [item for item in candidates if item["stage2"]["matched"] and not item["stage2"]["false_candidates"]]
            choice = max(clean or candidates, key=lambda item: item["stage2"]["matched"])
            rule = "disjoint per-class clean-control card"
        choice = {**choice, "reasons": [f"{rule}: {name}"]}
        samples.append(choice)
        rules[name] = {"image": choice["frame"]["image"].name, "rule": rule}
    return samples, rules


def _draw_boxes(image: Image.Image, candidates: list[dict], *, blind: bool = False) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for candidate in candidates:
        color = "#48cae4" if blind else COLORS[candidate["status"]]
        draw.rectangle(candidate["xyxy"], outline=color, width=3)
        label = f"#{candidate['rank']} {candidate['confidence']:.2f}"
        if not blind:
            label += f" {candidate['status']}"
        draw.text((candidate["xyxy"][0] + 3, max(2, candidate["xyxy"][1] - 15)), label, fill=color)
    return output


def _truth(image: Image.Image, gt: list[list[float]]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for box in gt:
        draw.rectangle(box, outline=COLORS["miss"], width=3)
    return output


def _strip(panels: list[tuple[str, Image.Image]]) -> Image.Image:
    width, height = panels[0][1].size
    canvas = Image.new("RGB", (width * len(panels), height + 28), "#0c1424")
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        canvas.paste(panel, (index * width, 28))
        draw.text((index * width + 8, 7), title, fill="white")
    return canvas


def _render_samples(samples: list[dict], assets: Path, settings: dict) -> list[dict]:
    manifest = []
    for index, sample in enumerate(samples, 1):
        raw = Image.open(sample["frame"]["image"]).convert("RGB")
        evidence_name = f"evidence-{index:02d}.jpg"
        _strip([
            ("Raw RGB", raw),
            (f"Stage 1 seed @ {settings['confidence']:.2f}", _draw_boxes(raw, sample["stage1"]["candidates"])),
            (f"Stage 2 selected @ {settings['confidence']:.2f}", _draw_boxes(raw, sample["stage2"]["candidates"])),
            (f"Stage 2 lower view @ {LOW_CONFIDENCE:.2f}", _draw_boxes(raw, sample["stage2_low"]["candidates"])),
            ("Processed ground truth", _truth(raw, sample["gt"])),
        ]).save(assets / evidence_name, quality=92)
        zoom_name = None
        if sample["gt"] and min(_short_side(box) for box in sample["gt"]) < 32:
            box = min(sample["gt"], key=_short_side)
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            crop = raw.crop((max(0, cx - 48), max(0, cy - 48), min(raw.width, cx + 48), min(raw.height, cy + 48)))
            zoom_name = f"evidence-{index:02d}-zoom.jpg"
            crop.resize((384, 384), Image.Resampling.NEAREST).save(assets / zoom_name, quality=95)
        manifest.append({
            "id": f"sample-{index:02d}",
            "image": sample["frame"]["image"].name,
            "source_image": str(sample["frame"]["image"]),
            "target": sample["target"],
            "reasons": sample["reasons"],
            "ground_truth_xyxy": sample["gt"],
            "stage1_candidates": sample["stage1"]["candidates"],
            "stage2_candidates": sample["stage2"]["candidates"],
            "stage2_low_candidates": sample["stage2_low"]["candidates"],
            "assets": {"evidence": evidence_name, "zoom": zoom_name},
            "settings": settings,
        })
    return manifest


def _render_blind(samples: list[dict], assets: Path, settings: dict) -> list[dict]:
    manifest = []
    for index, sample in enumerate(samples, 1):
        raw = Image.open(sample["frame"]["image"]).convert("RGB")
        model_name = f"blind-{index:02d}-model.jpg"
        truth_name = f"blind-{index:02d}-truth.jpg"
        _draw_boxes(raw, sample["stage2"]["candidates"], blind=True).save(assets / model_name, quality=92)
        _truth(raw, sample["gt"]).save(assets / truth_name, quality=92)
        manifest.append({
            "id": f"blind-{index:02d}",
            "image": sample["frame"]["image"].name,
            "source_image": str(sample["frame"]["image"]),
            "target": sample["target"],
            "selection_rule": sample["reasons"][0],
            "ground_truth_xyxy": sample["gt"],
            "stage2_candidates": sample["stage2"]["candidates"],
            "assets": {"model": model_name, "truth": truth_name},
            "settings": settings,
        })
    return manifest


def _table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _metric(value) -> str:
    return "—" if value is None else f"{100 * float(value):.2f}%"


def _html(report: dict, runtime: dict, manifest: list[dict], blind_manifest: list[dict],
          rules: dict, assets_rel: str) -> str:
    settings = report["selection"]["runtime"]
    selected_name = Path(report["selection"]["checkpoint"]).name
    selected_sha = report["runtime_contract"]["checkpoint_sha256"]
    head_name = "one-to-one" if settings["end2end"] else "one-to-many/NMS"
    nms_text = "n/a" if settings["nms_iou"] is None else f"{settings['nms_iou']:.2f}"
    operational = next(item for item in report["runtime_size_comparisons"] if item["imgsz"] == settings["imgsz"])
    selected_ap = next(item for item in report["ap"] if item["end2end"] == settings["end2end"])
    worst_name, worst = min(operational["classes"].items(), key=lambda item: item[1]["recall"])
    screening_rows = [[Path(item["checkpoint"]).name, _metric(item["macro_f1"]), _metric(min(v["recall"] for v in item["classes"].values())), _metric(item["sizes"]["tiny"]["recall"]), "selected" if item["checkpoint"] == report["selection"]["checkpoint"] else ""] for item in report["screening"]]
    ap_by_name = {item["Class"]: item for item in selected_ap["classes"]}
    class_rows = [[name, _metric(values["precision"]), _metric(values["recall"]), _metric(values["f1"]), _metric(ap_by_name[name]["mAP50"]), _metric(ap_by_name[name]["mAP50-95"])] for name, values in operational["classes"].items()]
    size_rows = [[name, values["gt"], values["matched"], _metric(values["recall"]), operational["size_bucket_definitions"][name]] for name, values in operational["sizes"].items()]
    heads = []
    for end2end in (True, False):
        choices = [item for item in report["comparisons"] if item["end2end"] == end2end and item["confidence"] == 0.001 and item["candidate_limit"] == 20]
        item = max(choices, key=lambda value: value["macro_f1"])
        heads.append(["one-to-one" if end2end else "one-to-many/NMS", item["nms_iou"], _metric(item["macro_f1"]), _metric(item["recall"]), item["false_candidates"]])
    config_rows = [[item["confidence"], item["nms_iou"], item["candidate_limit"], _metric(item["macro_f1"]), _metric(min(v["recall"] for v in item["classes"].values())), item["false_candidates"]] for item in report["comparisons"] if item["end2end"] == settings["end2end"]]
    size_compare_rows = [[item["imgsz"], _metric(item["macro_f1"]), _metric(min(v["recall"] for v in item["classes"].values())), _metric(item["sizes"]["tiny"]["recall"]), _metric(item["recall"])] for item in report["runtime_size_comparisons"]]
    recall_rows = [[f"@{limit}", _metric(values["recall"]), _metric(values["macro_f1"]), values["false_candidates"]] for limit, values in report["target_conditioned_recall_at"].items()]
    evidence = "".join(f"<article class='sample'><h3>{html.escape(', '.join(item['reasons']))}</h3><p><b>{html.escape(item['target'])}</b> · {html.escape(item['image'])}</p><img src='{assets_rel}/{item['assets']['evidence']}' alt='Five-panel exact-target comparison'>{f'''<details><summary>Enlarged native-pixel target crop</summary><img class='zoom' src='{assets_rel}/{item['assets']['zoom']}' alt='Enlarged crop'></details>''' if item['assets']['zoom'] else ''}</article>" for item in manifest)
    blind = "".join(f"<article class='blind-card' data-card='{item['id']}'><h3>{item['id']} · requested target: {html.escape(item['target'])}</h3><img src='{assets_rel}/{item['assets']['model']}' alt='Blind exact-target candidates'><div class='review-grid'><label>Found?<select name='found'><option value=''>—</option><option>yes</option><option>no</option><option>ambiguous</option></select></label><label>Box accurate for depth/approach?<select name='box'><option value=''>—</option><option>yes</option><option>no</option><option>ambiguous</option></select></label><label>False candidates unsafe?<select name='unsafe'><option value=''>—</option><option>yes</option><option>no</option><option>ambiguous</option></select></label><label>Overall<select name='overall'><option value=''>—</option><option>acceptable</option><option>concerning</option><option>ambiguous</option></select></label></div><label>Notes<textarea name='notes'></textarea></label><details><summary>Reveal processed ground truth</summary><img src='{assets_rel}/{item['assets']['truth']}' alt='Processed ground truth'></details></article>" for item in blind_manifest)
    limitations = "".join(f"<li>{html.escape(item)}</li>" for item in report["known_limitations"])
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Stage 2 provisional robot-view readiness</title><style>
:root{{--bg:#09111f;--panel:#111d30;--ink:#e8f0fa;--muted:#9fb1c8;--cyan:#48cae4;--amber:#ffbd59;--green:#27c47d;--red:#ff5964}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1240px;margin:auto;padding:28px}}h1,h2,h3{{line-height:1.15}}h2{{margin-top:48px;border-bottom:1px solid #29405e;padding-bottom:8px}}.banner{{border:2px solid var(--amber);background:#2a210e;padding:14px;border-radius:10px;font-weight:800}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}.metric,.sample,.blind-card,.flow{{background:var(--panel);border:1px solid #263a55;border-radius:12px;padding:16px}}.metric b{{display:block;font-size:24px;color:var(--cyan)}}.flow{{font-size:18px;text-align:center;color:var(--cyan)}}img{{display:block;width:100%;height:auto;border-radius:8px}}.zoom{{max-width:384px}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;background:var(--panel)}}th,td{{padding:8px 10px;border-bottom:1px solid #263a55;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}.sample,.blind-card{{margin:14px 0}}.review-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin:12px 0}}label{{display:grid;gap:4px;color:var(--muted)}}select,textarea,button{{background:#0a1424;color:var(--ink);border:1px solid #3a5578;border-radius:6px;padding:8px}}textarea{{min-height:70px}}button{{cursor:pointer;margin-right:8px}}.warning{{color:var(--amber)}}.pending{{color:var(--red);font-weight:800}}code{{color:var(--cyan)}}details{{margin-top:10px}}@media(max-width:700px){{main{{padding:14px}}}}
</style></head><body><main>
<div class='banner'>PROVISIONAL ROBOT-VIEW READINESS · nine classes only · human review pending</div>
<h1>Stage 2 near-field detector readiness</h1><p>Selected <code>{html.escape(selected_name)}</code>, {head_name}, confidence {settings['confidence']:.2f}, NMS IoU {nms_text}, top {settings['candidate_limit']}, {settings['imgsz']} px, {str(settings['quantize']).upper()}. The same validation split selected and reports this model.</p>
<div class='cards'><div class='metric'><span>Macro F1</span><b>{_metric(operational['macro_f1'])}</b></div><div class='metric'><span>Worst recall · {html.escape(worst_name)}</span><b>{_metric(worst['recall'])}</b></div><div class='metric'><span>Tiny recall</span><b>{_metric(operational['sizes']['tiny']['recall'])}</b></div><div class='metric'><span>Overall recall</span><b>{_metric(operational['recall'])}</b></div><div class='metric'><span>Warm batch-1 mean</span><b>{runtime['benchmark']['mean_ms']:.2f} ms</b></div><div class='metric'><span>Peak CUDA allocated</span><b>{runtime['benchmark']['peak_allocated_gib']:.3f} GiB</b></div></div>
<h2>Runtime path</h2><div class='flow'>chassis RGB → exact requested class → confidence-descending candidates → reveal-disc localization → depth/approach candidate</div><p>Errors and unsupported targets fail immediately; only <code>not_detected</code> may continue scanning.</p>
<h2>Training and selected-head diagnostics</h2><div class='cards'><img src='{assets_rel}/training-results.png' alt='Training curves'><img src='{assets_rel}/BoxPR_curve.png' alt='Selected checkpoint PR curve'><img src='{assets_rel}/BoxF1_curve.png' alt='Selected checkpoint F1 curve'></div>
<h2>Saved-checkpoint ranking</h2>{_table(['checkpoint','macro F1','worst recall','tiny recall','decision'],screening_rows)}
<h2>Per-class result</h2>{_table(['class','precision','recall','F1','AP50','AP50–95'],class_rows)}
<h2>Selected-checkpoint confusion matrices</h2><div class='cards'><img src='{assets_rel}/confusion_matrix.png' alt='Raw confusion matrix'><img src='{assets_rel}/confusion_matrix_normalized.png' alt='Normalized confusion matrix'></div><p>These plots were regenerated from the selected <code>{html.escape(selected_name)}</code> {head_name} head, not reused from the trainer-final checkpoint.</p>
<h2>Native-pixel size recall</h2>{_table(['bucket','GT','matched','recall','definition'],size_rows)}
<h2>Equal-budget head comparison</h2>{_table(['head','NMS IoU','macro F1','overall recall','false candidates'],heads)}
<h2>Confidence / NMS / candidate-limit sweep</h2>{_table(['confidence','NMS IoU','limit','macro F1','worst recall','false candidates'],config_rows)}
<h2>Resolution comparison</h2>{_table(['imgsz','macro F1','worst recall','tiny recall','overall recall'],size_compare_rows)}
<h2>Exact-target recall by candidate budget</h2>{_table(['budget','overall recall','macro F1','false candidates'],recall_rows)}
<h2>Absent-target and empty-label evidence</h2><p>{operational['target_requests']['absent']:,} exact-target requests had that target absent from processed labels; {operational['target_requests']['absent_with_candidates']} returned candidates ({operational['target_requests']['false_candidates_when_absent']} unmatched under IoU≥0.5). Across {operational['empty_label_frames']['frames']} empty processed-label frames, the selected run produced {operational['empty_label_frames']['false_candidates']} unmatched candidates. This does not establish physical absence or intentional-negative status.</p>
<h2>Exact before/after and failure evidence</h2><p>Every strip uses the same frame, requested class, {settings['imgsz']} px, NMS IoU {nms_text} and top-{settings['candidate_limit']} budget. Green is IoU≥0.5 match, red is unmatched candidate, amber is processed ground truth. The {LOW_CONFIDENCE:.2f} panel is explanatory only; {settings['confidence']:.2f} remains selected.</p>{evidence}
<h2>Blind target-conditioned review</h2><p class='pending' id='progress'>Human review pending: 0 / {len(blind_manifest)}</p><p>This disjoint set is not shown in the evidence section: five class cards are risk-oriented and four are clean controls. Prediction cards do not expose ground truth until opened. Review cannot change the numeric checkpoint selection, but may veto readiness.</p>{blind}<p><button id='export'>Export review JSON</button><button id='clear'>Clear local review</button></p>
<h2>Traceability</h2><p>Selected training SHA-256: <code>{selected_sha}</code>. Deployment SHA-256: <code>{runtime['weights_sha256']}</code>. Evaluator SHA-256: <code>{report['evaluator']['sha256']}</code>. Numeric inputs: <code>readiness_metrics.json</code>; sample decisions and boxes: <code>sample_manifest.json</code>. Report selection rules: <code>{html.escape(json.dumps(rules, sort_keys=True))}</code>.</p>
<h2>Final verdict</h2><p><b>Numeric gate: provisional pass.</b> Checkpoint selection and real-interface runtime verification completed. <span class='pending'>Human blind review is still pending, so integration approval is not complete.</span></p><ul>{limitations}</ul><p class='warning'>Do not commit, publish, merge, or claim final challenge readiness until Luka completes the blind review and approves the provisional bundle.</p>
<script>
const key='marc-stage2-readiness-review-v1', cards=[...document.querySelectorAll('.blind-card')];
function data(){{const out={{}};for(const card of cards){{out[card.dataset.card]={{}};for(const field of card.querySelectorAll('[name]'))out[card.dataset.card][field.name]=field.value;}}return out}}
function save(){{localStorage.setItem(key,JSON.stringify(data()));const done=cards.filter(c=>c.querySelector('[name=overall]').value).length;document.querySelector('#progress').textContent=`Human review ${{done===cards.length?'complete':'pending'}}: ${{done}} / ${{cards.length}}`;}}
const stored=JSON.parse(localStorage.getItem(key)||'{{}}');for(const card of cards)for(const field of card.querySelectorAll('[name]'))field.value=stored[card.dataset.card]?.[field.name]||'';document.addEventListener('input',save);document.addEventListener('change',save);save();
document.querySelector('#export').onclick=()=>{{const reviews=data(),done=cards.filter(c=>c.querySelector('[name=overall]').value).length,payload={{schema:'marc-stage2-human-review-v1',status:done===cards.length?'complete':'in_progress',exported_at:new Date().toISOString(),selected_checkpoint:{json.dumps(selected_name)},selected_checkpoint_sha256:{json.dumps(selected_sha)},fixed_settings:{json.dumps(settings)},reviews}};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}}));a.download='human-review-stage2-nearfield.json';a.click();URL.revokeObjectURL(a.href)}};
document.querySelector('#clear').onclick=()=>{{if(confirm('Clear browser-local review progress?')){{localStorage.removeItem(key);location.reload()}}}};
</script></main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--runtime-verification", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--stage1", required=True, type=Path)
    parser.add_argument("--stage2", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    report = json.loads(args.evaluation.read_text(encoding="utf-8"))
    runtime = json.loads(args.runtime_verification.read_text(encoding="utf-8"))
    names, frames = _frames(args.data)
    settings = report["selection"]["runtime"]
    stage1 = _collect(args.stage1, names, frames, settings)
    stage2 = _collect(args.stage2, names, frames, settings)
    records = _records(names, frames, stage1, stage2, settings)
    samples, rules = _pick(records, names)
    excluded_images = {item["frame"]["image"].name for item in samples}
    blind_samples, blind_rules = _pick_blind(records, names, excluded_images)

    assets = args.output_dir / "assets" / "stage2-nearfield-detector-readiness"
    assets.mkdir(parents=True, exist_ok=True)
    for pattern in ("evidence-*.jpg", "blind-*.jpg"):
        for stale in assets.glob(pattern):
            stale.unlink()
    selected_plot_dir = Path(next(item["save_dir"] for item in report["ap"] if item["end2end"] == settings["end2end"]))
    for source, name in [
        (args.run_dir / "results.png", "training-results.png"),
        (selected_plot_dir / "BoxPR_curve.png", "BoxPR_curve.png"),
        (selected_plot_dir / "BoxF1_curve.png", "BoxF1_curve.png"),
        (selected_plot_dir / "confusion_matrix.png", "confusion_matrix.png"),
        (selected_plot_dir / "confusion_matrix_normalized.png", "confusion_matrix_normalized.png"),
    ]:
        shutil.copy2(source, assets / name)
    manifest = _render_samples(samples, assets, settings)
    blind_manifest = _render_blind(blind_samples, assets, settings)
    sample_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_selection_rules": rules,
        "blind_selection_rules": blind_rules,
        "stage1": {"path": str(args.stage1.resolve()), "sha256": _sha256(args.stage1)},
        "stage2": {"path": str(args.stage2.resolve()), "sha256": _sha256(args.stage2)},
        "settings": settings,
        "iou_match_threshold": 0.5,
        "evidence_samples": manifest,
        "blind_samples": blind_manifest,
    }
    (assets / "sample_manifest.json").write_text(json.dumps(sample_manifest, indent=2) + "\n", encoding="utf-8")
    readiness = {
        "status": "provisional",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection": report["selection"],
        "screening": report["screening"],
        "ap": report["ap"],
        "comparisons": report["comparisons"],
        "runtime_size_comparisons": report["runtime_size_comparisons"],
        "target_conditioned_recall_at": report["target_conditioned_recall_at"],
        "runtime_verification": runtime,
        "evaluator": report["evaluator"],
        "report_generator": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
        "sample_manifest": "sample_manifest.json",
        "human_review": {"status": "pending", "artifact": None, "instructions": "Complete the browser review and export JSON."},
        "known_limitations": report["known_limitations"],
    }
    (assets / "readiness_metrics.json").write_text(json.dumps(readiness, indent=2) + "\n", encoding="utf-8")
    html_path = args.output_dir / "stage2-nearfield-detector-readiness.html"
    html_path.write_text(_html(report, runtime, manifest, blind_manifest, rules, "assets/stage2-nearfield-detector-readiness"), encoding="utf-8")
    print(json.dumps({"html": str(html_path), "evidence_samples": len(manifest), "blind_samples": len(blind_manifest), "rules": rules}, indent=2))


if __name__ == "__main__":
    main()
