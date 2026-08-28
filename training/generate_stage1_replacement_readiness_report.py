"""Generate the standalone Stage 1 corrective replacement readiness report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


CAMERA_RE = re.compile(r"(rig_\d+_[a-z])$")
EXPECTED_CAMERAS = {"rig_1_a", "rig_2_b", "rig_2_c", "rig_3_b", "rig_3_c", "rig_7_a"}
COLORS = {"match": "#27c47d", "false": "#ff5964", "truth": "#ffbd59", "blind": "#48cae4"}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _metric(value) -> str:
    return "—" if value is None else f"{100 * float(value):.2f}%"


def _table(headers: list[str], rows: list[list[object]], highlight: str | None = None) -> str:
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = []
    for row in rows:
        css = " class='highlight'" if highlight and str(row[0]) == highlight else ""
        body.append(f"<tr{css}>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _dataset_coverage(data_yaml: Path) -> dict:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = data_yaml.parent
    names = [str(data["names"][key]) for key in range(len(data["names"]))]
    umbrella_id = names.index("umbrella")
    output = {
        "ordered_classes": names, "ignored_classes": data["ignored_classes"],
        "source_revision": data["source_revision"],
        "source_data_yaml_sha256": data["source_data_yaml_sha256"],
        "camera_metadata_source": "organizer filename convention only; no independent camera or scene manifest available",
        "splits": {}, "scene_overlap": [],
    }
    scenes = {}
    for split in ("train", "val"):
        counts, cameras, sizes, scenes[split] = 0, Counter(), Counter(), set()
        for image_path in sorted((root / data[split]).iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            match = CAMERA_RE.search(image_path.stem)
            camera = match.group(1) if match else "metadata-unavailable"
            scenes[split].add(image_path.stem.split("_rig_", 1)[0])
            with Image.open(image_path) as image:
                width, height = image.size
            label_path = root / "labels" / split / f"{image_path.stem}.txt"
            for line in label_path.read_text(encoding="utf-8").splitlines():
                class_id, _cx, _cy, box_width, box_height = map(float, line.split())
                if int(class_id) != umbrella_id:
                    continue
                short = min(box_width * width, box_height * height)
                bucket = "lt8" if short < 8 else "8_to_lt16" if short < 16 else "16_to_lt32" if short < 32 else "ge32"
                counts += 1
                cameras[camera] += 1
                sizes[bucket] += 1
        output["splits"][split] = {
            "umbrella_boxes": counts, "umbrella_by_camera": dict(sorted(cameras.items())),
            "umbrella_by_native_short_side": {key: sizes[key] for key in ("lt8", "8_to_lt16", "16_to_lt32", "ge32")},
            "tiny_lt16": sizes["lt8"] + sizes["8_to_lt16"],
        }
    output["scene_overlap"] = sorted(scenes["train"] & scenes["val"])
    return output


def _training_curves(results_csv: Path, selected: str, destination: Path) -> None:
    import matplotlib.pyplot as plt

    with results_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(float(row["epoch"])) for row in rows]
    series = [
        ("train/box_loss", "Train box loss"), ("val/box_loss", "Val box loss"),
        ("metrics/precision(B)", "Generic precision"), ("metrics/recall(B)", "Generic recall"),
        ("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50–95"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for axis, (key, label) in zip(axes.flat, series):
        axis.plot(epochs, [float(row[key]) for row in rows], color="#48cae4", linewidth=1.8)
        axis.set(title=label, xlabel="epoch")
        axis.grid(alpha=0.25)
    figure.suptitle(f"Selected run training curves · {selected}")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def _draw(image: Image.Image, boxes: list[dict], *, truth: bool = False, blind: bool = False) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    for item in boxes:
        color = COLORS["truth"] if truth else COLORS["blind"] if blind else COLORS["match"] if item.get("matched") else COLORS["false"]
        xyxy = item["xyxy"]
        draw.rectangle(xyxy, outline=color, width=3)
        label = item.get("class_name", "")
        if not truth:
            label += f" {item.get('conf', item.get('confidence', 0)):.2f}"
        draw.text((xyxy[0] + 3, max(1, xyxy[1] - 12)), label, fill=color, font=font)
    return output


def _panel(frame: dict, old: dict, destination: Path) -> None:
    raw = Image.open(frame["source_image"]).convert("RGB")
    panels = [raw, _draw(raw, old["candidates"]), _draw(raw, frame["candidates"]), _draw(raw, frame["ground_truth"], truth=True)]
    titles = ["raw", "old published Stage 1", "new replacement", "processed ground truth"]
    width = 430
    resized = []
    for title, panel in zip(titles, panels):
        height = round(panel.height * width / panel.width)
        panel = panel.resize((width, height))
        tile = Image.new("RGB", (width, height + 28), "#09111f")
        ImageDraw.Draw(tile).text((8, 8), title, fill="white", font=ImageFont.load_default())
        tile.paste(panel, (0, 28))
        resized.append(tile)
    canvas = Image.new("RGB", (sum(item.width for item in resized), max(item.height for item in resized)), "#09111f")
    offset = 0
    for item in resized:
        canvas.paste(item, (offset, 0))
        offset += item.width
    canvas.save(destination, quality=91)


def _pick_evidence(report: dict) -> tuple[list[dict], dict]:
    frames = report["operational"]["frames"]
    floor = {item["image"]: item for item in report["confidence_floor_operational"]["frames"]}
    old = {item["image"]: item for item in report["old_published_comparison"]["old"]["frames"]}
    selected, rules, used = [], {}, set()

    def add(rule: str, candidates: list[dict], *, allow_reuse: bool = False) -> None:
        item = next((value for value in candidates if allow_reuse or value["image"] not in used), None)
        if item:
            selected.append({"reason": rule, "new": item, "old": old[item["image"]]})
            used.add(item["image"])
            rules[rule] = item["image"]
        else:
            rules[rule] = None

    umbrella_success = [item for item in frames if any(gt["class_name"] == "umbrella" and gt["matched"] for gt in item["ground_truth"])]
    cameras = []
    for item in umbrella_success:
        if item["camera"] not in cameras:
            cameras.append(item["camera"])
            add(f"umbrella success · {item['camera']}", [item])
        if len(cameras) == 2:
            break
    add("smallest useful umbrella success", sorted(umbrella_success, key=lambda item: min(gt["short_side"] for gt in item["ground_truth"] if gt["class_name"] == "umbrella" and gt["matched"])))
    umbrella_miss = [item for item in frames if any(gt["class_name"] == "umbrella" and not gt["matched"] for gt in item["ground_truth"])]
    recovered, persistent = [], []
    for item in umbrella_miss:
        floor_truth = floor[item["image"]]["ground_truth"]
        missed = [index for index, gt in enumerate(item["ground_truth"]) if gt["class_name"] == "umbrella" and not gt["matched"]]
        if any(floor_truth[index]["matched"] for index in missed):
            recovered.append(item)
        if any(not floor_truth[index]["matched"] for index in missed):
            persistent.append(item)
    add("persistent umbrella miss", persistent, allow_reuse=True)
    add("umbrella threshold recovery", recovered, allow_reuse=True)
    add("strongest umbrella false candidate", sorted(
        [item for item in frames if any(candidate["class_name"] == "umbrella" and not candidate["matched"] for candidate in item["candidates"])],
        key=lambda item: max(candidate["conf"] for candidate in item["candidates"] if candidate["class_name"] == "umbrella" and not candidate["matched"]),
        reverse=True,
    ))
    add("strongest false candidate", sorted(
        [item for item in frames if any(not candidate["matched"] for candidate in item["candidates"])],
        key=lambda item: max(candidate["conf"] for candidate in item["candidates"] if not candidate["matched"]),
        reverse=True,
    ))
    add("tiny success", [item for item in frames if any(gt["short_side"] < 16 and gt["matched"] for gt in item["ground_truth"])])
    add("tiny miss", [item for item in frames if any(gt["short_side"] < 16 and not gt["matched"] for gt in item["ground_truth"])])
    deltas = report["old_published_comparison"]["class_deltas"]
    improved = max((name for name in deltas if (deltas[name]["f1"] or 0) > 0), key=lambda name: deltas[name]["f1"], default=None)
    regressed = min((name for name in deltas if (deltas[name]["f1"] or 0) < 0), key=lambda name: deltas[name]["f1"], default=None)
    add(f"largest shared-class improvement · {improved}" if improved else "largest shared-class improvement", [] if not improved else [item for item in frames if any(gt["class_name"] == improved and gt["matched"] for gt in item["ground_truth"]) and any(gt["class_name"] == improved and not gt["matched"] for gt in old[item["image"]]["ground_truth"])])
    add(f"largest shared-class regression · {regressed}" if regressed else "largest shared-class regression", [] if not regressed else [item for item in frames if any(gt["class_name"] == regressed and not gt["matched"] for gt in item["ground_truth"]) and any(gt["class_name"] == regressed and gt["matched"] for gt in old[item["image"]]["ground_truth"])])
    worst_camera = min(report["operational"]["cameras"], key=lambda name: report["operational"]["cameras"][name]["recall"])
    add(f"worst camera · {worst_camera}", sorted([item for item in frames if item["camera"] == worst_camera], key=lambda item: sum(not gt["matched"] for gt in item["ground_truth"]), reverse=True))
    add("highest false-candidate load", sorted(frames, key=lambda item: sum(not candidate["matched"] for candidate in item["candidates"]), reverse=True))
    add("crowded scene", sorted(frames, key=lambda item: len(item["ground_truth"]), reverse=True))
    add("clean control", [item for item in frames if item["ground_truth"] and all(gt["matched"] for gt in item["ground_truth"]) and all(candidate["matched"] for candidate in item["candidates"])])
    return selected, rules


def _blind(frames: list[dict], excluded: set[str]) -> list[dict]:
    by_camera = defaultdict(list)
    for item in frames:
        if item["image"] not in excluded:
            complexity = len(item["ground_truth"]) + 2 * sum(gt["short_side"] < 16 for gt in item["ground_truth"])
            by_camera[item["camera"]].append((complexity, item))
    chosen = []
    for camera in sorted(by_camera):
        options = sorted(by_camera[camera], key=lambda pair: pair[0], reverse=True)
        if options:
            chosen.append({"mix": "risk-heavy (dense/tiny scene)", **options[0][1]})
        lower = next(((complexity, item) for complexity, item in reversed(options) if item["image"] != options[0][1]["image"]), None)
        if lower:
            chosen.append({"mix": "clean/control (lower scene complexity)", **lower[1]})
    chosen = chosen[:12]
    if not 10 <= len(chosen) <= 12 or {item["camera"] for item in chosen} != EXPECTED_CAMERAS:
        raise ValueError("blind review must contain 10-12 samples covering all six cameras")
    if {item["image"] for item in chosen} & excluded:
        raise ValueError("blind review overlaps visible evidence")
    return chosen


def _copy_plots(report: dict, run_dir: Path, assets: Path) -> None:
    selected = Path(report["selection"]["checkpoint"]).name
    _training_curves(run_dir / "results.csv", selected, assets / "training-curves.png")
    plot_dir = Path(report["generic_validation"]["save_dir"])
    for source, destination in (
        ("BoxPR_curve.png", "BoxPR_curve.png"), ("BoxF1_curve.png", "BoxF1_curve.png"),
        ("confusion_matrix.png", "confusion_matrix.png"),
        ("confusion_matrix_normalized.png", "confusion_matrix_normalized.png"),
    ):
        shutil.copy2(plot_dir / source, assets / destination)


def _html(report: dict, runtime: dict, coverage: dict, evidence: list[dict], blind: list[dict], assets_rel: str, manifest_sha: str, approved: bool) -> str:
    settings, operational = report["selection"]["runtime"], report["operational"]
    completion = report["training_completion"]
    umbrella = operational["classes"]["umbrella"]
    ap_rows = {row["Class"]: row for row in report["generic_validation"]["classes"]}
    worst_class = min(operational["classes"], key=lambda name: operational["classes"][name]["recall"])
    worst_camera = min(operational["cameras"], key=lambda name: operational["cameras"][name]["recall"])
    shared_delta = report["old_published_comparison"]["new"]["macro_f1"] - report["old_published_comparison"]["old"]["macro_f1"]
    umbrella_misses = sum(
        gt["class_name"] == "umbrella" and not gt["matched"]
        for frame in operational["frames"] for gt in frame["ground_truth"]
    )
    umbrella_false = [
        (frame["image"], candidate) for frame in operational["frames"] for candidate in frame["candidates"]
        if candidate["class_name"] == "umbrella" and not candidate["matched"]
    ]
    umbrella_confusions = Counter()
    umbrella_duplicates = 0
    for frame in operational["frames"]:
        for candidate in frame["candidates"]:
            confused = candidate.get("confused_with")
            if candidate["class_name"] == "umbrella" and not candidate["matched"] and confused == "umbrella":
                umbrella_duplicates += 1
            elif candidate["class_name"] == "umbrella" and confused:
                umbrella_confusions[f"{confused}→umbrella"] += 1
            elif candidate["class_name"] != "umbrella" and confused == "umbrella":
                umbrella_confusions[f"umbrella→{candidate['class_name']}"] += 1
    class_rows = [[name, value["gt"], _metric(value["precision"]), _metric(value["recall"]), _metric(value["f1"]), _metric(ap_rows[name]["mAP50"]), _metric(ap_rows[name]["mAP50-95"]), _metric(value["tiny_recall"])] for name, value in operational["classes"].items()]
    checkpoint_rows = [[Path(item["checkpoint"]).name, _metric(item["macro_f1"]), _metric(min(v["recall"] for v in item["classes"].values())), _metric(item["classes"]["umbrella"]["recall"]), _metric(item["sizes"]["tiny"]["recall"]), _metric(min(v["recall"] for v in item["cameras"].values())), _metric(item["recall"]), "selected" if item["checkpoint"] == report["selection"]["checkpoint"] else "eligible" if item["eligible"] else "rejected: " + ", ".join(item["rejection"])] for item in report["screening"]]
    head_rows = []
    for end2end in (True, False):
        choices = [item for item in report["head_and_runtime_comparisons"] if item["end2end"] == end2end and item["confidence"] == 0.001 and item["candidate_limit"] == 20]
        item = max(choices, key=lambda value: (
            value["macro_recall"], min(v["recall"] for v in value["classes"].values()),
            value["classes"]["umbrella"]["recall"], value["sizes"]["tiny"]["recall"],
            min(v["recall"] for v in value["cameras"].values()), value["recall"],
        ))
        head_rows.append([
            "one-to-one" if end2end else "one-to-many/NMS", item["nms_iou"],
            _metric(item["macro_recall"]), _metric(min(v["recall"] for v in item["classes"].values())),
            _metric(item["classes"]["umbrella"]["recall"]), _metric(item["sizes"]["tiny"]["recall"]),
            _metric(min(v["recall"] for v in item["cameras"].values())), _metric(item["recall"]),
            item["candidate_load"],
        ])
    setting_rows = [[item["confidence"], item["nms_iou"], item["candidate_limit"], _metric(item["macro_f1"]), _metric(min(v["recall"] for v in item["classes"].values())), _metric(item["classes"]["umbrella"]["recall"]), item["candidate_load"]] for item in report["head_and_runtime_comparisons"] if item["end2end"] == settings["end2end"]]
    camera_rows = [[name, value["gt"], _metric(value["precision"]), _metric(value["recall"]), _metric(value["f1"])] for name, value in operational["cameras"].items()]
    size_rows = [[name, value["gt"], value["matched"], _metric(value["recall"]), operational["size_bucket_definitions"][name]] for name, value in operational["sizes"].items()]
    delta_rows = [[name, _metric(values["old"]["precision"]), _metric(values["new"]["precision"]), _metric(values["precision"]), _metric(values["old"]["recall"]), _metric(values["new"]["recall"]), _metric(values["recall"]), _metric(values["old"]["f1"]), _metric(values["new"]["f1"]), _metric(values["f1"])] for name, values in sorted(report["old_published_comparison"]["class_deltas"].items(), key=lambda item: item[1]["f1"] if item[1]["f1"] is not None else float("inf"))]
    camera_delta_rows = [[name, _metric(value["recall"]), _metric(value["f1"])] for name, value in report["old_published_comparison"]["camera_deltas"].items()]
    size_delta_rows = [[name, _metric(value)] for name, value in report["old_published_comparison"]["size_recall_deltas"].items()]
    coverage_rows = [[split, values["umbrella_boxes"], values["tiny_lt16"], json.dumps(values["umbrella_by_camera"], sort_keys=True), json.dumps(values["umbrella_by_native_short_side"], sort_keys=True)] for split, values in coverage["splits"].items()]
    evidence_html = "".join(f"<article class='sample'><h3>{html.escape(item['reason'])}</h3><p>{html.escape(item['image'])} · {html.escape(item['camera'])}</p><img src='{assets_rel}/{item['asset']}' alt='raw, old, new, and processed truth panels'></article>" for item in evidence)
    blind_html = "".join(f"<article class='blind-card' data-card='{item['id']}'><h3>{item['id']} · {html.escape(item['camera'])} · {html.escape(item['mix'])}</h3><img src='{assets_rel}/{item['model_asset']}' alt='new model full candidate overlay'><div class='review-grid'><label>Candidate coverage<select name='coverage'><option value=''>—</option><option>acceptable</option><option>concerning</option><option>ambiguous</option></select></label><label>False candidates<select name='false_candidates'><option value=''>—</option><option>acceptable</option><option>concerning</option><option>ambiguous</option></select></label><label>Overall<select name='overall'><option value=''>—</option><option>acceptable</option><option>concerning</option><option>ambiguous</option></select></label></div><label>Notes<textarea name='notes'></textarea></label><details><summary>Reveal processed ground truth</summary><img src='{assets_rel}/{item['truth_asset']}' alt='processed ground truth'></details></article>" for item in blind)
    generic = report["generic_validation"]["generic"]
    known_limitations = [
        "Blind review was explicitly waived by human approval on 2026-08-29."
        if approved and value.startswith("Human blind full-frame review is pending") else value
        for value in report["known_limitations"]
    ]
    limitations = "".join(f"<li>{html.escape(value)}</li>" for value in known_limitations)
    banner = "HUMAN APPROVED · BLIND REVIEW WAIVED" if approved else "HUMAN REVIEW PENDING"
    gate = (
        "Human approval on 2026-08-29 explicitly waived the blind review."
        if approved else "Final replacement readiness is not claimed until the blind review is completed and exported."
    )
    approval_section = (
        "<h2>Human approval</h2><p><b>Approved on 2026-08-29.</b> The human approver waived completion of the blind-review form. The full interactive blind-review section remains intact below.</p>"
        if approved else ""
    )
    blind_section = f"<h2>Blind full-frame review</h2><p class='pending' id='progress'>Human review pending: 0 / {len(blind)}</p><p>This {len(blind)}-frame set is entirely frame-disjoint from visible evidence and covers all six filename-derived camera groups. The risk-heavy/clean mix is based only on scene density and tiny-object presence, never evaluator outcomes. The new model’s candidates use one neutral color so match status remains hidden until processed ground truth is revealed.</p>{blind_html}<p><button id='export'>Export review JSON</button><button id='clear'>Clear local review</button></p>"
    verdict = (
        "<b>Corrective replacement approved for promotion.</b> Human approval explicitly waived blind review."
        if approved else
        "<b>Offline corrective candidate prepared; human approval pending.</b> <span class='pending'>Do not replace the published bundle or claim final readiness until the blind review is complete.</span>"
    )
    review_script = f"""<script>
const key={json.dumps('marc-stage1-replacement-review-v1:' + runtime['candidate_weights_sha256'] + ':' + manifest_sha)},cards=[...document.querySelectorAll('.blind-card')];
function data(){{const out={{}};for(const card of cards){{out[card.dataset.card]={{}};for(const field of card.querySelectorAll('[name]'))out[card.dataset.card][field.name]=field.value;}}return out}}
function complete(card){{return ['coverage','false_candidates','overall'].every(name=>card.querySelector(`[name=${{name}}]`).value)}}
let memory={{}};function load(){{try{{return JSON.parse(localStorage.getItem(key)||'{{}}')}}catch(_error){{return memory}}}}function store(value){{memory=value;try{{localStorage.setItem(key,JSON.stringify(value))}}catch(_error){{}}}}
function save(){{store(data());const done=cards.filter(complete).length;document.querySelector('#progress').textContent=`Human review ${{done===cards.length?'complete':'pending'}}: ${{done}} / ${{cards.length}}`;}}
const stored=load();for(const card of cards)for(const field of card.querySelectorAll('[name]'))field.value=stored[card.dataset.card]?.[field.name]||'';document.addEventListener('input',save);document.addEventListener('change',save);save();
document.querySelector('#export').onclick=()=>{{const reviews=data(),done=cards.filter(complete).length,payload={{schema:'marc-stage1-replacement-human-review-v1',status:done===cards.length?'complete':'in_progress',exported_at:new Date().toISOString(),selected_checkpoint:{json.dumps(Path(report['selection']['checkpoint']).name)},selected_checkpoint_sha256:{json.dumps(runtime['selected_training_checkpoint_sha256'])},deployment_sha256:{json.dumps(runtime['candidate_weights_sha256'])},fixed_settings:{json.dumps(settings)},sample_manifest_sha256:{json.dumps(manifest_sha)},blind_samples:{json.dumps([{'id': item['id'], 'image': item['image'], 'camera': item['camera']} for item in blind])},reviews}};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}}));a.download='human-review-stage1-replacement.json';a.click();URL.revokeObjectURL(a.href)}};
document.querySelector('#clear').onclick=()=>{{if(confirm('Clear browser-local review progress?')){{memory={{}};try{{localStorage.removeItem(key)}}catch(_error){{}}for(const field of document.querySelectorAll('.blind-card [name]'))field.value='';save()}}}};
</script>"""
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Stage 1 corrective replacement candidate</title><style>
:root{{--bg:#09111f;--panel:#111d30;--ink:#e8f0fa;--muted:#9fb1c8;--cyan:#48cae4;--amber:#ffbd59;--green:#27c47d;--red:#ff5964}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1240px;margin:auto;padding:28px}}h1,h2,h3{{line-height:1.15}}h2{{margin-top:46px;border-bottom:1px solid #29405e;padding-bottom:8px}}.banner{{border:2px solid var(--amber);background:#2a210e;padding:14px;border-radius:10px;font-weight:800}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}.metric,.sample,.blind-card,.note{{background:var(--panel);border:1px solid #263a55;border-radius:12px;padding:16px}}.metric b{{display:block;font-size:23px;color:var(--cyan)}}img{{display:block;width:100%;height:auto;border-radius:8px}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;background:var(--panel)}}th,td{{padding:8px 10px;border-bottom:1px solid #263a55;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}tr.highlight{{background:#493b12}}.sample,.blind-card{{margin:14px 0}}.review-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:12px 0}}label{{display:grid;gap:4px;color:var(--muted)}}select,textarea,button{{background:#0a1424;color:var(--ink);border:1px solid #3a5578;border-radius:6px;padding:8px}}textarea{{min-height:70px}}button{{cursor:pointer;margin-right:8px}}.pending,.warning{{color:var(--red);font-weight:800}}code{{color:var(--cyan)}}details{{margin-top:10px}}@media(max-width:700px){{main{{padding:14px}}}}
</style></head><body><main>
<div class='banner'>STAGE 1 CORRECTIVE REPLACEMENT · 34 CLASSES · VALIDATION-ONLY · {banner}</div>
<h1>Stage 1 CCTV detector replacement readiness</h1><p>Selected <code>{html.escape(Path(report['selection']['checkpoint']).name)}</code>, {'one-to-one' if settings['end2end'] else 'one-to-many/NMS'}, confidence {settings['confidence']}, NMS {settings['nms_iou']}, global top-{settings['candidate_limit']}, 1920 px, {str(settings['quantize']).upper()}.</p>
<div class='note warning'>No independent test split exists. The same validation split was used for checkpoint/settings selection and every reported model metric. {gate}</div>
<div class='cards'><div class='metric'>Macro F1<b>{_metric(operational['macro_f1'])}</b></div><div class='metric'>Umbrella recall<b>{_metric(umbrella['recall'])}</b></div><div class='metric'>Shared-class Δ F1<b>{shared_delta:+.2%}</b></div><div class='metric'>Worst class · {worst_class}<b>{_metric(operational['classes'][worst_class]['recall'])}</b></div><div class='metric'>Tiny &lt;16 recall<b>{_metric(operational['sizes']['tiny']['recall'])}</b></div><div class='metric'>Worst camera · {worst_camera}<b>{_metric(operational['cameras'][worst_camera]['recall'])}</b></div><div class='metric'>Warm p95 latency<b>{runtime['benchmark']['p95_ms']:.2f} ms</b></div><div class='metric'>Runtime peak CUDA allocated<b>{runtime['benchmark']['peak_allocated_gib']:.3f} GiB</b></div></div>
<h2>Class-map correction and artifact lineage</h2><p>The old 33-class head is <b>unsupported for ordinary umbrella</b>; it cannot emit that class. The candidate contains ordinary <code>umbrella</code> and excludes exactly <code>long_umbrella</code>, <code>master_chef_can</code>, and <code>wood_block</code>. Selected training SHA <code>{runtime['selected_training_checkpoint_sha256']}</code>; stripped deployment SHA <code>{runtime['candidate_weights_sha256']}</code>; superseded model SHA <code>{runtime['published_weights_sha256']}</code>.</p>
<h2>Dataset and filename-derived coverage evidence</h2>{_table(['split','umbrella GT','tiny <16','by camera','by native short-side bin'],coverage_rows)}<p>Filename-derived camera groups include all six expected suffixes but are not independently verified. Filename-derived scene-token overlap: {len(coverage['scene_overlap'])}. {html.escape(coverage['camera_metadata_source'])}. Source revision <code>{coverage['source_revision']}</code>.</p>
<h2>Training completion, curves, and saved-checkpoint ranking</h2><p>Normal completion proven: {completion['completed_epoch']}/100 finite contiguous CSV rows; normal end callback and final metrics present; <code>best.pt</code>, <code>last.pt</code>, and expected periodic checkpoints <code>epoch0.pt</code> through <code>epoch90.pt</code> present; no live trainer. Peak training memory: {completion['peak_training_memory_gib']:.3f} GiB.</p><img src='{assets_rel}/training-curves.png' alt='training curves generated from the selected run results.csv'>{_table(['checkpoint','macro F1','worst recall','umbrella recall','tiny recall','worst camera','overall recall','decision'],checkpoint_rows)}
<h2>Equal-budget head and bounded runtime selection</h2><h3>Confidence floor / global top-20</h3>{_table(['head','NMS','macro recall','worst recall','umbrella recall','tiny recall','worst camera','overall recall','candidate load'],head_rows)}<h3>Validation confidence / NMS / top-20 or top-50 grid</h3>{_table(['confidence','NMS','global cap','macro F1','worst recall','umbrella recall','candidate load'],setting_rows)}
<h2>Complete 34-class result</h2><p>Operational bounded-global result: P {_metric(operational['precision'])}, R {_metric(operational['recall'])}, F1 {_metric(2 * operational['precision'] * operational['recall'] / (operational['precision'] + operational['recall']) if operational['precision'] + operational['recall'] else 0)}, macro F1 {_metric(operational['macro_f1'])}. Generic validation is separate: P {_metric(generic['metrics/precision(B)'])}, R {_metric(generic['metrics/recall(B)'])}, F1 {_metric(2 * generic['metrics/precision(B)'] * generic['metrics/recall(B)'] / (generic['metrics/precision(B)'] + generic['metrics/recall(B)']) if generic['metrics/precision(B)'] + generic['metrics/recall(B)'] else 0)}, mAP50 {_metric(generic['metrics/mAP50(B)'])}, mAP50–95 {_metric(generic['metrics/mAP50-95(B)'])}.</p>{_table(['class','GT','precision','recall','F1','AP50','AP50–95','tiny recall'],class_rows,'umbrella')}
<h2>Camera, native-pixel size, and candidate load</h2>{_table(['camera','GT','precision','recall','F1'],camera_rows)}{_table(['bucket','GT','matched','recall','definition'],size_rows)}<p>Global false candidates: {operational['false_candidates']}. Candidates/frame: mean {operational['candidates_per_frame']['mean']:.2f}, median {operational['candidates_per_frame']['median']:.1f}, p95 {operational['candidates_per_frame']['p95']:.1f}, max {operational['candidates_per_frame']['max']:.0f}; {operational['candidates_per_frame']['frames_hitting_cap']} frames hit the cap.</p>
<h2>Ordinary umbrella deep dive</h2><p>P {_metric(umbrella['precision'])}, R {_metric(umbrella['recall'])}, F1 {_metric(umbrella['f1'])}, AP50 {_metric(ap_rows['umbrella']['mAP50'])}, AP50–95 {_metric(ap_rows['umbrella']['mAP50-95'])}. Misses: {umbrella_misses}; false umbrella candidates: {len(umbrella_false)}; duplicate umbrella candidates: {umbrella_duplicates}; IoU≥0.5 class confusions: <code>{html.escape(json.dumps(dict(umbrella_confusions), sort_keys=True))}</code>. Matched-confidence distribution: <code>{html.escape(json.dumps(umbrella['matched_confidence'], sort_keys=True))}</code>.</p>{_table(['camera','GT','precision','recall','F1'],[[name,value['gt'],_metric(value['precision']),_metric(value['recall']),_metric(value['f1'])] for name,value in umbrella['by_camera'].items()])}{_table(['size','GT','matched','recall'],[[name,value['gt'],value['matched'],_metric(value['recall'])] for name,value in umbrella['by_size'].items()])}<p>Representative cross-camera successes, smallest success, persistent miss, threshold recovery, strongest umbrella false candidate, and relevant overlap confusions are rule-selected below. Unavailable categories remain explicitly null in the manifest.</p>
<h2>Old published versus corrective replacement</h2><p>Same revised frames, head, confidence, NMS, 1920 resolution, and global cap. Shared-class macro F1: old {_metric(report['old_published_comparison']['old']['macro_f1'])}, new {_metric(report['old_published_comparison']['new']['macro_f1'])}; recall: old {_metric(report['old_published_comparison']['old']['macro_recall'])}, new {_metric(report['old_published_comparison']['new']['macro_recall'])}. Mean global candidates/frame: old {report['old_published_comparison']['old']['candidates_per_frame']['mean']:.2f}, new {report['old_published_comparison']['new']['candidates_per_frame']['mean']:.2f}. Umbrella is unsupported by the old class head.</p>{_table(['shared class','old P','new P','Δ P','old R','new R','Δ R','old F1','new F1','Δ F1'],delta_rows)}<h3>Per-camera deltas</h3>{_table(['camera','Δ recall','Δ F1'],camera_delta_rows)}<h3>Native-size recall deltas</h3>{_table(['bucket','Δ recall'],size_delta_rows)}<p>Runtime deltas (new − old): mean {runtime['benchmark_delta']['mean_ms']:+.2f} ms, p95 {runtime['benchmark_delta']['p95_ms']:+.2f} ms, allocated {_metric(runtime['benchmark_delta']['peak_allocated_gib']) if runtime['benchmark_delta']['peak_allocated_gib'] is None else f"{runtime['benchmark_delta']['peak_allocated_gib']:+.3f} GiB"}.</p>
<h2>Selected-head PR/F1 curves and confusion matrices</h2><div class='cards'><img src='{assets_rel}/BoxPR_curve.png' alt='selected-head PR curve'><img src='{assets_rel}/BoxF1_curve.png' alt='selected-head F1 curve'><img src='{assets_rel}/confusion_matrix.png' alt='raw confusion matrix'><img src='{assets_rel}/confusion_matrix_normalized.png' alt='normalized confusion matrix'></div><p>These plots were generated by the selected checkpoint and selected head validation call, not reused from trainer-final validation.</p>
<h2>Runtime and offline verification</h2><p>Fresh process, explicit local path, socket connect/bind denied, batch 1 after {runtime['benchmark']['warmup_iterations']} warmups. Mean {runtime['benchmark']['mean_ms']:.2f} ms, median {runtime['benchmark']['median_ms']:.2f} ms, p95 {runtime['benchmark']['p95_ms']:.2f} ms; CUDA allocated/reserved {runtime['benchmark']['peak_allocated_gib']:.3f}/{runtime['benchmark']['peak_reserved_gib']:.3f} GiB. Exact 34-class order, three exclusions, global descending cap, original-image coordinates, and consumer conversion passed.</p>
<h2>Exact before / after evidence</h2><p>Every panel uses the identical frame, 1920 profile, selected head/confidence/NMS/global cap, and scaling: raw | old published Stage 1 | new replacement | processed ground truth.</p>{evidence_html}
{approval_section}{blind_section}
<h2>Traceability, commands, and limitations</h2><p>Evaluator <code>{report['evaluator']['sha256']}</code>; runtime verifier <code>{runtime['verifier']['sha256']}</code>; report generator <code>{_sha256(Path(__file__))}</code>. Evaluator command <code>{html.escape(json.dumps(report['evaluator']['argv']))}</code>; verifier command <code>{html.escape(json.dumps(runtime['verifier']['argv']))}</code>; report command <code>{html.escape(json.dumps(sys.argv))}</code>. Exact numeric inputs and manifests are relative assets.</p><ul>{limitations}</ul>
<h2>Replacement verdict</h2><p>{verdict} Numeric selection, shared-class comparison, bundle staging, and fresh-process interface verification completed.</p>
{review_script}</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--runtime-verification", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--human-approved-without-blind-review", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.evaluation.read_text(encoding="utf-8"))
    runtime = json.loads(args.runtime_verification.read_text(encoding="utf-8"))
    bundle = Path(runtime["candidate_bundle"]).resolve()
    bundle_contract = json.loads((bundle / "runtime_contract.json").read_text(encoding="utf-8"))
    required_bundle_files = {
        "model.pt", "runtime_contract.json", "SOURCE.md", "LICENSE",
        "candidate_output_example.json", "runtime_verification.json",
    }
    sums_path = bundle / "SHA256SUMS"
    recorded_sums = {
        name: digest for digest, name in (
            line.split("  ", 1) for line in sums_path.read_text(encoding="utf-8").splitlines()
        )
    } if sums_path.is_file() else {}
    bundle_complete = (
        bundle_contract.get("local_checkpoint_load_verified") is True
        and bundle_contract.get("offline_bundle_load_verified") is True
        and set(recorded_sums) == required_bundle_files
        and all((bundle / name).is_file() and _sha256(bundle / name) == digest for name, digest in recorded_sums.items())
    )
    expected_training_sha = report["runtime_contract"]["selected_training_checkpoint"]["sha256"]
    if (
        Path(report["dataset"]).resolve() != args.data.resolve()
        or Path(runtime["dataset"]).resolve() != args.data.resolve()
        or runtime["dataset_sha256"] != _sha256(args.data.resolve())
        or runtime["dataset_identity"] != bundle_contract["dataset"]
        or bundle_contract["dataset"] != report["runtime_contract"]["dataset"]
        or Path(report["training_completion"]["results_csv"]).resolve() != (args.run_dir / "results.csv").resolve()
        or not runtime.get("verified")
        or not bundle_complete
        or Path(args.runtime_verification).resolve() != bundle / "runtime_verification.json"
        or runtime["candidate_weights_sha256"] != _sha256(bundle / "model.pt")
        or runtime["candidate_weights_sha256"] != bundle_contract["checkpoint_sha256"]
        or runtime["selected_training_checkpoint_sha256"] != expected_training_sha
        or bundle_contract["selected_training_checkpoint"]["sha256"] != expected_training_sha
        or runtime["settings"] != bundle_contract["inference"]
        or bundle_contract["inference"] != report["runtime_contract"]["inference"]
        or runtime["ordered_classes"] != report["runtime_contract"]["ordered_classes"]
        or bundle_contract["ordered_classes"] != runtime["ordered_classes"]
        or runtime["ignored_classes"] != report["runtime_contract"]["ignored_classes"]
        or bundle_contract["ignored_classes"] != runtime["ignored_classes"]
    ):
        raise ValueError("evaluation, run, dataset, runtime verification, and staged bundle lineage differ")
    coverage = _dataset_coverage(args.data)
    output, assets = args.output_dir, args.output_dir / "assets" / "stage1-cctv-detector-replacement-readiness"
    assets.mkdir(parents=True, exist_ok=True)
    _copy_plots(report, args.run_dir, assets)
    picked, rules = _pick_evidence(report)
    evidence_manifest = []
    for index, item in enumerate(picked, 1):
        asset = f"evidence-{index:02d}.jpg"
        _panel(item["new"], item["old"], assets / asset)
        evidence_manifest.append({
            "id": f"evidence-{index:02d}", "reason": item["reason"],
            "image": item["new"]["image"], "source_image": item["new"]["source_image"],
            "camera": item["new"]["camera"], "asset": asset,
            "old_candidates": item["old"]["candidates"],
            "new_candidates": item["new"]["candidates"],
            "processed_ground_truth": item["new"]["ground_truth"],
        })
    blind_frames = _blind(report["operational"]["frames"], {item["image"] for item in evidence_manifest})
    blind_manifest = []
    for index, item in enumerate(blind_frames, 1):
        model_asset, truth_asset = f"blind-{index:02d}-model.jpg", f"blind-{index:02d}-truth.jpg"
        raw = Image.open(item["source_image"]).convert("RGB")
        _draw(raw, item["candidates"], blind=True).save(assets / model_asset, quality=91)
        _draw(raw, item["ground_truth"], truth=True).save(assets / truth_asset, quality=91)
        blind_manifest.append({
            "id": f"blind-{index:02d}", "image": item["image"], "camera": item["camera"],
            "source_image": item["source_image"], "mix": item["mix"],
            "model_asset": model_asset, "truth_asset": truth_asset,
            "new_candidates": item["candidates"], "processed_ground_truth": item["ground_truth"],
        })
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "settings": report["selection"]["runtime"], "evidence_selection_rules": rules,
        "evidence_samples": evidence_manifest, "blind_samples": blind_manifest,
        "blind_frame_disjoint": not ({item["image"] for item in evidence_manifest} & {item["image"] for item in blind_manifest}),
    }
    manifest_path = assets / "sample_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_sha = _sha256(manifest_path)
    compact_old = {key: value for key, value in report["old_published_comparison"].items()}
    compact_old["old"] = {key: value for key, value in compact_old["old"].items() if key != "frames"}
    readiness = {
        "status": "approved" if args.human_approved_without_blind_review else "human_review_pending",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection": report["selection"], "training_completion": report["training_completion"],
        "dataset_coverage": coverage, "screening": report["screening"],
        "head_and_runtime_comparisons": report["head_and_runtime_comparisons"],
        "generic_validation": report["generic_validation"],
        "operational": {key: value for key, value in report["operational"].items() if key != "frames"},
        "old_published_comparison": compact_old, "runtime_verification": runtime,
        "evaluator": report["evaluator"],
        "report_generator": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__)), "argv": sys.argv},
        "sample_manifest": "sample_manifest.json",
        "human_review": (
            {"status": "waived_by_human_approval", "artifact": None, "approved_at": "2026-08-29"}
            if args.human_approved_without_blind_review else
            {"status": "pending", "artifact": None, "instructions": "Complete the blind review in the HTML and export JSON."}
        ),
        "known_limitations": [
            "Blind review was explicitly waived by human approval on 2026-08-29."
            if args.human_approved_without_blind_review and value.startswith("Human blind full-frame review is pending") else value
            for value in report["known_limitations"]
        ],
        "input_artifacts": {
            "evaluation_sha256": _sha256(args.evaluation),
            "runtime_verification_sha256": _sha256(args.runtime_verification),
            "dataset_sha256": _sha256(args.data),
            "results_csv_sha256": _sha256(args.run_dir / "results.csv"),
            "sample_manifest_sha256": manifest_sha,
        },
    }
    (assets / "readiness_metrics.json").write_text(json.dumps(readiness, indent=2) + "\n", encoding="utf-8")
    html_path = output / "stage1-cctv-detector-replacement-readiness.html"
    html_path.write_text(_html(report, runtime, coverage, evidence_manifest, blind_manifest, "assets/stage1-cctv-detector-replacement-readiness", manifest_sha, args.human_approved_without_blind_review), encoding="utf-8")
    print(json.dumps({"html": str(html_path.resolve()), "evidence": len(evidence_manifest), "blind": len(blind_manifest)}, indent=2))


if __name__ == "__main__":
    main()
