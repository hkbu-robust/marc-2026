#!/usr/bin/env python3
"""Rebuild the paper tables and figures from archived MARC evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "figures" / "generated"
STAGE1_EVAL_DEFAULT = Path(
    "/home/marc2026/marc-2026-work/experiments/"
    "stage1-cctv-yolo26x-1920-umbrella-ceccc62/evaluations/evaluation_report.json"
)


def load(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))]


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def finish(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def result_runs() -> list[dict]:
    rows = []
    for path in RAW.glob("robust_*.json"):
        d = load(path)
        rows.append({"run": path.stem.removeprefix("robust_"), **d["scores"], "details": d["round_details"]})
    return sorted(rows, key=lambda x: x["run"])


def plot_score_progression(runs: list[dict]) -> None:
    labels = [r["run"][4:9] + "\n" + r["run"][10:] for r in runs]
    scores = [r["stage1_average"] for r in runs]
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    colors = ["#607d8b" if r["run"].startswith("20260830") else "#1565c0" for r in runs]
    ax.plot(range(len(runs)), scores, color="#263238", linewidth=1.3, zorder=1)
    ax.scatter(range(len(runs)), scores, c=colors, s=42, zorder=2)
    ax.axvline(4.5, color="#7b1fa2", linestyle="--", linewidth=1.1)
    ax.text(4.65, min(scores) + 2, "fine-tuned runtime era", color="#7b1fa2", rotation=90, va="bottom")
    ax.set_xticks(range(len(runs)), labels, rotation=35, ha="right")
    ax.set_ylabel("Stage 1 practice score / 100")
    ax.set_title("End-to-end practice score progression (observational, not a model ablation)")
    ax.set_ylim(max(0, min(scores) - 5), 100)
    best = max(range(len(scores)), key=scores.__getitem__)
    ax.annotate(f"best {scores[best]:.4f}", (best, scores[best]), xytext=(-45, -24),
                textcoords="offset points", arrowprops={"arrowstyle": "->", "lw": 0.8})
    finish(fig, "fig01_stage1_score_progression.png")


def plot_components(best: dict) -> dict:
    details = best["details"]
    keys = ["camera", "object_type", "landmark", "relation", "anchor", "target"]
    # Score only rounds where the organizer assigned a non-zero component weight.
    means = {k: statistics.fmean(float(r[k]) for r in details if r["weights"].get(k, 0) > 0) for k in keys}
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    bars = ax.bar(keys, [100 * means[k] for k in keys], color=["#00897b"] * 4 + ["#fb8c00", "#1565c0"])
    ax.bar_label(bars, labels=[f"{100 * means[k]:.1f}" for k in keys], padding=3)
    ax.set_ylim(0, 106)
    ax.set_ylabel("Mean score on applicable rounds (%)")
    ax.set_title(f"Best recorded Stage 1 run: {best['stage1_average']:.4f}/100 across {len(details)} rounds")
    finish(fig, "fig02_stage1_component_scores.png")
    return means


def plot_qwen() -> dict:
    state = load(RAW / "qwen_trainer_state.json")
    evals = [x for x in state["log_history"] if "eval_loss" in x]
    trains = [x for x in state["log_history"] if "loss" in x]
    first, best = evals[0], min(evals, key=lambda x: x["eval_loss"])
    reduction = 1 - best["eval_loss"] / first["eval_loss"]
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    ax.plot([x["step"] for x in evals], [x["eval_loss"] for x in evals], marker="o", ms=3,
            color="#7b1fa2", label="scene-disjoint validation loss")
    ax.plot([x["step"] for x in trains], [x["loss"] for x in trains], alpha=.24,
            color="#00897b", linewidth=.7, label="logged training loss")
    ax.scatter([best["step"]], [best["eval_loss"]], color="#d32f2f", zorder=4)
    ax.annotate(f"best {best['eval_loss']:.5f}\n{100 * reduction:.1f}% below first eval",
                (best["step"], best["eval_loss"]), xytext=(-115, 35), textcoords="offset points",
                arrowprops={"arrowstyle": "->", "lw": .8})
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Qwen3-VL-Instruct (8.77B) LoRA fine-tuning")
    ax.legend(frameon=False)
    finish(fig, "fig03_qwen_finetuning_curve.png")
    return {"first_eval_loss": first["eval_loss"], "best_eval_loss": best["eval_loss"],
            "best_step": best["step"], "relative_reduction": reduction,
            "final_eval_loss": evals[-1]["eval_loss"]}


def plot_stage1_yolo(report: dict) -> list[dict]:
    classes = report["generic_validation"]["classes"]
    rows = [{"class": x["Class"], "precision": x["Box-P"], "recall": x["Box-R"],
             "f1": x["Box-F1"], "map50": x["mAP50"], "map50_95": x["mAP50-95"]} for x in classes]
    rows.sort(key=lambda x: x["f1"])
    fig, ax = plt.subplots(figsize=(8.4, 9.3))
    y = range(len(rows))
    ax.barh(list(y), [r["f1"] for r in rows], color="#1565c0", alpha=.84, label="F1")
    ax.scatter([r["recall"] for r in rows], list(y), color="#f57c00", s=10, label="Recall", zorder=3)
    ax.set_yticks(list(y), [r["class"] for r in rows])
    ax.set_xlim(.82, 1.01)
    ax.set_xlabel("Validation metric")
    ax.set_title("Stage 1 YOLO26x class performance (343-image selection validation)")
    ax.legend(frameon=False, loc="lower right")
    finish(fig, "fig04_stage1_yolo_per_class.png")
    return rows


def plot_stage2_yolo(readiness: dict) -> list[dict]:
    selected = readiness["target_conditioned_recall_at"]["10"]
    rows = [{"class": k, "precision": v["precision"], "recall": v["recall"], "f1": v["f1"],
             "gt": v["gt"]} for k, v in selected["classes"].items()]
    rows.sort(key=lambda x: x["f1"])
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    y = range(len(rows))
    ax.barh(list(y), [r["f1"] for r in rows], color="#00897b", alpha=.84, label="F1")
    ax.scatter([r["recall"] for r in rows], list(y), color="#d32f2f", s=18, label="Recall", zorder=3)
    ax.set_yticks(list(y), [r["class"] for r in rows])
    ax.set_xlim(.72, 1.01)
    ax.set_xlabel("Operational validation metric")
    ax.set_title("Stage 2 exact-target YOLO26x at confidence 0.20, top 10")
    ax.legend(frameon=False, loc="lower right")
    finish(fig, "fig05_stage2_yolo_per_class.png")
    return rows


def plot_qwen_dataset(coverage: dict) -> None:
    panels = [
        ("Object classes", coverage["object_class_counts"]),
        ("Spatial relations", coverage["object_relation_counts"]),
        ("SAR situations", coverage["sar_situation_counts"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.3))
    for ax, (title, values) in zip(axes, panels):
        items = sorted(values.items(), key=lambda x: x[1])
        ax.barh([x[0] for x in items], [x[1] for x in items], color="#5e35b1")
        ax.set_title(title)
        ax.set_xlabel("Examples")
    fig.suptitle("Qwen3-VL fine-tuning corpus coverage (known imbalance retained)")
    finish(fig, "fig06_qwen_dataset_distribution.png")


def plot_runtime(rounds: list[dict], s1_verify: dict, s2_verify: dict) -> dict:
    q = [x["verifier_metrics"]["latency_ms"] for x in rounds
         if (x.get("verifier_metrics") or {}).get("backend") == "qwen_remote"]
    names = ["Stage 1 YOLO\nmodel-only", "Stage 2 YOLO\nmodel-only", "Qwen request\nend-to-end"]
    med = [s1_verify["benchmark"]["median_ms"], s2_verify["benchmark"]["median_ms"], statistics.median(q)]
    p95 = [s1_verify["benchmark"]["p95_ms"], s2_verify["benchmark"]["p95_ms"], percentile(q, .95)]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = list(range(3))
    ax.bar(x, med, color=["#1565c0", "#00897b", "#7b1fa2"], label="median")
    ax.scatter(x, p95, color="#d32f2f", marker="_", s=180, linewidths=2, label="p95")
    ax.set_yscale("log")
    ax.set_xticks(x, names)
    ax.set_ylabel("Latency (ms, logarithmic)")
    ax.set_title("Recorded runtime latency (paths and workloads differ)")
    ax.legend(frameon=False)
    finish(fig, "fig07_runtime_latency.png")
    return {"qwen_samples": len(q), "qwen_median_ms": statistics.median(q),
            "qwen_mean_ms": statistics.fmean(q), "qwen_p95_ms": percentile(q, .95),
            "qwen_max_ms": max(q)}


def montage(paths: list[Path], labels: list[str], name: str, cell=(640, 360)) -> None:
    canvas = Image.new("RGB", (cell[0] * 2, (cell[1] + 34) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (path, label) in enumerate(zip(paths, labels)):
        im = Image.open(path).convert("RGB")
        im.thumbnail(cell)
        panel = Image.new("RGB", cell, "#eeeeee")
        panel.paste(im, ((cell[0] - im.width) // 2, (cell[1] - im.height) // 2))
        x, y = (i % 2) * cell[0], (i // 2) * (cell[1] + 34)
        canvas.paste(panel, (x, y))
        draw.text((x + 8, y + cell[1] + 8), label, fill="black")
    canvas.save(OUT / name)


def comparison_strip(paths: list[Path], labels: list[str], name: str) -> None:
    cell = (600, 338)
    canvas = Image.new("RGB", (cell[0] * len(paths), cell[1] + 36), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (path, label) in enumerate(zip(paths, labels)):
        im = ImageOps.fit(Image.open(path).convert("RGB"), cell, method=Image.Resampling.LANCZOS)
        canvas.paste(im, (i * cell[0], 0))
        draw.text((i * cell[0] + 8, cell[1] + 9), label, fill="black")
    canvas.save(OUT / name)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-evaluation", type=Path, default=STAGE1_EVAL_DEFAULT)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    style()

    runs = result_runs()
    best = max(runs, key=lambda x: x["stage1_average"])
    qwen = plot_qwen()
    coverage = load(RAW / "qwen_dataset_coverage.json")
    s1_contract = load(RAW / "stage1_runtime_contract.json")
    s1_verify = load(RAW / "stage1_runtime_verification.json")
    s2_contract = load(RAW / "stage2_runtime_contract.json")
    s2_verify = load(RAW / "stage2_runtime_verification.json")
    readiness = load(RAW / "stage2_readiness_metrics.json")
    stage1_report = load(args.stage1_evaluation)
    with (RAW / "stage1_rounds.jsonl").open(encoding="utf-8") as f:
        rounds = [json.loads(line) for line in f if line.strip()]

    plot_score_progression(runs)
    components = plot_components(best)
    stage1_classes = plot_stage1_yolo(stage1_report)
    stage2_classes = plot_stage2_yolo(readiness)
    plot_qwen_dataset(coverage)
    runtime = plot_runtime(rounds, s1_verify, s2_verify)
    montage([
        ROOT / "images/stage1/object_mug_picnic_table.png",
        ROOT / "images/stage1/object_umbrella_trash_can.png",
        ROOT / "images/stage1/sar_sitting_trash_can.png",
        ROOT / "images/stage1/sar_prone_parking.png",
    ], ["Lost object: mug on picnic table", "Lost object: umbrella near trash can",
        "SAR: person sitting on trash can", "SAR: prone person in parking lot"],
        "fig08_stage1_detection_montage.png")
    montage([
        ROOT / "images/stage2/runtime_drive_tumbler.png",
        ROOT / "images/stage2/runtime_align_tumbler.png",
        ROOT / "images/stage2/blind_model_02.jpg",
        ROOT / "images/stage2/blind_truth_02.jpg",
    ], ["Runtime drive view: detected tumbler", "Runtime alignment view",
        "Blind review: model output", "Blind review: ground truth"],
        "fig09_stage2_detection_montage.png")
    comparison_strip([
        ROOT / "images/stage1/object_umbrella_trash_can.png",
        ROOT / "images/stage2/blind_model_02.jpg",
        ROOT / "images/stage2/blind_truth_02.jpg",
    ], ["Stage 1 detector output", "Stage 2 model output", "Stage 2 ground truth"],
        "fig10_best_detection_comparison.png")

    selected2 = readiness["target_conditioned_recall_at"]["10"]
    operational1 = stage1_report["operational"]
    metrics = {
        "scope": "MARC 2026 local practice and validation evidence; not hidden-test results",
        "qwen": {
            "checkpoint_name": "Qwen3-VL-8B-Instruct",
            "parameter_count": 8_767_123_696,
            "rounded_name": "approximately 9B parameters",
            "train_examples": coverage["num_train_sar"] + coverage["num_train_obj"],
            "validation_examples": coverage["num_val_sar"] + coverage["num_val_obj"],
            "train_results": load(RAW / "qwen_train_results.json"),
            "eval_results": load(RAW / "qwen_eval_results.json"),
            "curve": qwen,
            "coverage_complete": coverage["coverage_complete"],
            "coverage_missing": coverage["coverage_missing"],
            "runtime": runtime,
        },
        "stage1_yolo": {
            "deployed_checkpoint_sha256": s1_contract["checkpoint_sha256"],
            "classes": len(s1_contract["ordered_classes"]),
            "final_continuation_validation_map50": s1_contract["continuation"]["validation_map50"],
            "final_continuation_validation_map50_95": s1_contract["continuation"]["validation_map50_95"],
            "selection_checkpoint_operational": {k: operational1[k] for k in
                ["gt", "predictions", "matched", "false_candidates", "recall", "precision",
                 "macro_recall", "macro_precision", "macro_f1"]},
            "runtime_benchmark": s1_verify["benchmark"],
        },
        "stage2_yolo": {
            "status": s2_contract["status"],
            "deployed_checkpoint_sha256": s2_contract["checkpoint_sha256"],
            "classes": len(s2_contract["ordered_classes"]),
            "operational": {k: selected2[k] for k in
                ["gt", "predictions", "false_candidates", "recall", "precision", "macro_recall",
                 "macro_precision", "macro_f1"]},
            "tiny_recall": selected2["sizes"]["tiny"]["recall"],
            "worst_class_recall": min(v["recall"] for v in selected2["classes"].values()),
            "runtime_benchmark": s2_verify["benchmark"],
            "known_limitations": s2_contract["known_limitations"],
        },
        "practice": {
            "run_count": len(runs),
            "best_stage1": best["stage1_average"],
            "best_run": best["run"],
            "rounds": len(best["details"]),
            "component_means": components,
            "stage2_scores_observed": [r["stage2"] for r in runs if r["stage2"] > 0],
            "latest_recorded_stage2_failure": "navigation reached; GraspGen-X returned no_grasp after 3 attempts",
        },
    }
    save_json(ROOT / "data" / "metrics_summary.json", metrics)
    write_csv(ROOT / "data" / "stage1_yolo_per_class.csv", stage1_classes)
    write_csv(ROOT / "data" / "stage2_yolo_per_class.csv", stage2_classes)
    write_csv(ROOT / "data" / "practice_runs.csv", [{k: v for k, v in r.items() if k != "details"} for r in runs])

    assert metrics["qwen"]["train_examples"] == 5535
    assert metrics["qwen"]["validation_examples"] == 609
    assert abs(metrics["practice"]["best_stage1"] - 95.9562) < 1e-4
    assert len(stage1_classes) == 34 and len(stage2_classes) == 10
    manifest = {}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256" and "__pycache__" not in path.parts:
            manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / "MANIFEST.sha256").write_text(
        "".join(f"{digest}  {path}\n" for path, digest in manifest.items()), encoding="utf-8"
    )
    print(f"Built {len(list(OUT.glob('*.png')))} figures; best practice Stage 1={best['stage1_average']:.4f}")


if __name__ == "__main__":
    main()
