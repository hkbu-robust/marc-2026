"""Stage and finalize a separate optimizer-stripped Stage 1 candidate bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


REQUIRED = (
    "model.pt", "runtime_contract.json", "SOURCE.md", "LICENSE",
    "candidate_output_example.json", "runtime_verification.json",
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _stage(args) -> None:
    from ultralytics.utils.torch_utils import strip_optimizer

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite candidate bundle: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.evaluation.read_text(encoding="utf-8"))
    record = json.loads(args.training_record.read_text(encoding="utf-8"))
    selected = Path(report["selection"]["checkpoint"]).resolve()
    selected_sha = _sha256(selected)
    recorded = report["runtime_contract"]["selected_training_checkpoint"]
    if (
        selected != args.checkpoint.resolve()
        or selected_sha != record["selected_checkpoint_sha256"]
        or recorded != {"name": selected.name, "sha256": selected_sha}
    ):
        raise ValueError("selected checkpoint does not match evaluation/training provenance")
    deployment = args.output / "model.pt"
    if not strip_optimizer(selected, str(deployment)):
        raise ValueError("Ultralytics could not strip the selected checkpoint")
    deployment_sha = _sha256(deployment)
    contract = report["runtime_contract"]
    contract.update({
        "checkpoint": "model.pt", "checkpoint_sha256": deployment_sha,
        "local_checkpoint_load_verified": False, "offline_bundle_load_verified": False,
    })
    _json(args.output / "runtime_contract.json", contract)
    shutil.copy2(args.published_bundle / "LICENSE", args.output / "LICENSE")
    data = __import__("yaml").safe_load(Path(report["dataset"]).read_text(encoding="utf-8"))
    old_sha = _sha256(args.published_bundle / "model.pt")
    source = f"""# Stage 1 corrective replacement candidate

This is a separately staged, optimizer-stripped **candidate**. Human review is pending.
The published 33-class bundle at `{args.published_bundle.resolve()}` remains untouched.

## Lineage

- Selected run: `{args.training_record.parent.resolve()}`
- Selected training checkpoint: `{selected.name}`
- Full selected training-checkpoint SHA-256: `{record['selected_checkpoint_sha256']}`
- Stripped deployment SHA-256: `{deployment_sha}`
- Corrected source dataset revision: `{data['source_revision']}`
- Corrected source descriptor SHA-256: `{data['source_data_yaml_sha256']}`
- Training commit: `{record['git']['commit']}`
- Superseded published model SHA-256: `{old_sha}`

## Contract

- 34 ordered classes including ordinary `umbrella`
- Excluded exactly `long_umbrella`, `master_chef_can`, and `wood_block`
- Global all-class candidates, confidence descending, bounded by the recorded cap
- Native full-frame inference at 1920; original-image pixel `xyxy`
- No independent test split; the same validation split selected and reports this candidate
- Final replacement readiness is not claimed until the blind review is complete

## Reproduction

Evaluator command: `{json.dumps(report['evaluator']['argv'])}`

Stage command: `{json.dumps(sys.argv)}`

The runtime verification command and machine results are stored in
`runtime_verification.json`. `candidate_output_example.json` is generated through
the actual Stage 1 interface during that fresh-process verification.

## License

The bundled `LICENSE` is the AGPL-3.0 license shipped with the installed Ultralytics
distribution. Confirm team compliance or an appropriate Ultralytics license before submission.
"""
    (args.output / "SOURCE.md").write_text(source, encoding="utf-8")
    print(json.dumps({"candidate_bundle": str(args.output.resolve()), "deployment_sha256": deployment_sha}, indent=2))


def _finalize(args) -> None:
    verification_path = args.bundle / "runtime_verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    contract_path = args.bundle / "runtime_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    identity = {
        "candidate_weights_sha256": contract["checkpoint_sha256"],
        "selected_training_checkpoint_sha256": contract["selected_training_checkpoint"]["sha256"],
        "settings": contract["inference"],
        "ordered_classes": contract["ordered_classes"],
        "ignored_classes": contract["ignored_classes"],
        "dataset_identity": contract["dataset"],
    }
    if (
        not verification.get("verified")
        or Path(verification["candidate_bundle"]).resolve() != args.bundle.resolve()
        or any(verification.get(key) != value for key, value in identity.items())
        or _sha256(args.bundle / "model.pt") != contract["checkpoint_sha256"]
    ):
        raise ValueError("runtime verification identity does not match this candidate bundle")
    _json(args.bundle / "candidate_output_example.json", verification["candidate_output_example"])
    contract.update({"local_checkpoint_load_verified": True, "offline_bundle_load_verified": True})
    _json(contract_path, contract)
    missing = [name for name in REQUIRED if not (args.bundle / name).is_file()]
    if missing:
        raise ValueError(f"candidate bundle is incomplete: {missing}")
    sums = "".join(f"{_sha256(args.bundle / name)}  {name}\n" for name in REQUIRED)
    (args.bundle / "SHA256SUMS").write_text(sums, encoding="utf-8")
    print(json.dumps({"candidate_bundle": str(args.bundle.resolve()), "files": [*REQUIRED, "SHA256SUMS"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage = subparsers.add_parser("stage")
    stage.add_argument("--evaluation", required=True, type=Path)
    stage.add_argument("--training-record", required=True, type=Path)
    stage.add_argument("--checkpoint", required=True, type=Path)
    stage.add_argument("--published-bundle", required=True, type=Path)
    stage.add_argument("--output", required=True, type=Path)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args()
    _stage(args) if args.command == "stage" else _finalize(args)


if __name__ == "__main__":
    main()
