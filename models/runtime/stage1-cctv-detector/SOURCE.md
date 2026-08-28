# Stage 1 CCTV detector corrective replacement

This optimizer-stripped 34-class detector was approved for promotion on 2026-08-29.
The human approver explicitly waived the optional blind-review workflow. No completed
blind-review JSON was fabricated. The superseded 33-class release remains immutable
at Hugging Face revision `9227ecd11069d103c7dabdd287c6fc20a83eb6c1`.

## Lineage

- Selected run: `/home/marc2026/marc-2026-work/runs/stage1-cctv-yolo26x-1920-umbrella-ceccc62`
- Selected training checkpoint: `last.pt`
- Full selected training-checkpoint SHA-256: `12712d12e3321dffeb49296285a7d99f8fab12b46aa99a808b5d3cd1da80da22`
- Stripped deployment SHA-256: `d799aacb8df1dce3372f5ec29495e7bb467898e453fc8d8fa32bdc7ecaff738e`
- Corrected source dataset revision: `ceccc62d27e7fc186fd186e66632da38aeac6668`
- Corrected source descriptor SHA-256: `a9277d45dfaeefa3aaec34f6cf82218c4b362dd7b19c06271f8dc6ff417b1f9a`
- Training commit: `4010e06cce663c0c8c9ff3617731511ffd89b5a4`
- Superseded published model SHA-256: `06a0abac38a897c0d3c72392154699305a0855bbc12779c6a7bdfff7fe803e2f`

## Permanent archive

- Private Hugging Face model repository:
  <https://huggingface.co/roBUst-marc-2026/stage1-cctv-detector>
- Immutable replacement revision:
  `cb5a918f5bcd2c6561ef367fd275cbba49cbcf9c`

## Contract

- 34 ordered classes including ordinary `umbrella`
- Excluded exactly `long_umbrella`, `master_chef_can`, and `wood_block`
- Global all-class candidates, confidence descending, bounded by the recorded cap
- Native full-frame inference at 1920; original-image pixel `xyxy`
- No independent test split; the same validation split selected and reports this candidate
- Human approval on 2026-08-29 explicitly waived blind review and authorized promotion

## Reproduction

Evaluator command: `["/home/marc2026/marc-2026-robust/training/evaluate_cctv_detector.py", "/home/marc2026/marc-2026-work/runs/stage1-cctv-yolo26x-1920-umbrella-ceccc62/weights/best.pt", "/home/marc2026/marc-2026-work/runs/stage1-cctv-yolo26x-1920-umbrella-ceccc62/weights/epoch0.pt", "/home/marc2026/marc-2026-work/runs/stage1-cctv-yolo26x-1920-umbrella-ceccc62/weights/epoch10.pt", "/home/marc2026/marc-2026-work/runs/stage1-cctv-yolo26x-1920-umbrella-ceccc62/weights/epoch20.pt", "/home/marc2026/marc-2026-work/runs/stage1-cctv-yolo26x-1920-umbrella-ceccc62/weights/epoch30.pt", "/home/marc2026/marc-2026-work/runs/stage1-cctv-yolo26x-1920-umbrella-ceccc62/weights/epoch40.pt", "/home/marc2026/marc-2026-work/runs/stage1-cctv-yolo26x-1920-umbrella-ceccc62/weights/epoch50.pt", "/home/marc2026/marc-2026-work/runs/stage1-cctv-yolo26x-1920-umbrella-ceccc62/weights/epoch60.pt", "/home/marc2026/marc-2026-work/runs/stage1-cctv-yolo26x-1920-umbrella-ceccc62/weights/epoch70.pt", "/home/marc2026/marc-2026-work/runs/stage1-cctv-yolo26x-1920-umbrella-ceccc62/weights/epoch80.pt", "/home/marc2026/marc-2026-work/runs/stage1-cctv-yolo26x-1920-umbrella-ceccc62/weights/epoch90.pt", "/home/marc2026/marc-2026-work/runs/stage1-cctv-yolo26x-1920-umbrella-ceccc62/weights/last.pt", "--data", "/home/marc2026/marc-2026-work/datasets/processed/robust-stage1-yolo34-ceccc62/data.yaml", "--output", "/home/marc2026/marc-2026-work/evaluations/stage1-cctv-yolo26x-1920-umbrella-ceccc62/evaluation_report.json", "--training-record", "/home/marc2026/marc-2026-work/runs/stage1-cctv-yolo26x-1920-umbrella-ceccc62/training_record.json", "--old-model", "/home/marc2026/marc-2026-robust/models/runtime/stage1-cctv-detector/model.pt", "--imgsz", "1920", "--batch", "4", "--device", "0", "--confidence", "0.01", "0.02", "0.05", "0.1", "0.2", "0.3", "--candidate-limits", "20", "50", "--nms-iou", "0.5", "0.7", "--quantize", "fp16"]`

Stage command: `["/home/marc2026/marc-2026-robust/training/stage1_candidate_bundle.py", "stage", "--evaluation", "/home/marc2026/marc-2026-work/evaluations/stage1-cctv-yolo26x-1920-umbrella-ceccc62/evaluation_report.json", "--training-record", "/home/marc2026/marc-2026-work/runs/stage1-cctv-yolo26x-1920-umbrella-ceccc62/training_record.json", "--checkpoint", "/home/marc2026/marc-2026-work/runs/stage1-cctv-yolo26x-1920-umbrella-ceccc62/weights/last.pt", "--published-bundle", "/home/marc2026/marc-2026-robust/models/runtime/stage1-cctv-detector", "--output", "/home/marc2026/marc-2026-work/candidates/stage1-cctv-detector-replacement-ceccc62"]`

The runtime verification command and machine results are stored in
`runtime_verification.json`. `candidate_output_example.json` is generated through
the actual Stage 1 interface during that fresh-process verification.

## License

The bundled `LICENSE` is the AGPL-3.0 license shipped with the installed Ultralytics
distribution. Confirm team compliance or an appropriate Ultralytics license before submission.
