#!/usr/bin/env python3
"""Add MARC RLDS registration, validation, and early stopping to OpenVLA finetune.py."""

from __future__ import annotations

import argparse
from pathlib import Path


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"expected one OpenVLA anchor, found {source.count(old)}: {old[:60]!r}")
    return source.replace(old, new, 1)


def patch_source(source: str) -> str:
    source = _replace_once(
        source,
        'os.environ["TOKENIZERS_PARALLELISM"] = "false"\n',
        '''os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Local RLDS schema; keeps the upstream checkout itself untouched.
from prismatic.vla.datasets.rlds.oxe.configs import (  # noqa: E402
    ActionEncoding,
    OXE_DATASET_CONFIGS,
    StateEncoding,
)
from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS  # noqa: E402


def marc_pnp_dataset_transform(trajectory):
    return trajectory


OXE_DATASET_CONFIGS["marc_pnp"] = {
    "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": ["state"],
    "state_encoding": StateEncoding.JOINT,
    "action_encoding": ActionEncoding.EEF_POS,
}
OXE_STANDARDIZATION_TRANSFORMS["marc_pnp"] = marc_pnp_dataset_transform
''',
    )
    source = _replace_once(
        source,
        '    image_aug: bool = True                                          # Whether to train with image augmentations\n',
        '''    image_aug: bool = True                                          # Whether to train with image augmentations
    use_val_set: bool = True                                        # Evaluate the episode-level `val` split
    val_steps: int = 500                                            # Validation interval in optimizer steps
    val_batches: int = 32                                           # Bounded validation cost
    early_stopping_patience: int = 5                                # Validation checks without improvement
    early_stopping_min_delta: float = 1e-4                          # Minimum useful validation loss decrease
''',
    )
    source = _replace_once(
        source,
        '@draccus.wrap()\ndef finetune(cfg: FinetuneConfig) -> None:\n',
        '''def run_validation(vla, dataloader, device_id, max_batches):
    losses = []
    vla.eval()
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"].to(device_id),
                )
            losses.append(float(output.loss.item()))
            if index + 1 >= max(1, int(max_batches)):
                break
    totals = torch.tensor(
        [sum(losses), len(losses)], dtype=torch.float64,
        device=torch.device("cuda", device_id),
    )
    dist.all_reduce(totals)
    vla.train()
    if totals[1].item() == 0:
        raise RuntimeError("validation split yielded no batches")
    return float((totals[0] / totals[1]).item())


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
''',
    )
    source = _replace_once(
        source,
        '''        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics''',
        '''        image_aug=cfg.image_aug,
    )
    val_dataset = None
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            train=False,
            image_aug=False,
        )

    # [Important] Save Dataset Statistics''',
    )
    source = _replace_once(
        source,
        '''    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging''',
        '''    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
        )

    # Initialize Logging''',
    )
    source = _replace_once(
        source,
        '''    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:''',
        '''    # Train!
    best_val_loss = float("inf")
    bad_val_checks = 0
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:''',
    )
    source = _replace_once(
        source,
        '''            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:''',
        '''            if (
                val_dataloader is not None
                and gradient_step_idx > 0
                and gradient_step_idx % cfg.val_steps == 0
                and (batch_idx + 1) % cfg.grad_accumulation_steps == 0
            ):
                val_loss = run_validation(vla, val_dataloader, device_id, cfg.val_batches)
                improved = val_loss < best_val_loss - cfg.early_stopping_min_delta
                if distributed_state.is_main_process:
                    wandb.log({"val_loss": val_loss}, step=gradient_step_idx)
                    print(f"Validation loss at step {gradient_step_idx}: {val_loss:.6f}")
                if improved:
                    best_val_loss = val_loss
                    bad_val_checks = 0
                    if distributed_state.is_main_process:
                        best_dir = Path(os.environ.get(
                            "MARC_OPENVLA_BEST_ADAPTER_DIR", str(adapter_dir / "best")
                        ))
                        os.makedirs(best_dir, exist_ok=True)
                        processor.save_pretrained(best_dir)
                        vla.module.save_pretrained(best_dir)
                        save_dataset_statistics(vla_dataset.dataset_statistics, best_dir)
                    dist.barrier()
                else:
                    bad_val_checks += 1
                if (
                    cfg.early_stopping_patience > 0
                    and bad_val_checks >= cfg.early_stopping_patience
                ):
                    print(f"Early stopping at step {gradient_step_idx}; best val loss={best_val_loss:.6f}")
                    break

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:''',
    )
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    source = Path(args.source).read_text(encoding="utf-8")
    Path(args.output).write_text(patch_source(source), encoding="utf-8")


if __name__ == "__main__":
    main()
