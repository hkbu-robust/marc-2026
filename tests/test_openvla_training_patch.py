"""The OpenVLA patcher fails closed when upstream anchors drift."""

import unittest

from training.patch_openvla_finetune import patch_source


class OpenVLATrainingPatchTests(unittest.TestCase):
    def test_patch_adds_registration_validation_and_early_stop(self):
        source = '''os.environ["TOKENIZERS_PARALLELISM"] = "false"
    image_aug: bool = True                                          # Whether to train with image augmentations
@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
        image_aug=cfg.image_aug,
    )

    # [Important] Save Dataset Statistics
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    # Initialize Logging
    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
'''
        patched = patch_source(source)
        self.assertIn('OXE_DATASET_CONFIGS["marc_pnp"]', patched)
        self.assertIn("train=False", patched)
        self.assertIn("early_stopping_patience", patched)
        self.assertIn("MARC_OPENVLA_BEST_ADAPTER_DIR", patched)


if __name__ == "__main__":
    unittest.main()
