"""Frozen held-out and public evaluation sets for the selector comparison (#87).

Pipeline, all outputs under ``./work/experiments/ablations/05-frozen-eval-sets/``:

1. ``training_record``: what the deployed Qwen adapter was trained on, read once
   from the text-only Arrow caches (strata counts, leakage index).
2. ``heldout specs``: generator GT -> one case spec per uniquely described target.
3. ``detect_pass`` (inside the frozen agent image): deployed detector + refinement.
4. ``heldout assemble`` / ``public build``: gold candidate IDs -> case folders.
5. ``case_store freeze``: checksummed manifest. Arms load cases with ``case_store.load_set``.
"""
