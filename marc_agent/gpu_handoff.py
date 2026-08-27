"""Unload GPU residents so CCTV YOLO, Qwen, near-field YOLO, and GraspGen-X do not overlap."""

from __future__ import annotations

import gc
import logging

log = logging.getLogger("marc_agent.gpu_handoff")


def empty_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        cuda = getattr(torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.empty_cache()
    except Exception:
        pass


def unload_cctv_yolo() -> None:
    from marc_agent.perception.cctv_yolo import unload_detector

    unload_detector()
    empty_cuda_cache()
    log.info("[gpu] unloaded CCTV YOLO")


def unload_qwen() -> None:
    """Drop the process Qwen verifier so Stage 2 pick does not share VRAM with 8B."""
    from marc_agent.perception.stage1_pipeline import reset_cached_verifier

    reset_cached_verifier()
    empty_cuda_cache()
    log.info("[gpu] unloaded Qwen")


def unload_nearfield_detector(detector) -> None:
    unload = getattr(detector, "unload", None)
    if callable(unload):
        unload()
    empty_cuda_cache()
    log.info("[gpu] unloaded near-field YOLO")


def unload_graspgenx_runtime(pick=None) -> None:
    if pick is not None:
        drop = getattr(pick, "unload_generator", None)
        if callable(drop):
            drop()
    from marc_agent.graspgenx_policy import unload_graspgenx

    unload_graspgenx()
    empty_cuda_cache()
    log.info("[gpu] unloaded GraspGen-X")


def reload_nearfield_detector(detector) -> None:
    if detector is None:
        return
    load = getattr(detector, "load", None)
    if callable(load):
        load()
    log.info("[gpu] reloaded near-field YOLO")


def before_graspgenx(detector) -> None:
    """Drop near-field YOLO before GraspGen-X / IK occupy the GPU."""
    unload_nearfield_detector(detector)


def before_reacquire(pick, detector) -> None:
    """Free GraspGen-X and bring near-field YOLO back for another localize."""
    unload_graspgenx_runtime(pick)
    reload_nearfield_detector(detector)


def after_successful_pick(pick, detector) -> None:
    """Free GraspGen-X after a basket success and keep near-field YOLO ready."""
    unload_graspgenx_runtime(pick)
    reload_nearfield_detector(detector)
