"""Load the published checkpoint into MLX and save MLX weights back in the published export layout.

A saved directory is a complete SentenceTransformers export (runtime files at the root, the module in
``0_BekkoInference/``) whose ``model.safetensors`` uses the original parameter names in fp32, so the
published ``BekkoSentenceTransformer`` loads it with ``strict=True``.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .config import (
    MODULE_DIR,
    REPO_ID,
    REVISION,
    RUNTIME_FILES,
    SNAPSHOT_PATTERNS,
    BackboneConfig,
    InferenceSettings,
)
from .encoding import BekkoTokenizer
from .model import BekkoMLX


def resolve(path_or_repo: str | os.PathLike = REPO_ID, revision: str | None = REVISION, *,
            local_files_only: bool = False) -> Path:
    """A local export directory, or a pinned Hub snapshot (revision defaults to the pinned v0 17M commit)."""
    path = Path(path_or_repo).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(path_or_repo), revision=revision, local_files_only=local_files_only,
                                  allow_patterns=SNAPSHOT_PATTERNS, ignore_patterns=["onnx_browser/*"]))


def _module_dir(root: Path) -> Path:
    modules = root / "modules.json"
    if modules.exists():
        entries = json.loads(modules.read_text())
        if len(entries) != 1 or not entries[0]["type"].endswith("BekkoInference"):
            raise ValueError("expected a single-module Bekko v0 export")
        return root / entries[0]["path"]
    if (root / "inference_config.json").exists():
        return root
    return root / MODULE_DIR


def _configs(module: Path):
    config = BackboneConfig.from_dict(json.loads((module / "backbone_config.json").read_text()))
    settings = InferenceSettings.from_dict(json.loads((module / "inference_config.json").read_text()),
                                           config.max_position_embeddings)
    return config, settings


def load_tokenizer(path_or_repo: str | os.PathLike = REPO_ID, revision: str | None = REVISION, *,
                   local_files_only: bool = False) -> BekkoTokenizer:
    module = _module_dir(resolve(path_or_repo, revision, local_files_only=local_files_only))
    config, settings = _configs(module)
    return BekkoTokenizer(module / "tokenizer", settings, config.max_position_embeddings)


def load(path_or_repo: str | os.PathLike = REPO_ID, revision: str | None = REVISION, *, dtype=mx.float32,
         local_files_only: bool = False) -> BekkoMLX:
    """Build BekkoMLX from published safetensors (strict name/shape match).

    ``dtype`` sets encoder parameters and compute (fp32 default; bf16 halves memory, approximate).
    Task heads stay fp32 in every mode. The tokenizer is attached as ``model.tokenizer``.
    """
    root = resolve(path_or_repo, revision, local_files_only=local_files_only)
    module = _module_dir(root)
    config, settings = _configs(module)
    model = BekkoMLX(config, settings)
    weights = mx.load(str(module / "model.safetensors"))
    expected = dict(tree_flatten(model.parameters()))
    if set(weights) != set(expected):
        raise ValueError(f"checkpoint/parameter mismatch: missing {sorted(set(expected) - set(weights))[:5]}, "
                         f"unexpected {sorted(set(weights) - set(expected))[:5]}")
    model.load_weights([(k, v.astype(mx.float32)) for k, v in weights.items()], strict=True)
    if dtype != mx.float32:
        model.encoder.set_dtype(dtype)
    model.set_compute_dtype(dtype)
    model.tokenizer = BekkoTokenizer(module / "tokenizer", settings, config.max_position_embeddings)
    model.source_dir = root
    mx.eval(model.parameters())
    return model.eval()


def state_dict(model: BekkoMLX) -> dict[str, mx.array]:
    """Parameters under their original PyTorch names, fp32."""
    return {name: value.astype(mx.float32) for name, value in tree_flatten(model.parameters())}


def save(model: BekkoMLX, directory: str | os.PathLike, *, provenance: dict | None = None) -> Path:
    """Write a published-runtime-loadable export: runtime files, configs, tokenizer, fp32 safetensors.

    Runtime and tokenizer files are copied from ``model.source_dir``; configs are written from the
    model. Refuses to write into the source export itself.
    """
    out = Path(directory)
    source = Path(model.source_dir) if model.source_dir is not None else None
    if source is not None and out.resolve() == source.resolve():
        raise ValueError("refusing to overwrite the source checkpoint")
    module = out / MODULE_DIR
    module.mkdir(parents=True, exist_ok=True)
    if source is not None:
        for name in RUNTIME_FILES:
            if (source / name).exists():
                shutil.copyfile(source / name, out / name)
        src_module = _module_dir(source)
        shutil.copytree(src_module / "tokenizer", module / "tokenizer", dirs_exist_ok=True)
    (module / "backbone_config.json").write_text(json.dumps(model.config.raw, indent=2))
    (module / "inference_config.json").write_text(json.dumps(model.settings.raw, indent=4))
    weights = state_dict(model)
    mx.eval(weights)
    # Atomic replace; the temporary name keeps the .safetensors suffix MLX would otherwise append.
    temporary = module / f".{uuid.uuid4().hex}.safetensors"
    try:
        mx.save_safetensors(str(temporary), weights, metadata={"format": "pt"})
        os.replace(temporary, module / "model.safetensors")
    finally:
        temporary.unlink(missing_ok=True)
    info = {"writer": "rh17m.bekko_mlx", "base_repo": REPO_ID, "base_revision": REVISION,
            "source_dir": str(source) if source else None, **(provenance or {})}
    (out / "rh17m_bekko_mlx.json").write_text(json.dumps(info, indent=2))
    return out
