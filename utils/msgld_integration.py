"""Integration helpers for invoking MSG-LD (MusicLDM) training/inference from
within the Music-Source-Separation-Training codebase.

These utilities allow users to keep using the familiar CLI from this project
while delegating the actual optimisation loop to the dedicated MSG-LD
implementation. The functions here take care of

* locating the sibling `MSG-LD-Pytorch2` repository,
* updating the YAML configuration with user-provided CLI overrides, and
* dispatching to the `train_musicldm.py` entrypoint shipped with MSG-LD.

The intent is to keep the surface-area small: we only make the minimum changes
required so that `train.py --model_type musicldm ...` works end-to-end. More
advanced customisation should continue to live in the MSG-LD project itself.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

# NOTE: the MSG-LD repository is expected to live next to this project under the
# same parent directory. Adjusting this constant is all that is required if the
# layout differs.
MSG_LD_REPO_NAME = "MSG-LD-Pytorch2"


class MSGIntegrationError(RuntimeError):
    """Raised when the MSG-LD integration cannot be initialised."""


def _load_module(module_name: str, error_message: str):
    """Import ``module_name`` and raise :class:`MSGIntegrationError` on failure."""

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - optional dependency handling
        raise MSGIntegrationError(error_message) from exc


OmegaConf = _load_module(
    "omegaconf",
    "MSG-LD integration requires omegaconf. Install it in the current environment.",
).OmegaConf


def _resolve_repo_root() -> Path:
    """Return the absolute path to the MSG-LD repository.

    The function walks from the current file (utils/msgld_integration.py) up to
    the project root and then appends ``MSG_LD_REPO_NAME``. A descriptive error
    is raised when the expected directory is missing so that the caller can
    surface the problem to the user.
    """

    music_source_root = Path(__file__).resolve().parents[1]
    candidate = (music_source_root.parent / MSG_LD_REPO_NAME).resolve()
    if not candidate.exists():
        raise MSGIntegrationError(
            f"Expected MSG-LD repository at '{candidate}'.\n"
            "Please make sure both repositories share the same parent folder or "
            "update MSG_LD_REPO_NAME accordingly."
        )
    return candidate


def _ensure_msgld_importable(repo_root: Path) -> None:
    """Add the MSG-LD repository (and its src directory) to ``sys.path``."""

    for rel in (Path("."), Path("src")):
        full = (repo_root / rel).resolve()
        if str(full) not in sys.path:
            sys.path.insert(0, str(full))


def _as_list_or_str(values: Sequence[str]) -> Sequence[str] | str:
    """Return a single string if only one item is present, otherwise the list.

    MSG-LD configs accept either a string or a list for dataset paths. This
    helper keeps the resulting YAML compact when only one folder is supplied.
    """

    if not values:
        return values
    return values if len(values) > 1 else values[0]


def _update_nested(mapping: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    """Update ``mapping`` with ``value`` using a dotted key ("a.b.c")."""

    parts = dotted_key.split(".")
    cursor: MutableMapping[str, Any] = mapping
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], Mapping):
            cursor[part] = {}
        cursor = cursor[part]  # type: ignore[assignment]
    cursor[parts[-1]] = value


def _apply_overrides(config: MutableMapping[str, Any], overrides: Mapping[str, Any]) -> None:
    """Apply dotted overrides to the configuration mapping in-place."""

    for key, value in overrides.items():
        _update_nested(config, key, value)


def _export_config(config: Mapping[str, Any], results_path: Path) -> Path:
    """Persist the (possibly modified) configuration next to the results folder.

    The returned path is later forwarded to MSG-LD for reproducibility. Having a
    copy inside the run directory mirrors how the rest of this project stores
    metadata for other model families.
    """

    cfg_dir = results_path / "msgld_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "resolved_config.yaml"
    OmegaConf.save(config=OmegaConf.create(config), f=str(cfg_path))
    return cfg_path


def train_with_msgld(args: Any) -> None:
    """Entry point used by ``train.py`` when ``--model_type musicldm`` is set.

    Parameters
    ----------
    args:
        Namespace produced by ``utils.settings.parse_args_train``. Only a subset
        of attributes is required (``config_path``, ``results_path``,
        ``device_ids``, ``data_path``, ``valid_path``, ``num_workers``,
        ``seed``, ``wandb_key``). The rest is ignored and left to the MSG-LD
        training code.
    """

    repo_root = _resolve_repo_root()
    _ensure_msgld_importable(repo_root)

    if getattr(args, "wandb_offline", False):
        os.environ.setdefault("WANDB_MODE", "offline")
    elif getattr(args, "wandb_key", ""):
        os.environ.setdefault("WANDB_API_KEY", str(args.wandb_key).strip())

    train_musicldm = _load_module(
        "train_musicldm",
        "Unable to import MSG-LD training entry point. Make sure MSG-LD-Pytorch2 "
        "is available and its dependencies installed.",
    )
    if not hasattr(train_musicldm, "main"):
        raise MSGIntegrationError(
            "MSG-LD training module does not expose a 'main' function."
        )
    msgld_train = getattr(train_musicldm, "main")

    # Load and convert the YAML configuration used by MSG-LD. ``to_container``
    # ensures that the downstream code receives a plain dict/list structure.
    raw_cfg = OmegaConf.load(args.config_path)
    cfg: MutableMapping[str, Any] = OmegaConf.to_container(raw_cfg, resolve=True)  # type: ignore[assignment]

    results_path = Path(args.results_path).resolve()
    results_path.mkdir(parents=True, exist_ok=True)

    torch = _load_module(
        "torch",
        "PyTorch is required for MSG-LD integration. Install it before running musicldm training.",
    )

    requested_devices = list(getattr(args, "device_ids", []))
    use_gpu = bool(requested_devices) and torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    if use_gpu:
        devices_override: Any = requested_devices
    else:
        devices_override = len(requested_devices) or 1

    overrides: Dict[str, Any] = {
        "log_directory": str(results_path),
        "mode": "train",
        "dev": False,
        "trainer.accelerator": accelerator,
        "trainer.devices": devices_override,
        "data.params.num_workers": getattr(args, "num_workers", 0),
    }

    # Dataset paths (train/valid) come from the universal CLI. We keep the
    # original YAML values when the user did not provide explicit overrides.
    if getattr(args, "data_path", None):
        overrides["data.params.path.train_data"] = _as_list_or_str(list(args.data_path))
    if getattr(args, "valid_path", None):
        overrides["data.params.path.valid_data"] = _as_list_or_str(list(args.valid_path))

    # Seed handling: MSG-LD fixes the seed at 0, so we explicitly reseed here to
    # honour the user request before handing control over to Lightning.
    pytorch_lightning = _load_module(
        "pytorch_lightning",
        "MSG-LD integration requires pytorch-lightning. Install it in the current environment.",
    )
    if not hasattr(pytorch_lightning, "seed_everything"):
        raise MSGIntegrationError(
            "pytorch_lightning.seed_everything is required but missing. Check your installation."
        )
    seed_everything = getattr(pytorch_lightning, "seed_everything")

    seed_everything(int(getattr(args, "seed", 0)))

    _apply_overrides(cfg, overrides)

    # Persist the resolved configuration for traceability and provide the
    # resulting dictionary to the MSG-LD training entry point.
    exported_cfg_path = _export_config(cfg, results_path)
    cfg["integration_metadata"] = {
        "source_repo": "Music-Source-Separation-Training",
        "config_path": str(exported_cfg_path.relative_to(results_path)),
        "cli_args": {k: getattr(args, k) for k in vars(args)}
    }

    # The MSG-LD training code expects a regular mapping. Dump the final config
    # to assist debugging if something goes wrong.
    with (results_path / "msgld_resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)

    msgld_train(cfg)
