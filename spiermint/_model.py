from __future__ import annotations

import json
from importlib import resources
from os import getenv
from pathlib import Path
from shutil import copyfileobj
from urllib.request import Request, urlopen

import torch

from ._mint.model.esm2 import ESM2

CHECKPOINT_URL = "https://huggingface.co/varunullanat2012/mint/resolve/main/mint.ckpt"
CHECKPOINT_NAME = "mint.ckpt"
CACHE_DIR_ENV = "SPIERMINT_CACHE_DIR"
CONFIG_RESOURCE = "esm2_t33_650M_UR50D.json"


def _checkpoint_path() -> Path:
    if custom_cache := getenv(CACHE_DIR_ENV):
        return Path(custom_cache).expanduser() / CHECKPOINT_NAME

    cache_root = Path(getenv("XDG_CACHE_HOME") or "~/.cache").expanduser()
    return cache_root / "spiermint" / CHECKPOINT_NAME


def _ensure_checkpoint() -> Path:
    checkpoint = _checkpoint_path()
    if checkpoint.exists():
        return checkpoint

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(f"{checkpoint.name}.tmp")
    request = Request(CHECKPOINT_URL, headers={"User-Agent": "spiermint"})
    with urlopen(request) as response, temporary.open("wb") as handle:
        copyfileobj(response, handle)
    temporary.replace(checkpoint)
    return checkpoint


def _load_config() -> dict:
    resource = resources.files("spiermint.data").joinpath(CONFIG_RESOURCE)
    return json.loads(resource.read_text())


def load_model(device: torch.device):
    config = _load_config()
    checkpoint = _ensure_checkpoint()
    model = ESM2(
        num_layers=config["encoder_layers"],
        embed_dim=config["encoder_embed_dim"],
        attention_heads=config["encoder_attention_heads"],
        token_dropout=config["token_dropout"],
        use_multimer=True,
    )
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint_data.get("state_dict", checkpoint_data)
    state_dict = {key.removeprefix("model."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint
