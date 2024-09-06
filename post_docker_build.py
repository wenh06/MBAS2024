from pathlib import Path

from torch_ecg.utils.download import http_get, url_is_reachable

from const import MODEL_CACHE_DIR, REMOTE_MODELS
from models import MultiHead_MBAS2024


def cache_pretrained_models():
    """Cache the pretrained models."""
    print("Caching the pretrained models...")
    if url_is_reachable("https://drive.google.com/"):
        remote_model_source = "google-drive"
    else:
        remote_model_source = "deep-psp"

    # Download the pretrained models
    model_name = "vnet-v2"
    http_get(
        url=REMOTE_MODELS[model_name]["url"][remote_model_source],
        dst_dir=MODEL_CACHE_DIR,
        filename=REMOTE_MODELS[model_name]["filename"],
        extract=True,
    )

    assert (Path(MODEL_CACHE_DIR) / "stage0-model.pth.tar").exists()
    assert (Path(MODEL_CACHE_DIR) / "stage1-model.pth.tar").exists()

    # load the pretrained models
    model, train_config = MultiHead_MBAS2024.from_checkpoint(
        Path(MODEL_CACHE_DIR) / "stage0-model.pth.tar",
        device="cpu",
    )
    model = model.to("cpu").eval()
    print("stage0-model loaded")

    del model, train_config

    model, train_config = MultiHead_MBAS2024.from_checkpoint(
        Path(MODEL_CACHE_DIR) / "stage1-model.pth.tar",
        device="cpu",
    )
    model = model.to("cpu").eval()
    print("stage1-model loaded")
    del model, train_config


if __name__ == "__main__":
    cache_pretrained_models()
    print("Done.")
