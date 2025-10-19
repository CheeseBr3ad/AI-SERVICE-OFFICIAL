from sentence_transformers import SentenceTransformer
from config.config import EMBEDDING_MODEL
from config.logger import logger
import torch

_model: SentenceTransformer | None = None  # type hint for clarity


def set_model(model_name: str):
    """
    Load the embedding model once, preferring GPU if available.
    Accepts the model name as a parameter.
    """
    global _model
    if _model is not None:
        logger.info("⚙️ Model already loaded — skipping reload.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)
    logger.info(
        f"🔍 Checking for GPU... {'✅ GPU available' if device == 'cuda' else '⚙️ No GPU found, using CPU'}"
    )

    try:
        _model = SentenceTransformer(model_name, device=device)
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🖥 Using GPU: {gpu_name}")
        logger.info(f"✅ Embedding model '{model_name}' loaded on {device.upper()}")

    except Exception as e:
        logger.error(f"❌ Failed to load model on {device.upper()}: {e}")
        logger.info("Retrying with CPU fallback...")
        _model = SentenceTransformer(model_name, device="cpu")
        logger.info(f"✅ Embedding model '{model_name}' loaded on CPU")


def get_model() -> SentenceTransformer:
    """Return the currently loaded model instance."""
    global _model
    if _model is None:
        logger.warning("⚠️ Model not yet set — please call set_model(model_name) first.")
        raise ValueError(
            "Model not loaded. Call set_model(model_name) before using it."
        )
    return _model
