"""Model factory: build the right model from a config dictionary."""

from src.models.cnn import CNN1D
from src.models.hybrid import HybridCNNTransformer
from src.models.transformer import PatchTransformer

MODELS = {
    "cnn": CNN1D,
    "transformer": PatchTransformer,
    "hybrid": HybridCNNTransformer,
}


def build_model(config):
    """config['model'] picks the class, config['model_args'] are its keyword arguments."""
    model_class = MODELS[config["model"]]
    return model_class(**config.get("model_args", {}))
