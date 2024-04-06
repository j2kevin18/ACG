from .network import *
from .common import *

MODELS = {
    "ACG": ACG
}


def load_model(name="ACG"):
    assert name in MODELS.keys(), f"Model name can only be one of {MODELS.keys()}."
    print(f"Using model: '{name}'")
    return MODELS[name]
