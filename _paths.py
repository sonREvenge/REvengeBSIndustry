import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR  = os.path.join(BASE_DIR, "models")
IMAGES_DIR  = os.path.join(BASE_DIR, "images")
CFG_DIR     = os.path.join(BASE_DIR, "cfg")

MODEL_MAIN  = os.path.join(MODELS_DIR, "mainInGameModel.onnx")
MODEL_TILES = os.path.join(MODELS_DIR, "tileDetector.onnx")

TRT_CACHE_DIR = os.path.join(MODELS_DIR, "trt_cache", "shared")

IMG_STATES      = os.path.join(IMAGES_DIR, "states")
IMG_STAR_DROPS  = os.path.join(IMAGES_DIR, "star_drop_types")
IMG_END_RESULTS = os.path.join(IMAGES_DIR, "end_results")


def img_state(filename: str) -> str:
    return os.path.join(IMG_STATES, filename)


def img_star_drop(filename: str) -> str:
    return os.path.join(IMG_STAR_DROPS, filename)


def img_end_result(filename: str) -> str:
    return os.path.join(IMG_END_RESULTS, filename)
