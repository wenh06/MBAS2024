# Dummy folder following the requirements of the MBAS2024 Challenge

Since we are using the same model for validation and testing, we will treat the cli argument `model_pth` of `predict.py` as a dummy argument,
and use our pre-defined path (see the environment variable `MODEL_CACHE_DIR` of the `Dockerfile`) to load the models.
