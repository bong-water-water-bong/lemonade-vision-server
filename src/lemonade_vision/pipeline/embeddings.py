from __future__ import annotations

from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image


class EmbeddingModel:
    _model: Any | None = None
    _preprocess: Any | None = None
    _tokenizer: Any | None = None

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is None:
            model, _, preprocess = open_clip.create_model_and_transforms(
                self._model_name, pretrained=self._pretrained
            )
            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self._model_name)

    def encode_image(self, image_path: str) -> np.ndarray:
        self._load()
        model = self._model
        preprocess = self._preprocess
        assert model is not None
        assert preprocess is not None
        img = Image.open(image_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features = model.encode_image(tensor)
            features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        self._load()
        model = self._model
        tokenizer = self._tokenizer
        assert model is not None
        assert tokenizer is not None
        tokens = tokenizer([text])
        with torch.no_grad():
            features = model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)
