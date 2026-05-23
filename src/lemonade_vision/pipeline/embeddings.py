from __future__ import annotations

import numpy as np
from PIL import Image
import open_clip
import torch


class EmbeddingModel:
    _model: open_clip.CLIP | None = None
    _preprocess = None
    _tokenizer = None

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
        img = Image.open(image_path).convert("RGB")
        tensor = self._preprocess(img).unsqueeze(0)  # type: ignore[arg-type]
        with torch.no_grad():
            features = self._model.encode_image(tensor)  # type: ignore[union-attr]
            features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        self._load()
        tokens = self._tokenizer([text])  # type: ignore[call-arg]
        with torch.no_grad():
            features = self._model.encode_text(tokens)  # type: ignore[union-attr]
            features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)
