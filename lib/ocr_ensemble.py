"""Multi-model OCR ensemble for Nom character recognition.

Architecture:
  1. Each OCR model implements predict(crop_path) -> [(char, confidence)]
  2. Ensemble aggregates results via voting + dictionary filter
  3. Best char = highest agreement among models AND exists in dictionary

Usage:
  ensemble = OCREnsemble(qn_to_nom=qn_to_nom)
  ensemble.add_model(KimhannomOCR())        # existing API
  ensemble.add_model(LocalCNNOCR(ckpt))     # trained on labeled data
  best_char = ensemble.predict(crop_path, qn_syllable="dao")
"""

from pathlib import Path


class OCRModel:
    """Base class for OCR models."""

    name: str = "base"

    def predict(self, crop_path: str) -> list[tuple[str, float]]:
        """Predict character from crop image.

        Returns: [(char, confidence)] sorted by confidence descending.
        """
        raise NotImplementedError


class KimhannomOCR(OCRModel):
    """Use pre-computed OCR result from Kimhannom API cache.

    Doesn't call API — reads ocr_char already stored in detection JSON.
    """

    name = "kimhannom"

    def __init__(self):
        pass

    def predict_from_cache(self, ocr_char: str | None) -> list[tuple[str, float]]:
        """Return cached OCR result as prediction."""
        if ocr_char:
            return [(ocr_char, 0.9)]
        return []


class LocalCNNOCR(OCRModel):
    """CNN classifier trained on labeled Nom crops.

    Train: python -m embedding.train_embedding
    Input: 64x64 grayscale crop
    Output: top-k character predictions with confidence
    """

    name = "local_cnn"

    def __init__(self, model_path: str | None = None, class_map_path: str | None = None):
        self.model = None
        self.class_map = None
        self.idx_to_char = None

        if model_path and Path(model_path).exists():
            self._load(model_path, class_map_path)

    def _load(self, model_path: str, class_map_path: str | None):
        try:
            import json
            import torch
            import torch.nn.functional as F
            from torchvision import models, transforms

            # Load class map
            if class_map_path and Path(class_map_path).exists():
                with open(class_map_path) as f:
                    self.class_map = json.load(f)
                self.idx_to_char = {v["class_id"]: k for k, v in self.class_map.items()}
                num_classes = len(self.class_map)
            else:
                return

            # Load model
            model = models.resnet18(weights=None)
            model.conv1 = torch.nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            self.model = model

            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ])
        except Exception:
            self.model = None

    def predict(self, crop_path: str, top_k: int = 5) -> list[tuple[str, float]]:
        if self.model is None or self.idx_to_char is None:
            return []

        try:
            import torch
            import torch.nn.functional as F
            from PIL import Image

            img = Image.open(crop_path).convert("L")
            tensor = self.transform(img).unsqueeze(0)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)[0]
                top_probs, top_idx = probs.topk(top_k)

            results = []
            for prob, idx in zip(top_probs, top_idx):
                char = self.idx_to_char.get(idx.item())
                if char:
                    results.append((char, float(prob)))
            return results
        except Exception:
            return []


class OCREnsemble:
    """Aggregate predictions from multiple OCR models.

    Voting strategy:
      1. Collect predictions from all models
      2. Score each char: sum of confidences across models
      3. Bonus if char exists in dictionary for the given QN syllable
      4. Return best char
    """

    def __init__(self, qn_to_nom: dict[str, list[str]] | None = None):
        self.models: list[OCRModel] = []
        self.qn_to_nom = qn_to_nom or {}

    def add_model(self, model: OCRModel):
        self.models.append(model)

    def predict(
        self,
        crop_path: str,
        qn_syllable: str = "",
        ocr_char_cached: str | None = None,
    ) -> str | None:
        """Get best OCR prediction from ensemble.

        Args:
            crop_path: Path to crop image.
            qn_syllable: QN syllable for dictionary bonus.
            ocr_char_cached: Pre-computed OCR char from Kimhannom API.

        Returns: Best predicted character, or None.
        """
        # Collect all predictions
        char_scores: dict[str, float] = {}

        for model in self.models:
            if isinstance(model, KimhannomOCR):
                preds = model.predict_from_cache(ocr_char_cached)
            else:
                preds = model.predict(crop_path)

            for char, conf in preds:
                char_scores[char] = char_scores.get(char, 0) + conf

        if not char_scores:
            return ocr_char_cached

        # Dictionary bonus: chars that map to the QN syllable get +0.5
        dict_candidates = set(self.qn_to_nom.get(qn_syllable.lower(), []))
        for char in char_scores:
            if char in dict_candidates:
                char_scores[char] += 0.5

        # Return highest scoring char
        best = max(char_scores, key=char_scores.get)
        return best
