"""OCR engines for cross-checking character recognition.

Each engine is a separate subpackage exposing a class that implements
`ocr_engines.base.OCREngine`. The orchestrator in
`pipeline/step2_5_recognize.py` runs all enabled engines on the same
crops produced by Step 2 and writes per-engine JSON, then consensus
voting produces the final per-character decision.

Engines:
    - kimhannom       : reuses Step 1 cache (Kimhannom HCMUS API)
    - paddleocrv5_nom : MinhDS fine-tuned PaddleOCRv5 (Han-Nom specialist)
    - nomna_ocr       : ds4v NomNaOCR CRNN (confirmed handwritten, column-level)
"""

from .base import OCREngine, RecognitionResult

__all__ = ["OCREngine", "RecognitionResult"]
