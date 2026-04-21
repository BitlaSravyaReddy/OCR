from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
import logging
import os

import cv2

from ocr_pipeline import HybridOCR, OCRPageResult, StructuredExtractor

LOGGER = logging.getLogger(__name__)
_STRUCTURER = StructuredExtractor()
_EXTRACTOR_CACHE: Dict[str, HybridOCR] = {}


def _rotate_bgr(image: Any, angle: int) -> Any:
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported angle: {angle}")


def run_paddle_ocr(
    file_path: Path,
    languages: Optional[Sequence[str]] = None,
    pdf_dpi: int = 300,
    min_token_confidence: float = 0.20,
) -> Tuple[Dict[str, Any], str]:
    """
    Run PaddleOCR-based extraction and return (structured_json, extracted_text).
    For image uploads, tries 0/90/180/270 and picks best-scoring candidate.
    """
    if languages:
        langs = [lang.strip() for lang in languages if str(lang).strip()]
    else:
        configured = os.getenv("PADDLE_OCR_LANGS", "en")
        langs = [part.strip() for part in configured.split(",") if part.strip()]
        if not langs:
            langs = ["en"]
    cache_key = f"{','.join(sorted(langs))}|{int(pdf_dpi)}|{float(min_token_confidence):.3f}"
    extractor = _EXTRACTOR_CACHE.get(cache_key)
    if extractor is None:
        extractor = HybridOCR(
            languages=langs,
            render_dpi=pdf_dpi,
            min_token_confidence=min_token_confidence,
        )
        _EXTRACTOR_CACHE[cache_key] = extractor
        LOGGER.info("Paddle extractor cache miss | key=%s", cache_key)
    else:
        LOGGER.info("Paddle extractor cache hit | key=%s", cache_key)

    if file_path.suffix.lower() == ".pdf":
        pages = extractor.process_pdf_pages(file_path)
    else:
        image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unable to read image for Paddle OCR: {file_path}")

        best_candidate = None
        best_angle = 0
        best_score = -1.0

        # Fast path: run 0-degree first; only try other angles when confidence is weak.
        candidate_0 = extractor.ocr_image(image, source_label=f"{file_path.name}:rot0")
        score_0 = float(getattr(candidate_0, "ranking_score", 0.0))
        best_candidate = candidate_0
        best_angle = 0
        best_score = score_0

        if score_0 < 0.72:
            for angle in (90, 180, 270):
                rotated = _rotate_bgr(image, angle)
                candidate = extractor.ocr_image(rotated, source_label=f"{file_path.name}:rot{angle}")
                score = float(getattr(candidate, "ranking_score", 0.0))
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
                    best_angle = angle

        if best_candidate is None:
            pages = extractor.process_image_pages(file_path)
        else:
            LOGGER.info(
                "Paddle rotation selected | file=%s | angle=%s | text_len=%s | score=%.4f",
                file_path,
                best_angle,
                len(best_candidate.text or ""),
                best_score,
            )
            pages = [
                OCRPageResult(
                    page_number=1,
                    text=best_candidate.text,
                    source=best_candidate.source,
                    mean_confidence=best_candidate.mean_confidence,
                    lexical_quality=best_candidate.lexical_quality,
                    lines=best_candidate.lines,
                )
            ]

    extracted_text = "\n\n".join(page.text for page in pages if page.text)
    extracted_json = _STRUCTURER.extract_document_structure(file_path, pages)
    return extracted_json, extracted_text
