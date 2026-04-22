from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import logging
import os
import time

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_EAGER_INIT", "False")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

from ocr.fusion import fuse_ocr_outputs
from ocr.paddle_ocr import run_paddle_ocr
from ocr.tesseract_ocr import run_tesseract_ocr

LOGGER = logging.getLogger(__name__)


def supported_upload_extensions() -> List[str]:
    """Supported upload extensions for the UI."""
    return ["pdf", "png", "jpg", "jpeg"]


def run_multi_ocr(
    file_path: Path,
    languages: Optional[Sequence[str]] = None,
    pdf_dpi: int = 220,
    min_token_confidence: float = 0.20,
    tesseract_lang: str = "eng",
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """
    Run PaddleOCR + Tesseract in parallel, fuse outputs, and return:
    (fused_json, fused_text, debug_payload)
    """
    paddle_json: Dict[str, Any] = {}
    paddle_text = ""
    tesseract_text = ""
    tesseract_boxes: Optional[List[Dict[str, str]]] = None
    errors: Dict[str, str] = {}
    start_ts = time.perf_counter()
    LOGGER.info("Multi-OCR start | file=%s", file_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        paddle_future = executor.submit(
            run_paddle_ocr,
            file_path,
            languages,
            pdf_dpi,
            min_token_confidence,
        )
        tess_future = executor.submit(
            run_tesseract_ocr,
            file_path,
            pdf_dpi,
            tesseract_lang,
        )

        try:
            paddle_json, paddle_text = paddle_future.result()
            LOGGER.info(
                "Paddle OCR done | file=%s | text_len=%s | rows=%s",
                file_path,
                len(paddle_text or ""),
                len(paddle_json.get("rows", [])) if isinstance(paddle_json, dict) else 0,
            )
        except Exception as exc:  # noqa: BLE001
            errors["paddle"] = str(exc)
            LOGGER.exception("Paddle OCR failed | file=%s | error=%s", file_path, exc)
            paddle_json, paddle_text = {}, ""

        try:
            tesseract_text, tesseract_boxes = tess_future.result()
            LOGGER.info(
                "Tesseract OCR done | file=%s | text_len=%s | boxes=%s",
                file_path,
                len(tesseract_text or ""),
                len(tesseract_boxes) if tesseract_boxes else 0,
            )
            if not tesseract_text:
                errors.setdefault("tesseract", "no_text_extracted")
                LOGGER.warning("Tesseract OCR returned empty text | file=%s", file_path)
        except Exception as exc:  # noqa: BLE001
            errors["tesseract"] = str(exc)
            LOGGER.warning("Tesseract OCR failed | file=%s | error=%s", file_path, exc)
            tesseract_text, tesseract_boxes = "", None

    # Fallback safety rules: never break pipeline.
    if not paddle_json and not paddle_text and tesseract_text:
        fused_json = {
            "rows": [],
            "text": tesseract_text,
            "key_values": {},
            "normalized_fields": {},
        }
        fused_text = tesseract_text
    elif (paddle_json or paddle_text) and not tesseract_text:
        fused_json = dict(paddle_json) if isinstance(paddle_json, dict) else {"rows": []}
        fused_json["text"] = paddle_text
        fused_json.setdefault("key_values", {})
        fused_json.setdefault("normalized_fields", {})
        fused_text = paddle_text
    else:
        fused_json, fused_text = fuse_ocr_outputs(
            paddle_json=paddle_json,
            paddle_text=paddle_text,
            tesseract_text=tesseract_text,
        )

    debug_payload: Dict[str, Any] = {
        "paddle": {"json": paddle_json, "text": paddle_text},
        "tesseract": {
            "called": True,
            "text": tesseract_text,
            "boxes": tesseract_boxes,
            "text_length": len(tesseract_text or ""),
            "boxes_count": len(tesseract_boxes) if tesseract_boxes else 0,
        },
        "fused": {"json": fused_json, "text": fused_text},
        "errors": errors,
    }
    LOGGER.info(
        "Multi-OCR complete | file=%s | fused_text_len=%s | has_errors=%s | duration_sec=%.2f",
        file_path,
        len(fused_text or ""),
        bool(errors),
        time.perf_counter() - start_ts,
    )
    return fused_json, fused_text, debug_payload


def run_ocr(
    file_path: Path,
    languages: Optional[Sequence[str]] = None,
    pdf_dpi: int = 220,
    min_token_confidence: float = 0.20,
    tesseract_lang: str = "eng",
) -> Tuple[Dict[str, Any], str]:
    """
    Backward-compatible OCR entrypoint returning fused OCR result.
    """
    fused_json, fused_text, _ = run_multi_ocr(
        file_path=file_path,
        languages=languages,
        pdf_dpi=pdf_dpi,
        min_token_confidence=min_token_confidence,
        tesseract_lang=tesseract_lang,
    )
    return fused_json, fused_text
