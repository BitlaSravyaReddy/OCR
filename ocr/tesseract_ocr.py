from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import logging
import os
import shutil

import cv2
import numpy as np
import pypdfium2 as pdfium
import pytesseract
from PIL import Image

LOGGER = logging.getLogger(__name__)
_TESSERACT_AVAILABILITY: Optional[Tuple[bool, str]] = None


def _resolve_tesseract_cmd() -> str:
    """
    Resolve Tesseract executable path deterministically.
    Priority:
    1) TESSERACT_CMD env var
    2) shutil.which("tesseract")
    3) Common Windows install locations
    """
    env_cmd = os.getenv("TESSERACT_CMD", "").strip().strip('"')
    if env_cmd and Path(env_cmd).exists():
        return env_cmd

    found = shutil.which("tesseract")
    if found:
        return found

    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    return "tesseract"


def _get_tesseract_version_once(resolved_cmd: str) -> str:
    global _TESSERACT_AVAILABILITY
    if _TESSERACT_AVAILABILITY is not None:
        available, message = _TESSERACT_AVAILABILITY
        if available:
            return message
        raise RuntimeError(message)

    try:
        version = str(pytesseract.get_tesseract_version())
    except Exception as exc:  # noqa: BLE001
        message = (
            "Tesseract is not installed or not available on PATH "
            f"(resolved_cmd={resolved_cmd}). PaddleOCR will be used without Tesseract assist."
        )
        _TESSERACT_AVAILABILITY = (False, message)
        raise RuntimeError(message) from exc

    _TESSERACT_AVAILABILITY = (True, version)
    return version


def _render_pdf_pages(pdf_path: Path, dpi: int) -> List[np.ndarray]:
    scale = max(1.0, dpi / 72.0)
    doc = pdfium.PdfDocument(str(pdf_path))
    images: List[np.ndarray] = []
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            bitmap = page.render(scale=scale)
            arr = bitmap.to_numpy()
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            images.append(arr)
    finally:
        doc.close()
    return images


def _load_images(file_path: Path, dpi: int) -> List[np.ndarray]:
    if file_path.suffix.lower() == ".pdf":
        return _render_pdf_pages(file_path, dpi=dpi)

    image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to read file for Tesseract OCR: {file_path}")
    return [image]


def _to_pil(image_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _rotate_bgr(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported angle: {angle}")


def _ocr_single_page(image: np.ndarray, lang: str) -> Tuple[str, List[Dict[str, str]]]:
    pil_img = _to_pil(image)
    page_text = pytesseract.image_to_string(pil_img, lang=lang).strip()

    boxes: List[Dict[str, str]] = []
    data = pytesseract.image_to_data(pil_img, lang=lang, output_type=pytesseract.Output.DICT)
    n = len(data.get("text", []))
    for i in range(n):
        txt = str(data["text"][i]).strip()
        if not txt:
            continue
        boxes.append(
            {
                "text": txt,
                "left": str(data["left"][i]),
                "top": str(data["top"][i]),
                "width": str(data["width"][i]),
                "height": str(data["height"][i]),
                "conf": str(data["conf"][i]),
            }
        )
    return page_text, boxes


def _score_page(text: str, boxes: Sequence[Dict[str, str]]) -> float:
    # Deterministic quality score favoring readable longer text and more tokens.
    alnum = sum(1 for ch in text if ch.isalnum())
    return (len(text) * 0.01) + (alnum * 0.02) + (len(boxes) * 0.05)


def run_tesseract_ocr(
    file_path: Path,
    dpi: int = 220,
    lang: str = "eng",
) -> Tuple[str, Optional[List[Dict[str, str]]]]:
    """
    Run Tesseract OCR and return (plain_text, optional_boxes).
    Raises on failures so caller can log and fallback safely.
    """
    resolved_cmd = _resolve_tesseract_cmd()
    pytesseract.pytesseract.tesseract_cmd = resolved_cmd

    LOGGER.info(
        "Tesseract OCR start | file=%s | dpi=%s | lang=%s | tesseract_cmd=%s",
        file_path,
        dpi,
        lang,
        resolved_cmd,
    )

    # Explicit availability check to produce a clear failure reason in debug payload.
    version = _get_tesseract_version_once(resolved_cmd)
    LOGGER.info("Tesseract detected | version=%s", version)

    images = _load_images(file_path, dpi=dpi)
    LOGGER.info("Tesseract loaded %s page image(s) from %s", len(images), file_path)

    page_texts: List[str] = []
    boxes_out: List[Dict[str, str]] = []
    is_pdf = file_path.suffix.lower() == ".pdf"

    for page_idx, image in enumerate(images, start=1):
        best_text = ""
        best_boxes: List[Dict[str, str]] = []
        best_angle = 0
        best_score = -1.0

        # Fast path: run 0-degree first and only explore extra angles when weak.
        angle0_text, angle0_boxes = _ocr_single_page(image, lang=lang)
        angle0_score = _score_page(angle0_text, angle0_boxes)
        best_text = angle0_text
        best_boxes = angle0_boxes
        best_angle = 0
        best_score = angle0_score

        should_try_more = (not is_pdf) and (angle0_score < 40.0)
        angles = (90, 180, 270) if should_try_more else ()
        for angle in angles:
            rotated = _rotate_bgr(image, angle)
            page_text, page_boxes = _ocr_single_page(rotated, lang=lang)
            score = _score_page(page_text, page_boxes)
            if score > best_score:
                best_score = score
                best_text = page_text
                best_boxes = page_boxes
                best_angle = angle

        LOGGER.info(
            "Tesseract page selected | file=%s | page=%s | angle=%s | text_len=%s | boxes=%s",
            file_path,
            page_idx,
            best_angle,
            len(best_text),
            len(best_boxes),
        )

        page_texts.append(best_text)
        for box in best_boxes:
            box_with_page = dict(box)
            box_with_page["page"] = str(page_idx)
            boxes_out.append(box_with_page)

    full_text = "\n\n".join(part for part in page_texts if part)
    LOGGER.info(
        "Tesseract OCR done | file=%s | text_len=%s | boxes=%s",
        file_path,
        len(full_text),
        len(boxes_out),
    )
    return full_text, boxes_out or None
