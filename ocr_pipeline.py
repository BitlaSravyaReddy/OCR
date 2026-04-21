import argparse
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pypdfium2 as pdfium
import requests


# Reduce startup noise and avoid network model-host checks in production runs.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

from paddleocr import PaddleOCR

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency fallback.
    load_dotenv = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PDF_EXTENSIONS = {".pdf"}


FIELD_ALIASES: Dict[str, List[str]] = {
    "invoice_no": [
        "invoice no",
        "invoice number",
        "invoice #",
        "inv no",
        "inv #",
        "bill no",
        "bill number",
    ],
    "date": ["date", "invoice date", "bill date", "dt"],
    "po_no": ["po no", "po number", "purchase order", "order no", "order number"],
    "gstin": ["gstin", "gst no", "gstin/uin", "gst no."],
    "hsn": ["hsn", "hsn/sac", "sac", "hsn code"],
    "quantity": ["qty", "quantity", "qnty"],
    "rate": ["rate", "price", "unit rate"],
    "taxable_amount": ["taxable amount", "taxable value", "amount"],
    "cgst": ["cgst", "cgst amount"],
    "sgst": ["sgst", "sgst amount"],
    "igst": ["igst", "igst amount"],
    "total_amount": ["total", "grand total", "net total", "amount chargeable"],
    "net_amount": ["net amount", "bill amount", "invoice value"],
    "buyer_name": ["buyer", "bill to", "customer", "party"],
    "seller_name": ["seller", "supplier", "from"],
}


DATE_NUMERIC_PATTERN = re.compile(r"(?<!\d)(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})(?!\d)")
DATE_TEXT_PATTERN = re.compile(
    r"(?<!\w)(\d{1,2}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,/-]*\d{2,4})(?!\w)",
    re.IGNORECASE,
)

GSTIN_PATTERN = re.compile(r"\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]")

AMOUNT_PATTERN = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)(?!\d)")


@dataclass
class OCRWord:
    text: str
    confidence: float
    x: float
    y: float
    left: float
    right: float


@dataclass
class OCRLine:
    text: str
    confidence: float
    y: float
    left: float
    right: float
    words: List[OCRWord] = field(default_factory=list)


@dataclass
class OCRCandidate:
    text: str
    line_count: int
    char_count: int
    mean_confidence: float
    lexical_quality: float
    source: str
    lines: List[OCRLine] = field(default_factory=list)

    @property
    def ranking_score(self) -> float:
        confidence_weight = 0.68
        quality_weight = 0.22
        length_weight = 0.10
        length_factor = min(1.0, self.char_count / 1200.0)
        return (
            confidence_weight * self.mean_confidence
            + quality_weight * self.lexical_quality
            + length_weight * length_factor
        )


@dataclass
class OCRPageResult:
    page_number: int
    text: str
    source: str
    mean_confidence: float
    lexical_quality: float
    lines: List[OCRLine] = field(default_factory=list)


@dataclass
class FileSummary:
    input_file: str
    output_txt: str
    output_json: str
    status: str
    pages: int
    extraction_sources: List[str]
    detected_orientation: str = "unknown"
    error: str = ""


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def safe_file_slug(path: Path) -> str:
    stem = f"{path.stem}_{path.suffix.lstrip('.')}"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return slug or "document"


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    cleaned = [line for line in lines if line]
    return "\n".join(cleaned)


def normalize_key_tokens(text: str) -> List[str]:
    sanitized = re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()
    if not sanitized:
        return []
    return [part for part in sanitized.split() if part]


def text_lexical_quality(text: str) -> float:
    if not text:
        return 0.0
    visible_chars = [c for c in text if not c.isspace()]
    if not visible_chars:
        return 0.0

    good_chars = sum(
        1
        for c in visible_chars
        if c.isalnum() or c in ",.;:/-_%$()[]{}+#@&*'\""
    )
    replacement_chars = sum(1 for c in visible_chars if c == "\ufffd")

    base_ratio = good_chars / len(visible_chars)
    replacement_penalty = replacement_chars / len(visible_chars)
    return max(0.0, min(1.0, base_ratio - replacement_penalty))


def fix_ocr_numeric_noise(value: str) -> str:
    if not value:
        return value

    def replace_if_numeric_like(token: str) -> str:
        if re.fullmatch(r"[0-9OolI|SBGZ,/.:\\-]+", token):
            fixed = token
            fixed = fixed.replace("O", "0").replace("o", "0")
            fixed = fixed.replace("I", "1").replace("l", "1").replace("|", "1")
            fixed = fixed.replace("S", "5")
            fixed = fixed.replace("B", "8")
            fixed = fixed.replace("G", "6")
            fixed = fixed.replace("Z", "2")
            return fixed
        return token

    return " ".join(replace_if_numeric_like(part) for part in value.split())


def normalize_date_value(value: str) -> str:
    cleaned = fix_ocr_numeric_noise(value)
    match_numeric = DATE_NUMERIC_PATTERN.search(cleaned)
    match_text = DATE_TEXT_PATTERN.search(cleaned)

    if not match_numeric and not match_text:
        return ""

    group = ""
    if match_numeric:
        group = match_numeric.group(1)
    elif match_text:
        group = match_text.group(1)

    group = group.strip().replace(".", "-").replace("/", "-")
    group = re.sub(r"\s+", "-", group)
    group = re.sub(r"-+", "-", group)
    return group


def normalize_amount_value(value: str) -> str:
    cleaned = fix_ocr_numeric_noise(value)
    matches = AMOUNT_PATTERN.findall(cleaned)
    if not matches:
        return ""

    def to_float(raw: str) -> float:
        try:
            return float(raw.replace(",", ""))
        except ValueError:
            return -1.0

    best = max(matches, key=to_float)
    best = best.replace(",", "")
    return best


def normalize_field_value(field: str, value: str) -> str:
    value = normalize_text(value)
    value = value.strip("-:| ")
    value = fix_ocr_numeric_noise(value)

    if not value:
        return ""

    if field in {"date"}:
        normalized = normalize_date_value(value)
        return normalized

    if field in {"gstin"}:
        compact = re.sub(r"[^A-Z0-9]", "", value.upper())
        match = GSTIN_PATTERN.search(compact)
        return match.group(0) if match else ""

    if field in {"invoice_no", "po_no"}:
        candidates = re.findall(r"[A-Za-z0-9/_-]{3,30}", value)
        for candidate in candidates:
            if any(char.isdigit() for char in candidate):
                return candidate
        return ""

    if field in {"hsn"}:
        match = re.search(r"\b\d{4,10}\b", value)
        return match.group(0) if match else ""

    if field in {"total_amount", "net_amount", "taxable_amount", "cgst", "sgst", "igst", "rate"}:
        normalized = normalize_amount_value(value)
        return normalized

    if field in {"quantity"}:
        normalized = normalize_amount_value(value)
        return normalized

    if field in {"buyer_name", "seller_name"}:
        cleaned = re.sub(
            r"^(?:buyer|seller|supplier|bill\s+to|customer|party|from)\b[:\-\s]*",
            "",
            value,
            flags=re.IGNORECASE,
        ).strip()
        if len(cleaned) < 2 or len(cleaned) > 120:
            return ""
        return cleaned

    return value


def deduplicate_records(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    deduped: List[Dict[str, str]] = []
    for row in records:
        serialized = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if serialized in seen:
            continue
        seen.add(serialized)
        deduped.append(row)
    return deduped


def _extract_json_from_text(raw_text: str) -> Optional[Any]:
    text = (raw_text or "").strip()
    if not text:
        return None

    candidate_blocks = [text]

    # Handle markdown fenced output.
    if text.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        stripped = re.sub(r"\s*```$", "", stripped)
        candidate_blocks.append(stripped.strip())

    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        candidate_blocks.append(text[first_obj : last_obj + 1])

    first_arr = text.find("[")
    last_arr = text.rfind("]")
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        candidate_blocks.append(text[first_arr : last_arr + 1])

    for block in candidate_blocks:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue

    return None


class GeminiLayoutAnalyzer:
    def __init__(self, model: str, timeout_seconds: int, enabled: bool) -> None:
        self.model = model
        self.timeout_seconds = max(10, timeout_seconds)
        self.enabled = enabled
        self.api_key = ""
        self._load_api_key()

    def _load_api_key(self) -> None:
        if load_dotenv is not None:
            try:
                load_dotenv(dotenv_path=Path(".env"), override=False)
            except Exception:
                pass

        # Fallback parser for environments where python-dotenv is unavailable.
        if not os.getenv("GEMINI_API_KEY"):
            env_path = Path(".env")
            if env_path.exists():
                try:
                    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                        line = raw_line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        if key.strip() == "GEMINI_API_KEY":
                            os.environ.setdefault("GEMINI_API_KEY", value.strip().strip('"\''))
                            break
                except OSError:
                    pass

        self.api_key = (os.getenv("GEMINI_API_KEY") or "").strip()

    def is_available(self) -> bool:
        return self.enabled and bool(self.api_key)

    def _sanitize_error_message(self, message: str) -> str:
        sanitized = message
        if self.api_key:
            sanitized = sanitized.replace(self.api_key, "***")
        sanitized = re.sub(r"(key=)[^&\s]+", r"\1***", sanitized)
        return sanitized

    def _build_prompt_payload(self, structured_payload: Dict[str, Any]) -> Dict[str, Any]:
        pages = structured_payload.get("pages", [])
        compact_pages: List[Dict[str, Any]] = []
        for page in pages:
            compact_pages.append(
                {
                    "page_number": page.get("page_number"),
                    "orientation": page.get("orientation"),
                    "layout": page.get("layout", [])[:400],
                    "rows": page.get("rows", [])[:120],
                    "key_values": page.get("key_values", {}),
                    "records": page.get("records", [])[:120],
                }
            )

        return {
            "input_file": structured_payload.get("input_file"),
            "document_orientation": structured_payload.get("document_orientation"),
            "normalized_fields": structured_payload.get("normalized_fields", {}),
            "records": structured_payload.get("records", [])[:160],
            "layout": structured_payload.get("layout", [])[:1200],
            "rows": structured_payload.get("rows", [])[:300],
            "pages": compact_pages,
        }

    def refine(self, structured_payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "status": "disabled"}
        if not self.api_key:
            return {"enabled": False, "status": "missing_api_key"}

        compact_payload = self._build_prompt_payload(structured_payload)
        prompt = (
            "You are a deterministic invoice/document parser. "
            "Input contains OCR layout tokens with coordinates and Y-grouped rows. "
            "Return ONLY strict JSON with this schema:\n"
            "{\n"
            "  \"normalized_fields\": {\"invoice_no\": \"...\", \"date\": \"...\", ...},\n"
            "  \"records\": [{...}],\n"
            "  \"row_analysis\": [{\"page_number\": 1, \"row_number\": 1, \"meaning\": \"header|item|summary|other\", \"text\": \"...\"}],\n"
            "  \"quality_notes\": [\"...\"]\n"
            "}\n"
            "Rules: do not hallucinate values, keep only values backed by input evidence, preserve numeric/date precision, "
            "and keep records clean for downstream processing.\n\n"
            f"Input JSON:\n{json.dumps(compact_payload, ensure_ascii=False)}"
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        request_body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        try:
            response = requests.post(url, json=request_body, timeout=self.timeout_seconds)
            response.raise_for_status()
            response_data = response.json()
        except Exception as exc:
            return {
                "enabled": True,
                "status": "request_failed",
                "model": self.model,
                "error": self._sanitize_error_message(str(exc)),
            }

        texts: List[str] = []
        for candidate in response_data.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                text = part.get("text") if isinstance(part, dict) else None
                if text:
                    texts.append(text)

        if not texts:
            return {
                "enabled": True,
                "status": "empty_response",
                "model": self.model,
            }

        raw_output = "\n".join(texts).strip()
        parsed = _extract_json_from_text(raw_output)
        if parsed is None:
            return {
                "enabled": True,
                "status": "invalid_json",
                "model": self.model,
                "raw_output": raw_output[:4000],
            }

        if isinstance(parsed, list):
            parsed = {"records": parsed}

        if not isinstance(parsed, dict):
            return {
                "enabled": True,
                "status": "invalid_json_type",
                "model": self.model,
            }

        return {
            "enabled": True,
            "status": "success",
            "model": self.model,
            "analysis": parsed,
        }


class HybridOCR:
    def __init__(
        self,
        languages: Sequence[str],
        render_dpi: int,
        min_token_confidence: float,
    ) -> None:
        self.languages = [lang.strip() for lang in languages if lang.strip()]
        self.render_dpi = max(120, render_dpi)
        self.min_token_confidence = max(0.0, min(1.0, min_token_confidence))
        self._ocr_engines: Dict[str, PaddleOCR] = {}

    def _get_engine(self, lang: str) -> PaddleOCR:
        if lang not in self._ocr_engines:
            logging.info("Initializing PaddleOCR model for language=%s", lang)
            self._ocr_engines[lang] = PaddleOCR(
                lang=lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        return self._ocr_engines[lang]

    def _safe_predict(self, lang: str, image: np.ndarray, source_label: str) -> List[dict]:
        last_error = ""
        for attempt in range(2):
            try:
                engine = self._get_engine(lang)
                result = engine.predict(input=image)
                if isinstance(result, list):
                    return result
                return []
            except Exception as exc:  # noqa: BLE001 - OCR engines may fail unpredictably.
                last_error = str(exc)
                logging.warning(
                    "OCR predict failed for %s lang=%s attempt=%d error=%s",
                    source_label,
                    lang,
                    attempt + 1,
                    exc,
                )
                # Force reinitialization for next attempt.
                self._ocr_engines.pop(lang, None)

        logging.error("OCR predict exhausted retries for %s lang=%s error=%s", source_label, lang, last_error)
        return []

    def _image_variants(self, image_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        variants: List[Tuple[str, np.ndarray]] = [("original", image_bgr)]

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10,
        )
        variants.append(("adaptive", cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)))
        return variants

    def _tokens_to_lines(
        self,
        texts: Sequence[str],
        scores: Sequence[float],
        polys: Sequence[Sequence[Sequence[float]]],
    ) -> List[OCRLine]:
        tokens = []
        for idx, raw_text in enumerate(texts):
            text = normalize_text(str(raw_text))
            if not text:
                continue

            score = float(scores[idx]) if idx < len(scores) else 0.0
            if score < self.min_token_confidence and len(text) <= 2:
                continue

            if idx < len(polys):
                points = np.array(polys[idx], dtype=np.float32)
                left = float(np.min(points[:, 0]))
                right = float(np.max(points[:, 0]))
                center_x = float(np.mean(points[:, 0]))
                center_y = float(np.mean(points[:, 1]))
                height = float(np.max(points[:, 1]) - np.min(points[:, 1]))
            else:
                center_x = float(idx) * 20.0
                center_y = float(idx) * 20.0
                left = center_x
                right = center_x + max(20.0, len(text) * 8.0)
                height = 20.0

            tokens.append(
                {
                    "text": text,
                    "score": score,
                    "x": center_x,
                    "y": center_y,
                    "left": left,
                    "right": right,
                    "height": max(1.0, height),
                }
            )

        if not tokens:
            return []

        tokens.sort(key=lambda t: (t["y"], t["x"]))
        rows: List[Dict[str, Any]] = []

        for token in tokens:
            attached = False
            for row in rows:
                row_y = float(row["y"])
                row_h = float(row["height"])
                tolerance = max(10.0, min(42.0, (row_h + token["height"]) * 0.40))
                if abs(token["y"] - row_y) <= tolerance:
                    row_tokens = row["tokens"]
                    assert isinstance(row_tokens, list)
                    row_tokens.append(token)
                    token_count = len(row_tokens)
                    row["y"] = ((row_y * (token_count - 1)) + token["y"]) / token_count
                    row["height"] = max(row_h, token["height"])
                    attached = True
                    break

            if not attached:
                rows.append({"y": token["y"], "height": token["height"], "tokens": [token]})

        rows.sort(key=lambda r: float(r["y"]))
        merged_lines: List[OCRLine] = []

        for row in rows:
            row_tokens = row["tokens"]
            assert isinstance(row_tokens, list)
            row_tokens.sort(key=lambda t: t["x"])

            line_text = " ".join(str(token["text"]) for token in row_tokens).strip()
            if not line_text:
                continue

            line_confidence = float(np.mean([float(token["score"]) for token in row_tokens]))
            words = [
                OCRWord(
                    text=str(token["text"]),
                    confidence=float(token["score"]),
                    x=float(token["x"]),
                    y=float(token["y"]),
                    left=float(token["left"]),
                    right=float(token["right"]),
                )
                for token in row_tokens
            ]
            line_left = min(word.left for word in words)
            line_right = max(word.right for word in words)

            merged_lines.append(
                OCRLine(
                    text=line_text,
                    confidence=line_confidence,
                    y=float(row["y"]),
                    left=line_left,
                    right=line_right,
                    words=words,
                )
            )

        return merged_lines

    def _candidate_from_result(self, result: dict, source: str) -> OCRCandidate:
        texts = result.get("rec_texts", [])
        scores = result.get("rec_scores", [])
        polys = result.get("rec_polys", [])

        lines = self._tokens_to_lines(texts, scores, polys)
        if not lines:
            return OCRCandidate(
                text="",
                line_count=0,
                char_count=0,
                mean_confidence=0.0,
                lexical_quality=0.0,
                source=source,
                lines=[],
            )

        merged_text = "\n".join(line.text for line in lines)
        merged_text = normalize_text(merged_text)
        mean_confidence = float(np.mean([line.confidence for line in lines]))

        return OCRCandidate(
            text=merged_text,
            line_count=len(lines),
            char_count=len(merged_text),
            mean_confidence=mean_confidence,
            lexical_quality=text_lexical_quality(merged_text),
            source=source,
            lines=lines,
        )

    def _candidate_from_plain_text(self, text: str, source: str) -> OCRCandidate:
        cleaned = normalize_text(text)
        if not cleaned:
            return OCRCandidate(
                text="",
                line_count=0,
                char_count=0,
                mean_confidence=0.0,
                lexical_quality=0.0,
                source=source,
                lines=[],
            )

        plain_lines = [
            OCRLine(
                text=line,
                confidence=1.0,
                y=float(idx),
                left=0.0,
                right=float(max(1, len(line))),
                words=[],
            )
            for idx, line in enumerate(cleaned.split("\n"), start=1)
            if line.strip()
        ]

        return OCRCandidate(
            text=cleaned,
            line_count=len(plain_lines),
            char_count=len(cleaned),
            mean_confidence=1.0,
            lexical_quality=text_lexical_quality(cleaned),
            source=source,
            lines=plain_lines,
        )

    def _choose_better(self, current: OCRCandidate, candidate: OCRCandidate) -> OCRCandidate:
        if candidate.char_count == 0:
            return current
        if current.char_count == 0:
            return candidate

        margin = candidate.ranking_score - current.ranking_score
        if margin > 0.01:
            return candidate
        if abs(margin) <= 0.01 and candidate.char_count > current.char_count:
            return candidate
        if abs(margin) <= 0.005 and candidate.mean_confidence > current.mean_confidence:
            return candidate
        return current

    def ocr_image(self, image_bgr: np.ndarray, source_label: str) -> OCRCandidate:
        best = OCRCandidate(
            text="",
            line_count=0,
            char_count=0,
            mean_confidence=0.0,
            lexical_quality=0.0,
            source=f"{source_label}:none",
            lines=[],
        )

        variants = self._image_variants(image_bgr)
        if not self.languages:
            return best

        def is_strong(candidate: OCRCandidate) -> bool:
            return (
                candidate.char_count >= 40
                and candidate.mean_confidence >= 0.88
                and candidate.lexical_quality >= 0.72
            )

        # Stage 1: primary language on original image.
        primary = self.languages[0]
        primary_result = self._safe_predict(primary, variants[0][1], source_label)
        if primary_result:
            candidate = self._candidate_from_result(primary_result[0], source=f"{source_label}:{primary}:{variants[0][0]}")
            best = self._choose_better(best, candidate)
        if is_strong(best):
            return best

        # Stage 2: primary language on additional variants.
        for variant_name, variant in variants[1:]:
            result = self._safe_predict(primary, variant, source_label)
            if not result:
                continue
            candidate = self._candidate_from_result(result[0], source=f"{source_label}:{primary}:{variant_name}")
            best = self._choose_better(best, candidate)
            if is_strong(best):
                return best

        # Stage 3: fallback languages only if primary result stays weak.
        for lang in self.languages[1:]:
            result = self._safe_predict(lang, variants[0][1], source_label)
            if result:
                candidate = self._candidate_from_result(result[0], source=f"{source_label}:{lang}:{variants[0][0]}")
                best = self._choose_better(best, candidate)
                if is_strong(best):
                    return best

            for variant_name, variant in variants[1:]:
                result = self._safe_predict(lang, variant, source_label)
                if not result:
                    continue
                candidate = self._candidate_from_result(result[0], source=f"{source_label}:{lang}:{variant_name}")
                best = self._choose_better(best, candidate)
                if is_strong(best):
                    return best

        return best

    def _pdf_text_layer_candidate(self, page: pdfium.PdfPage, source_label: str) -> OCRCandidate:
        text_page = page.get_textpage()
        raw_text = text_page.get_text_bounded()
        return self._candidate_from_plain_text(raw_text, source=f"{source_label}:pdf_text_layer")

    def _render_pdf_page(self, page: pdfium.PdfPage) -> np.ndarray:
        scale = max(1.0, self.render_dpi / 72.0)
        bitmap = page.render(scale=scale)
        image = bitmap.to_numpy()

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        return image

    def _pick_pdf_candidate(self, layer: OCRCandidate, ocr: OCRCandidate) -> OCRCandidate:
        if layer.char_count == 0:
            return ocr
        if ocr.char_count == 0:
            return layer

        layer_length = min(1.0, layer.char_count / 1500.0)
        layer_score = (0.80 * layer.lexical_quality) + (0.20 * layer_length)

        ocr_length = min(1.0, ocr.char_count / 1500.0)
        ocr_score = (0.65 * ocr.mean_confidence) + (0.25 * ocr.lexical_quality) + (0.10 * ocr_length)

        if layer_score >= (ocr_score + 0.08) and layer.lexical_quality >= 0.60:
            return layer
        return ocr

    def process_image_pages(self, image_path: Path) -> List[OCRPageResult]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unable to read image file: {image_path}")

        try:
            candidate = self.ocr_image(image, source_label=image_path.name)
        except Exception as exc:  # noqa: BLE001 - fallback allows downstream JSON generation.
            logging.error("OCR failed for image %s, using empty fallback: %s", image_path, exc)
            candidate = OCRCandidate(
                text="",
                line_count=0,
                char_count=0,
                mean_confidence=0.0,
                lexical_quality=0.0,
                source=f"{image_path.name}:ocr_failed",
                lines=[],
            )
        return [
            OCRPageResult(
                page_number=1,
                text=candidate.text,
                source=candidate.source,
                mean_confidence=candidate.mean_confidence,
                lexical_quality=candidate.lexical_quality,
                lines=candidate.lines,
            )
        ]

    def process_pdf_pages(self, pdf_path: Path) -> List[OCRPageResult]:
        doc = pdfium.PdfDocument(str(pdf_path))
        pages: List[OCRPageResult] = []

        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                page_label = f"{pdf_path.name}:p{page_index + 1}"

                layer_candidate = self._pdf_text_layer_candidate(page, source_label=page_label)
                rendered = self._render_pdf_page(page)
                try:
                    ocr_candidate = self.ocr_image(rendered, source_label=page_label)
                except Exception as exc:  # noqa: BLE001 - fallback to PDF text layer.
                    logging.error("OCR failed for %s, using text-layer fallback: %s", page_label, exc)
                    ocr_candidate = OCRCandidate(
                        text="",
                        line_count=0,
                        char_count=0,
                        mean_confidence=0.0,
                        lexical_quality=0.0,
                        source=f"{page_label}:ocr_failed",
                        lines=[],
                    )
                chosen = self._pick_pdf_candidate(layer_candidate, ocr_candidate)

                pages.append(
                    OCRPageResult(
                        page_number=page_index + 1,
                        text=chosen.text,
                        source=chosen.source,
                        mean_confidence=chosen.mean_confidence,
                        lexical_quality=chosen.lexical_quality,
                        lines=chosen.lines,
                    )
                )
        finally:
            doc.close()

        return pages

    # Backward-compatible wrappers.
    def process_image_file(self, image_path: Path) -> Tuple[str, List[str]]:
        pages = self.process_image_pages(image_path)
        return pages[0].text, [pages[0].source]

    def process_pdf_file(self, pdf_path: Path) -> Tuple[str, List[str]]:
        pages = self.process_pdf_pages(pdf_path)
        full_text = "\n\n".join(page.text for page in pages if page.text)
        sources = [page.source for page in pages]
        return full_text, sources


class StructuredExtractor:
    def __init__(self) -> None:
        self.alias_entries: List[Tuple[str, List[str]]] = self._build_alias_entries()
        self.value_patterns: Dict[str, re.Pattern[str]] = self._build_value_patterns()

    def _build_alias_entries(self) -> List[Tuple[str, List[str]]]:
        entries: List[Tuple[str, List[str]]] = []
        for field, aliases in FIELD_ALIASES.items():
            for alias in aliases:
                tokens = normalize_key_tokens(alias)
                if tokens:
                    entries.append((field, tokens))
        entries.sort(key=lambda item: len(item[1]), reverse=True)
        return entries

    def _build_value_patterns(self) -> Dict[str, re.Pattern[str]]:
        patterns: Dict[str, re.Pattern[str]] = {}
        for field, aliases in FIELD_ALIASES.items():
            alias_patterns = []
            for alias in aliases:
                tokens = normalize_key_tokens(alias)
                if not tokens:
                    continue
                alias_patterns.append(r"\\s+".join(re.escape(token) for token in tokens))

            if not alias_patterns:
                continue

            combined = "|".join(sorted(set(alias_patterns), key=len, reverse=True))
            patterns[field] = re.compile(
                rf"^\s*(?:{combined})\s*(?:[:=\-]\s*|\s+)(.+?)\s*$",
                re.IGNORECASE,
            )

        return patterns

    def _contains_sequence(self, tokens: List[str], alias_tokens: List[str]) -> bool:
        if len(alias_tokens) > len(tokens):
            return False

        for index in range(0, len(tokens) - len(alias_tokens) + 1):
            if tokens[index : index + len(alias_tokens)] == alias_tokens:
                return True
        return False

    def _detect_field_from_tokens(self, tokens: List[str]) -> str:
        if not tokens:
            return ""
        for field, alias_tokens in self.alias_entries:
            if self._contains_sequence(tokens, alias_tokens):
                return field
        return ""

    def _line_words(self, line: OCRLine) -> List[OCRWord]:
        if line.words:
            return line.words

        parts = line.text.split()
        if not parts:
            return []

        generated_words: List[OCRWord] = []
        for idx, part in enumerate(parts):
            center_x = float((idx + 1) * 100)
            generated_words.append(
                OCRWord(
                    text=part,
                    confidence=line.confidence,
                    x=center_x,
                    y=line.y,
                    left=center_x - 40,
                    right=center_x + 40,
                )
            )
        return generated_words

    def _build_layout_tokens(self, lines: List[OCRLine]) -> List[Dict[str, Any]]:
        tokens: List[Dict[str, Any]] = []
        for line in lines:
            words = self._line_words(line)
            for word in words:
                text = normalize_text(word.text).strip()
                if not text:
                    continue
                tokens.append(
                    {
                        "text": text,
                        "x": int(round(word.x)),
                        "y": int(round(word.y)),
                        "left": float(word.left),
                        "right": float(word.right),
                    }
                )

        tokens.sort(key=lambda item: (int(item["y"]), int(item["x"])))
        return tokens

    def _build_row_cells(self, row_tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not row_tokens:
            return []

        ordered = sorted(row_tokens, key=lambda item: float(item["x"]))
        widths = [max(8.0, float(token["right"]) - float(token["left"])) for token in ordered]
        gap_threshold = max(18.0, min(90.0, float(np.median(widths)) * 1.7))

        grouped: List[List[Dict[str, Any]]] = []
        current_group: List[Dict[str, Any]] = [ordered[0]]
        for token in ordered[1:]:
            previous = current_group[-1]
            gap = float(token["left"]) - float(previous["right"])
            if gap > gap_threshold:
                grouped.append(current_group)
                current_group = [token]
            else:
                current_group.append(token)
        grouped.append(current_group)

        cells: List[Dict[str, Any]] = []
        for group in grouped:
            cell_text = " ".join(str(token["text"]) for token in group).strip()
            if not cell_text:
                continue

            cells.append(
                {
                    "text": cell_text,
                    "x": int(round(float(np.mean([float(token["x"]) for token in group])))),
                    "y": int(round(float(np.mean([float(token["y"]) for token in group])))),
                }
            )

        return cells

    def _group_rows_by_y(self, layout_tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not layout_tokens:
            return []

        tokens = sorted(layout_tokens, key=lambda item: (int(item["y"]), int(item["x"])))
        y_values = sorted(int(token["y"]) for token in tokens)
        y_diffs = [
            float(y_values[idx + 1] - y_values[idx])
            for idx in range(len(y_values) - 1)
            if y_values[idx + 1] > y_values[idx]
        ]
        if y_diffs:
            y_tolerance = max(8.0, min(28.0, float(np.median(y_diffs)) * 0.55))
        else:
            y_tolerance = 12.0

        grouped_rows: List[Dict[str, Any]] = []
        for token in tokens:
            assigned = False
            for row in grouped_rows:
                if abs(float(token["y"]) - float(row["y"])) <= y_tolerance:
                    row_tokens = row["tokens"]
                    assert isinstance(row_tokens, list)
                    row_tokens.append(token)
                    count = len(row_tokens)
                    row["y"] = ((float(row["y"]) * (count - 1)) + float(token["y"])) / count
                    assigned = True
                    break

            if not assigned:
                grouped_rows.append({"y": float(token["y"]), "tokens": [token]})

        grouped_rows.sort(key=lambda row: float(row["y"]))
        row_payloads: List[Dict[str, Any]] = []
        for row_index, row in enumerate(grouped_rows, start=1):
            row_tokens = row["tokens"]
            assert isinstance(row_tokens, list)
            row_tokens.sort(key=lambda item: float(item["x"]))

            cells = self._build_row_cells(row_tokens)
            row_text = " | ".join(cell["text"] for cell in cells if cell.get("text"))
            if not row_text:
                continue

            row_payloads.append(
                {
                    "row_number": row_index,
                    "y": int(round(float(row["y"]))),
                    "cells": cells,
                    "row_text": row_text,
                }
            )

        return row_payloads

    def _looks_like_header_only_value(self, value: str, field: str) -> bool:
        tokens = normalize_key_tokens(value)
        if not tokens:
            return True

        detected = self._detect_field_from_tokens(tokens)
        if detected and detected != field and len(tokens) <= 4:
            return True

        if len(tokens) <= 3 and not re.search(r"\d", value):
            return True

        return False

    def _extract_vertical_pairs(self, lines: List[OCRLine]) -> Tuple[Dict[str, str], int]:
        pairs: Dict[str, str] = {}
        scores: Dict[str, float] = {}
        hits = 0

        for line in lines:
            text = line.text.strip()
            if not text:
                continue

            matched = False
            for field, pattern in self.value_patterns.items():
                result = pattern.match(text)
                if not result:
                    continue

                raw_value = result.group(1)
                value = normalize_field_value(field, raw_value)
                if not value or self._looks_like_header_only_value(value, field):
                    continue

                candidate_score = line.confidence + min(0.5, len(value) / 50.0)
                if candidate_score > scores.get(field, -1.0):
                    pairs[field] = value
                    scores[field] = candidate_score

                hits += 1
                matched = True
                break

            if matched:
                continue

            # Generic fallback for lines like KEY: VALUE where KEY maps to canonical aliases.
            generic_match = re.match(r"^\s*([A-Za-z][A-Za-z0-9 ./#&()_-]{1,60})\s*[:=]\s*(.+?)\s*$", text)
            if not generic_match:
                continue

            key_text = generic_match.group(1)
            value_text = generic_match.group(2)
            field = self._detect_field_from_tokens(normalize_key_tokens(key_text))
            if not field:
                continue

            value = normalize_field_value(field, value_text)
            if not value or self._looks_like_header_only_value(value, field):
                continue

            candidate_score = line.confidence + min(0.5, len(value) / 50.0)
            if candidate_score > scores.get(field, -1.0):
                pairs[field] = value
                scores[field] = candidate_score
            hits += 1

        return pairs, hits

    def _extract_header_columns(self, line: OCRLine) -> List[Dict[str, Any]]:
        words = self._line_words(line)
        columns: List[Dict[str, Any]] = []
        used_fields = set()

        index = 0
        while index < len(words):
            matched = False
            for span in (3, 2, 1):
                if index + span > len(words):
                    continue

                phrase = " ".join(words[position].text for position in range(index, index + span))
                field = self._detect_field_from_tokens(normalize_key_tokens(phrase))
                if not field or field in used_fields:
                    continue

                center_x = float(np.mean([words[position].x for position in range(index, index + span)]))
                columns.append({"field": field, "x": center_x})
                used_fields.add(field)
                index += span
                matched = True
                break

            if not matched:
                index += 1

        columns.sort(key=lambda item: float(item["x"]))
        return columns

    def _assign_row_to_columns(self, row_line: OCRLine, columns: List[Dict[str, Any]]) -> Dict[str, str]:
        words = self._line_words(row_line)
        if not words or not columns:
            return {}

        centers = [float(column["x"]) for column in columns]
        boundaries = [float("-inf")]
        for idx in range(len(centers) - 1):
            boundaries.append((centers[idx] + centers[idx + 1]) / 2.0)
        boundaries.append(float("inf"))

        buckets: List[List[str]] = [[] for _ in columns]
        for word in words:
            bucket_index = 0
            for idx in range(len(columns)):
                if boundaries[idx] <= word.x < boundaries[idx + 1]:
                    bucket_index = idx
                    break
            buckets[bucket_index].append(word.text)

        row_values: Dict[str, str] = {}
        for idx, column in enumerate(columns):
            field = str(column["field"])
            raw_value = " ".join(buckets[idx]).strip()
            value = normalize_field_value(field, raw_value)
            if value:
                row_values[field] = value

        # Filter rows that are effectively another header line.
        header_like_values = 0
        for field, value in row_values.items():
            detected = self._detect_field_from_tokens(normalize_key_tokens(value))
            if detected and detected != field:
                header_like_values += 1

        if header_like_values >= max(1, len(row_values) - 1):
            return {}

        return row_values

    def _extract_horizontal_records(self, lines: List[OCRLine]) -> Tuple[List[Dict[str, str]], int]:
        records: List[Dict[str, str]] = []
        hits = 0

        index = 0
        while index < len(lines) - 1:
            header_columns = self._extract_header_columns(lines[index])
            if len(header_columns) < 2:
                index += 1
                continue

            scan = index + 1
            extracted_rows = 0
            while scan < len(lines):
                if len(self._extract_header_columns(lines[scan])) >= 2:
                    break

                row_values = self._assign_row_to_columns(lines[scan], header_columns)
                row_values = {key: value for key, value in row_values.items() if value}
                if row_values:
                    records.append(row_values)
                    hits += 1
                    extracted_rows += 1

                if extracted_rows >= 8:
                    break
                scan += 1

            if extracted_rows == 0:
                index += 1
            else:
                index = scan

        return deduplicate_records(records), hits

    def _detect_orientation(self, vertical_hits: int, horizontal_hits: int) -> str:
        if vertical_hits > 0 and horizontal_hits > 0:
            return "mixed"
        if horizontal_hits > 0:
            return "horizontal"
        if vertical_hits > 0:
            return "vertical"
        return "unknown"

    def extract_page_structure(self, page: OCRPageResult) -> Dict[str, Any]:
        lines = [line for line in page.lines if line.text.strip()]
        if not lines and page.text:
            lines = [
                OCRLine(
                    text=text_line,
                    confidence=page.mean_confidence,
                    y=float(idx),
                    left=0.0,
                    right=float(max(1, len(text_line))),
                    words=[],
                )
                for idx, text_line in enumerate(page.text.split("\n"), start=1)
                if text_line.strip()
            ]

        vertical_pairs, vertical_hits = self._extract_vertical_pairs(lines)
        horizontal_records, horizontal_hits = self._extract_horizontal_records(lines)
        layout_tokens = self._build_layout_tokens(lines)
        grouped_rows = self._group_rows_by_y(layout_tokens)

        public_layout_tokens = [
            {
                "text": str(token["text"]),
                "x": int(token["x"]),
                "y": int(token["y"]),
            }
            for token in layout_tokens
        ]

        key_values = dict(vertical_pairs)
        for record in horizontal_records:
            for field, value in record.items():
                if field not in key_values or len(value) > len(key_values[field]):
                    key_values[field] = value

        orientation = self._detect_orientation(vertical_hits, horizontal_hits)

        return {
            "page_number": page.page_number,
            "orientation": orientation,
            "extraction_source": page.source,
            "mean_confidence": round(page.mean_confidence, 4),
            "lexical_quality": round(page.lexical_quality, 4),
            "vertical_pair_count": vertical_hits,
            "horizontal_row_count": horizontal_hits,
            "key_values": key_values,
            "records": horizontal_records,
            "layout": public_layout_tokens,
            "rows": grouped_rows,
        }

    def extract_document_structure(self, input_path: Path, pages: List[OCRPageResult]) -> Dict[str, Any]:
        page_payloads: List[Dict[str, Any]] = []
        orientation_counter: Dict[str, int] = {"vertical": 0, "horizontal": 0, "mixed": 0, "unknown": 0}
        field_candidates: Dict[str, Dict[str, Any]] = {}
        all_records: List[Dict[str, Any]] = []
        all_layout_tokens: List[Dict[str, Any]] = []
        all_grouped_rows: List[Dict[str, Any]] = []

        for page in pages:
            payload = self.extract_page_structure(page)
            page_payloads.append(payload)
            orientation_counter[payload["orientation"]] = orientation_counter.get(payload["orientation"], 0) + 1

            for field, value in payload["key_values"].items():
                rank = len(value) + (2 if re.search(r"\d", value) else 0)
                existing = field_candidates.get(field)
                if existing is None or rank > existing["rank"]:
                    field_candidates[field] = {
                        "rank": rank,
                        "value": value,
                        "page_number": payload["page_number"],
                    }

            for row in payload["records"]:
                row_with_page: Dict[str, Any] = {"page_number": payload["page_number"]}
                row_with_page.update(row)
                all_records.append(row_with_page)

            for token in payload.get("layout", []):
                token_with_page: Dict[str, Any] = {"page_number": payload["page_number"]}
                token_with_page.update(token)
                all_layout_tokens.append(token_with_page)

            for grouped_row in payload.get("rows", []):
                row_with_page: Dict[str, Any] = {"page_number": payload["page_number"]}
                row_with_page.update(grouped_row)
                all_grouped_rows.append(row_with_page)

        normalized_fields = {field: details["value"] for field, details in field_candidates.items()}

        non_unknown = {key: value for key, value in orientation_counter.items() if key != "unknown" and value > 0}
        if not non_unknown:
            document_orientation = "unknown"
        elif len(non_unknown) > 1:
            document_orientation = "mixed"
        else:
            document_orientation = max(non_unknown.items(), key=lambda item: item[1])[0]

        return {
            "input_file": str(input_path),
            "document_orientation": document_orientation,
            "orientation_votes": orientation_counter,
            "normalized_fields": normalized_fields,
            "records": deduplicate_records(all_records),
            "layout": all_layout_tokens,
            "rows": all_grouped_rows,
            "pages": page_payloads,
        }


def collect_input_files(input_dir: Path) -> List[Path]:
    files: List[Path] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS or suffix in PDF_EXTENSIONS:
            files.append(path)
    return files


def expected_output_text_path(output_root: Path, input_path: Path) -> Path:
    slug = safe_file_slug(input_path)
    return output_root / f"results_{slug}" / f"{slug}.txt"


def expected_output_json_path(output_root: Path, input_path: Path) -> Path:
    slug = safe_file_slug(input_path)
    return output_root / f"results_{slug}" / f"{slug}.json"


def write_output_text(output_root: Path, input_path: Path, text: str) -> Path:
    slug = safe_file_slug(input_path)
    file_dir = output_root / f"results_{slug}"
    file_dir.mkdir(parents=True, exist_ok=True)

    output_txt = file_dir / f"{slug}.txt"
    output_txt.write_text(text, encoding="utf-8")
    return output_txt


def write_output_json(output_root: Path, input_path: Path, payload: Dict[str, Any]) -> Path:
    slug = safe_file_slug(input_path)
    file_dir = output_root / f"results_{slug}"
    file_dir.mkdir(parents=True, exist_ok=True)

    output_json = file_dir / f"{slug}.json"
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="High-accuracy OCR pipeline for PDFs/images with normalized structured JSON output.",
    )
    parser.add_argument(
        "--input-dir",
        "--input",
        "-dir",
        dest="input_dir",
        nargs="?",
        default="Completed",
        const="Completed",
        help="Directory containing PDFs and images.",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        default="results",
        help="Directory where OCR outputs are written.",
    )
    parser.add_argument(
        "--languages",
        default="en,hi",
        help="Comma-separated PaddleOCR language list to try per page/image.",
    )
    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=300,
        help="PDF render DPI used before OCR fallback.",
    )
    parser.add_argument(
        "--min-token-confidence",
        type=float,
        default=0.20,
        help="Minimum token confidence gate for very short OCR tokens.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional limit for number of input files to process (0 means all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess files even if TXT and JSON outputs already exist.",
    )
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="Disable Gemini-based post processing even if GEMINI_API_KEY is available.",
    )
    parser.add_argument(
        "--llm-model",
        default="gemini-2.5-flash",
        help="Gemini model to use for layout-aware JSON refinement.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=60,
        help="Timeout in seconds for Gemini API requests.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logging.error("Input directory does not exist: %s", input_dir)
        return 2

    input_files = collect_input_files(input_dir)
    if args.max_files > 0:
        input_files = input_files[: args.max_files]

    if not input_files:
        logging.warning("No supported PDFs/images found in %s", input_dir)
        return 0

    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    if not languages:
        logging.error("No valid OCR languages were provided.")
        return 2

    extractor = HybridOCR(
        languages=languages,
        render_dpi=args.pdf_dpi,
        min_token_confidence=args.min_token_confidence,
    )
    structurer = StructuredExtractor()
    llm_analyzer = GeminiLayoutAnalyzer(
        model=args.llm_model,
        timeout_seconds=args.llm_timeout,
        enabled=(not args.disable_llm),
    )

    if llm_analyzer.is_available():
        logging.info("Gemini refinement enabled with model=%s", args.llm_model)
    elif not args.disable_llm:
        logging.info("Gemini refinement is skipped because GEMINI_API_KEY is not available.")

    summaries: List[FileSummary] = []
    failures = 0

    logging.info("Found %d files to process.", len(input_files))
    for file_path in input_files:
        expected_txt = expected_output_text_path(output_dir, file_path)
        expected_json = expected_output_json_path(output_dir, file_path)

        if expected_txt.exists() and expected_json.exists() and not args.overwrite:
            logging.info("Skipping existing output for: %s", file_path.name)
            summaries.append(
                FileSummary(
                    input_file=str(file_path),
                    output_txt=str(expected_txt),
                    output_json=str(expected_json),
                    status="skipped",
                    pages=0,
                    extraction_sources=["existing_output"],
                    detected_orientation="unknown",
                )
            )
            continue

        logging.info("Processing: %s", file_path.name)
        try:
            if file_path.suffix.lower() in PDF_EXTENSIONS:
                page_results = extractor.process_pdf_pages(file_path)
            else:
                page_results = extractor.process_image_pages(file_path)

            full_text = "\n\n".join(page.text for page in page_results if page.text)
            sources = [page.source for page in page_results]
            structured_payload = structurer.extract_document_structure(file_path, page_results)

            llm_result = llm_analyzer.refine(structured_payload)
            structured_payload["llm_analysis"] = llm_result

            if llm_result.get("status") == "success":
                analysis = llm_result.get("analysis", {})
                if isinstance(analysis, dict):
                    refined_fields = analysis.get("normalized_fields")
                    if isinstance(refined_fields, dict) and refined_fields:
                        structured_payload["normalized_fields_refined"] = refined_fields

                    refined_records = analysis.get("records")
                    if isinstance(refined_records, list) and refined_records:
                        structured_payload["records_refined"] = refined_records

            structured_payload["final_fields"] = structured_payload.get(
                "normalized_fields_refined",
                structured_payload.get("normalized_fields", {}),
            )
            structured_payload["final_records"] = structured_payload.get(
                "records_refined",
                structured_payload.get("records", []),
            )

            document_orientation = str(structured_payload.get("document_orientation", "unknown"))
            llm_status = str(llm_result.get("status", "not_run"))

            output_txt = write_output_text(output_dir, file_path, full_text)
            output_json = write_output_json(output_dir, file_path, structured_payload)

            logging.info(
                "Saved %s and %s | chars=%d | orientation=%s | llm=%s",
                output_txt,
                output_json,
                len(full_text),
                document_orientation,
                llm_status,
            )

            summaries.append(
                FileSummary(
                    input_file=str(file_path),
                    output_txt=str(output_txt),
                    output_json=str(output_json),
                    status="success",
                    pages=len(page_results),
                    extraction_sources=sources,
                    detected_orientation=document_orientation,
                )
            )
        except Exception as exc:  # noqa: BLE001 - production pipeline should continue per file.
            failures += 1
            logging.exception("Failed processing %s", file_path)
            summaries.append(
                FileSummary(
                    input_file=str(file_path),
                    output_txt="",
                    output_json="",
                    status="failed",
                    pages=0,
                    extraction_sources=[],
                    detected_orientation="unknown",
                    error=str(exc),
                )
            )

    summary_path = output_dir / "ocr_summary.json"
    summary_payload = [asdict(item) for item in summaries]
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    success_count = len([item for item in summaries if item.status == "success"])
    skipped_count = len([item for item in summaries if item.status == "skipped"])
    logging.info(
        "Completed OCR. Success=%d Skipped=%d Failed=%d Summary=%s",
        success_count,
        skipped_count,
        failures,
        summary_path,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
