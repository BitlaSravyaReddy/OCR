import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


GST_PATTERN = re.compile(r"\b[0-9A-Z]{15}\b")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?91[\s\-]?)?[6-9]\d{9}(?!\d)")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
DATE_PATTERN = re.compile(
    r"(?<!\d)(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}|\d{1,2}[\s\-\/]*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[,\s\-\/]*\d{2,4})(?!\d)",
    re.I,
)
INVOICE_PATTERN = re.compile(
    r"(?:invoice\s*(?:no|number|#)|bill\s*(?:no|number)|inv\s*#?)\s*[:\-]?\s*([A-Za-z0-9\-\/]+)",
    re.I,
)
NUMBER_PATTERN = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?)(?!\d)")
TOTAL_LINE_PATTERN = re.compile(r"(total|amount chargeable|grand total|invoice value|net amount)", re.I)


def clean_line(text: str) -> str:
    text = str(text or "").replace("\x00", " ").replace("|", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonical_num(token: str) -> str:
    return re.sub(r"[^\d]", "", token or "")


def parse_amount(token: str) -> Optional[float]:
    text = (token or "").replace(",", "").strip()
    text = re.sub(r"[^0-9.\-]", "", text)
    if text.count(".") > 1:
        left, *rest = text.split(".")
        text = left + "." + "".join(rest)
    if text in {"", ".", "-", "-."}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_lines(text: str) -> List[str]:
    lines = [clean_line(line) for line in (text or "").splitlines()]
    return [line for line in lines if line]


def dedupe_near_lines(lines: Sequence[str], threshold: float = 0.92) -> List[str]:
    unique: List[str] = []
    for line in lines:
        if not line:
            continue
        if any(SequenceMatcher(None, line.lower(), seen.lower()).ratio() >= threshold for seen in unique):
            continue
        unique.append(line)
    return unique


def combine_texts(paddle_text: str, tesseract_text: str) -> str:
    paddle_lines = normalize_lines(paddle_text)
    tess_lines = normalize_lines(tesseract_text)

    combined = paddle_lines[:]
    for line in tess_lines:
        if len(line) < 3:
            continue
        # Prefer cleaner tesseract line when paddle has noisy near-duplicate.
        replaced = False
        for i, existing in enumerate(combined):
            sim = SequenceMatcher(None, line.lower(), existing.lower()).ratio()
            if sim >= 0.86:
                score_existing = _line_quality(existing)
                score_new = _line_quality(line)
                if score_new > score_existing:
                    combined[i] = line
                replaced = True
                break
        if not replaced:
            combined.append(line)

    combined = dedupe_near_lines(combined, threshold=0.92)
    return "\n".join(combined)


def _line_quality(line: str) -> float:
    if not line:
        return 0.0
    useful = sum(1 for ch in line if ch.isalnum() or ch in ",./:-()%")
    bad = sum(1 for ch in line if ch in {"�", "~"})
    return (useful / max(1, len(line))) - (bad / max(1, len(line)))


def extract_key_candidates(text: str, paddle_json: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    candidates: Dict[str, List[str]] = {
        "invoice_number": [],
        "date": [],
        "gstin": [],
        "phone": [],
        "email": [],
        "total_amount": [],
    }

    for match in INVOICE_PATTERN.finditer(text or ""):
        candidates["invoice_number"].append(clean_line(match.group(1)))
    for match in DATE_PATTERN.finditer(text or ""):
        candidates["date"].append(clean_line(match.group(1)))
    for value in GST_PATTERN.findall((text or "").upper()):
        candidates["gstin"].append(value)
    for value in PHONE_PATTERN.findall(text or ""):
        candidates["phone"].append(clean_line(value))
    for value in EMAIL_PATTERN.findall(text or ""):
        candidates["email"].append(clean_line(value))

    for line in normalize_lines(text):
        if TOTAL_LINE_PATTERN.search(line):
            nums = NUMBER_PATTERN.findall(line)
            candidates["total_amount"].extend(nums)

    if isinstance(paddle_json, dict):
        nf = paddle_json.get("normalized_fields", {})
        if isinstance(nf, dict):
            _add_if(nf.get("invoice_no"), candidates["invoice_number"])
            _add_if(nf.get("date"), candidates["date"])
            _add_if(nf.get("gstin"), candidates["gstin"])
            _add_if(nf.get("total_amount"), candidates["total_amount"])
            _add_if(nf.get("net_amount"), candidates["total_amount"])

    return {k: _dedupe_keep_order(v) for k, v in candidates.items() if v}


def _add_if(value: Any, bucket: List[str]) -> None:
    text = clean_line(str(value)) if value is not None else ""
    if text and text.lower() not in {"none", "null", "na", "n/a"}:
        bucket.append(text)


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def resolve_conflicts(
    paddle_candidates: Dict[str, List[str]],
    tesseract_candidates: Dict[str, List[str]],
) -> Dict[str, Optional[str]]:
    resolved: Dict[str, Optional[str]] = {}

    resolved["invoice_number"] = _resolve_generic(
        paddle_candidates.get("invoice_number", []),
        tesseract_candidates.get("invoice_number", []),
    )
    resolved["date"] = _resolve_generic(
        paddle_candidates.get("date", []),
        tesseract_candidates.get("date", []),
    )
    resolved["gstin"] = _resolve_regex_first(
        paddle_candidates.get("gstin", []),
        tesseract_candidates.get("gstin", []),
        GST_PATTERN,
    )
    resolved["phone"] = _resolve_regex_first(
        paddle_candidates.get("phone", []),
        tesseract_candidates.get("phone", []),
        PHONE_PATTERN,
    )
    resolved["email"] = _resolve_regex_first(
        paddle_candidates.get("email", []),
        tesseract_candidates.get("email", []),
        EMAIL_PATTERN,
    )
    resolved["total_amount"] = _resolve_total_amount(
        paddle_candidates.get("total_amount", []),
        tesseract_candidates.get("total_amount", []),
    )
    return resolved


def _resolve_generic(primary: Sequence[str], secondary: Sequence[str]) -> Optional[str]:
    if not primary and not secondary:
        return None
    all_values = [clean_line(v) for v in list(primary) + list(secondary) if clean_line(v)]
    if not all_values:
        return None
    counts = Counter(v.lower() for v in all_values)
    top = sorted(
        all_values,
        key=lambda v: (counts[v.lower()], 1 if v in primary else 0, len(v)),
        reverse=True,
    )
    return top[0]


def _resolve_regex_first(primary: Sequence[str], secondary: Sequence[str], pattern: re.Pattern[str]) -> Optional[str]:
    for value in list(primary) + list(secondary):
        if pattern.search(value or ""):
            return clean_line(value)
    return None


def _resolve_total_amount(primary: Sequence[str], secondary: Sequence[str]) -> Optional[str]:
    if not primary and not secondary:
        return None

    parsed: List[Tuple[str, float, int]] = []
    all_values = list(primary) + list(secondary)
    counts = Counter(clean_line(v).lower() for v in all_values if clean_line(v))
    for value in all_values:
        amount = parse_amount(value)
        if amount is None:
            continue
        parsed.append((clean_line(value), amount, counts[clean_line(value).lower()]))
    if not parsed:
        return _resolve_generic(primary, secondary)

    # Larger magnitude first; repeated values win tie.
    parsed.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return parsed[0][0]


def fix_rows_with_tesseract(rows: Sequence[Dict[str, Any]], paddle_text: str, tesseract_text: str) -> List[Dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    numeric_map = _build_numeric_replacement_map(paddle_text, tesseract_text)
    fixed_rows: List[Dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        row_copy = dict(row)
        row_text = clean_line(row_copy.get("row_text", ""))
        if row_text:
            row_copy["row_text"] = _replace_numbers(row_text, numeric_map)

        cells = row_copy.get("cells")
        if isinstance(cells, list):
            new_cells = []
            for cell in cells:
                if isinstance(cell, dict):
                    c = dict(cell)
                    c["text"] = _replace_numbers(clean_line(c.get("text", "")), numeric_map)
                    new_cells.append(c)
                else:
                    new_cells.append(cell)
            row_copy["cells"] = new_cells
            if not row_copy.get("row_text"):
                row_copy["row_text"] = " | ".join(clean_line(c.get("text", "")) for c in new_cells if isinstance(c, dict))
        fixed_rows.append(row_copy)
    return fixed_rows


def _build_numeric_replacement_map(paddle_text: str, tesseract_text: str) -> Dict[str, str]:
    pnums = NUMBER_PATTERN.findall(paddle_text or "")
    tnums = NUMBER_PATTERN.findall(tesseract_text or "")

    best_tess_by_canon: Dict[str, str] = {}
    for token in tnums:
        canon = canonical_num(token)
        if len(canon) < 4:
            continue
        current = best_tess_by_canon.get(canon)
        if current is None or _number_cleanliness(token) > _number_cleanliness(current):
            best_tess_by_canon[canon] = token

    mapping: Dict[str, str] = {}
    for token in pnums:
        canon = canonical_num(token)
        if len(canon) < 4 or canon not in best_tess_by_canon:
            continue
        best = best_tess_by_canon[canon]
        if _number_cleanliness(best) > _number_cleanliness(token):
            mapping[token] = best
    return mapping


def _number_cleanliness(token: str) -> float:
    token = token or ""
    commas = token.count(",")
    digits = len(re.sub(r"[^\d]", "", token))
    dots = token.count(".")
    return (digits * 1.0) + (commas * 0.5) + (0.2 if dots == 1 else 0.0)


def _replace_numbers(text: str, mapping: Dict[str, str]) -> str:
    if not text or not mapping:
        return text

    def repl(match: re.Match[str]) -> str:
        token = match.group(1)
        return mapping.get(token, token)

    return NUMBER_PATTERN.sub(repl, text)


def fuse_ocr_outputs(
    paddle_json: Optional[Dict[str, Any]],
    paddle_text: str,
    tesseract_text: str,
) -> Tuple[Dict[str, Any], str]:
    """
    Fuse Paddle + Tesseract OCR outputs into (fused_json, fused_text).
    Deterministic strategy with Paddle-first layout trust and Tesseract validation.
    """
    p_json = paddle_json if isinstance(paddle_json, dict) else {}
    p_text = paddle_text or ""
    t_text = tesseract_text or ""

    if not p_json and not p_text and not t_text:
        return {"rows": [], "text": "", "key_values": {}, "normalized_fields": {}}, ""

    fused_text = combine_texts(p_text, t_text) if t_text else p_text

    rows = p_json.get("rows", []) if isinstance(p_json.get("rows"), list) else []
    cleaned_rows = fix_rows_with_tesseract(rows, p_text, t_text) if rows else []

    p_cands = extract_key_candidates(p_text, p_json)
    t_cands = extract_key_candidates(t_text, None)
    resolved = resolve_conflicts(p_cands, t_cands)

    normalized_fields = dict(p_json.get("normalized_fields", {})) if isinstance(p_json.get("normalized_fields"), dict) else {}
    if resolved.get("invoice_number"):
        normalized_fields["invoice_no"] = resolved["invoice_number"]
    if resolved.get("date"):
        normalized_fields["date"] = resolved["date"]
    if resolved.get("gstin"):
        normalized_fields["gstin"] = resolved["gstin"]
    if resolved.get("total_amount"):
        normalized_fields["total_amount"] = resolved["total_amount"]

    key_values = {
        "Invoice_Number": resolved.get("invoice_number"),
        "Date": resolved.get("date"),
        "GSTIN": resolved.get("gstin"),
        "Phone": resolved.get("phone"),
        "Email": resolved.get("email"),
        "Total_Amount": resolved.get("total_amount"),
    }
    key_values = {k: v for k, v in key_values.items() if v}

    fused_json: Dict[str, Any] = dict(p_json)
    fused_json["rows"] = cleaned_rows if cleaned_rows else rows
    fused_json["text"] = fused_text
    fused_json["key_values"] = key_values
    fused_json["normalized_fields"] = normalized_fields
    return fused_json, fused_text

