import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


GST_PATTERN = re.compile(r"\b[0-9A-Z]{15}\b")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?91[\s\-]?)?[6-9]\d{9}(?!\d)")

SUPPLIER_ANCHORS = ("supplier", "seller", "from", "issuer")
CUSTOMER_ANCHORS = ("buyer", "bill to", "ship to", "consignee", "customer")
PRODUCT_ANCHORS = (
    "description of goods",
    "hsn",
    "quantity",
    "qty",
    "rate",
    "amount",
    "total",
)

NOISE_TOKENS = (
    "invoice",
    "date",
    "delivery note",
    "order no",
    "motor vehicle",
    "terms of delivery",
    "description of goods",
    "state code",
    "consignee",
    "ship to",
    "bill to",
    "buyer",
    "supplier",
    "seller",
    "gst invoice",
    "tax invoice",
)

ADDRESS_HINTS = (
    "road",
    "rd",
    "street",
    "st",
    "lane",
    "ln",
    "nagar",
    "sector",
    "block",
    "gate",
    "plot",
    "village",
    "mandi",
    "market",
    "near",
    "opp",
    "tehsil",
    "district",
    "state",
    "pincode",
    "pin",
    "h.no",
    "h no",
)

BUSINESS_HINTS = (
    "traders",
    "trading",
    "enterprise",
    "enterprises",
    "industries",
    "motors",
    "private",
    "limited",
    "pvt",
    "llp",
    "agency",
    "agencies",
    "co.",
    "company",
)


@dataclass
class PartyBlock:
    start: int
    end: int
    lines: List[str]
    label: str


def _clean(line: str) -> str:
    text = str(line or "").replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_lines(text: str) -> List[str]:
    lines = [_clean(line) for line in (text or "").splitlines()]
    return [line for line in lines if line]


def _left_header_lines_from_rows(rows: Any) -> List[str]:
    if not isinstance(rows, list) or not rows:
        return []

    header_rows: List[Dict[str, Any]] = []
    for row in rows[:120]:
        if not isinstance(row, dict):
            continue
        rt = _clean(row.get("row_text", ""))
        if rt and any(k in rt.lower() for k in PRODUCT_ANCHORS):
            break
        header_rows.append(row)

    xs: List[float] = []
    for row in header_rows:
        cells = row.get("cells", [])
        if not isinstance(cells, list):
            continue
        for cell in cells:
            if isinstance(cell, dict) and isinstance(cell.get("x"), (int, float)):
                xs.append(float(cell["x"]))
    if not xs:
        return []

    split_x = max(xs) * 0.58
    out: List[str] = []
    for row in header_rows:
        cells = row.get("cells", [])
        parts: List[str] = []
        if isinstance(cells, list):
            sorted_cells = sorted(
                [c for c in cells if isinstance(c, dict)],
                key=lambda c: float(c.get("x", 0.0)),
            )
            for c in sorted_cells:
                x = c.get("x")
                if not isinstance(x, (int, float)):
                    continue
                if float(x) <= split_x:
                    t = _clean(c.get("text", ""))
                    if t:
                        parts.append(t)
        line = _clean(" ".join(parts))
        if line and line not in out:
            out.append(line)
    return out


def _is_company_like(line: str) -> bool:
    if not line:
        return False
    lower = line.lower()
    if any(tok in lower for tok in NOISE_TOKENS):
        return False
    if re.fullmatch(r"(consignee|ship to|bill to|buyer|supplier|seller|customer)\b.*", lower):
        return False
    if ":" in line and any(tok in lower for tok in ("consignee", "ship to", "bill to", "buyer", "supplier", "seller")):
        return False
    if GST_PATTERN.search(line.upper()):
        return False
    if EMAIL_PATTERN.search(line):
        return False
    if PHONE_PATTERN.search(line):
        return False
    alpha = sum(1 for ch in line if ch.isalpha())
    digits = sum(1 for ch in line if ch.isdigit())
    if alpha < 4:
        return False
    if digits > alpha:
        return False
    if len(line) > 90:
        return False
    return True


def _looks_address_line(line: str) -> bool:
    low = (line or "").lower()
    if not low:
        return False
    if any(h in low for h in ADDRESS_HINTS):
        return True
    if bool(re.search(r"\b\d+/\d+\b", low)):
        return True
    if bool(re.search(r"^\s*\d{1,5}\s+[a-z]", low)):
        return True
    if bool(re.search(r"\b\d{6}\b", low)):
        return True
    if "," in low:
        return True
    return False


def _find_first_anchor(lines: Sequence[str], anchors: Sequence[str]) -> Optional[int]:
    for i, line in enumerate(lines):
        low = line.lower()
        if any(anchor in low for anchor in anchors):
            return i
    return None


def _find_product_start(lines: Sequence[str]) -> int:
    for i, line in enumerate(lines):
        low = line.lower()
        if any(anchor in low for anchor in PRODUCT_ANCHORS):
            return i
    return len(lines)


def _slice_block(lines: Sequence[str], start: int, end: int, label: str) -> PartyBlock:
    start = max(0, start)
    end = max(start, min(len(lines), end))
    return PartyBlock(start=start, end=end, lines=list(lines[start:end]), label=label)


def _nearest_company_above(lines: Sequence[str], idx: int, lookback: int = 4) -> Optional[int]:
    for scan in range(idx - 1, max(-1, idx - lookback - 1), -1):
        if scan >= 0 and _is_company_like(lines[scan]):
            return scan
    return None


def _extract_name(block: PartyBlock) -> Optional[str]:
    candidates: List[Tuple[int, int, str]] = []
    for i, line in enumerate(block.lines):
        if not _is_company_like(line):
            continue
        score = 0
        low = line.lower()
        if any(h in low for h in BUSINESS_HINTS):
            score += 3
        if i + 1 < len(block.lines) and _looks_address_line(block.lines[i + 1]):
            score += 2
        if i > 0 and _looks_address_line(block.lines[i - 1]):
            score -= 2
        if len(line.split()) >= 2:
            score += 1
        # Penalize probable city-only lines like "AGAR MALWA" that often follow street lines.
        tokens = [t for t in re.split(r"\s+", line) if t]
        if len(tokens) <= 3 and all(t.isalpha() for t in tokens) and all(t.upper() == t for t in tokens):
            if not any(h in low for h in BUSINESS_HINTS):
                score -= 2
        # Prefer earlier lines in block when scores tie.
        candidates.append((-score, i, line))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _extract_gst(block: PartyBlock) -> Optional[str]:
    text = "\n".join(block.lines).upper()
    found = GST_PATTERN.findall(text)
    return found[0] if found else None


def _extract_email(block: PartyBlock) -> Optional[str]:
    text = "\n".join(block.lines)
    match = EMAIL_PATTERN.search(text)
    return match.group(0) if match else None


def _extract_phone(block: PartyBlock) -> Optional[str]:
    text = "\n".join(block.lines)
    phones = PHONE_PATTERN.findall(text)
    if not phones:
        return None
    seen = []
    for phone in phones:
        p = _clean(phone)
        if p not in seen:
            seen.append(p)
    return " | ".join(seen)


def _extract_address(block: PartyBlock, name: Optional[str]) -> Optional[str]:
    parts: List[str] = []
    for line in block.lines:
        if name and line == name:
            continue
        if GST_PATTERN.search(line.upper()) or EMAIL_PATTERN.search(line) or PHONE_PATTERN.search(line):
            continue
        low = line.lower()
        if any(tok in low for tok in NOISE_TOKENS):
            continue
        if any(anchor in low for anchor in SUPPLIER_ANCHORS + CUSTOMER_ANCHORS):
            continue
        if len(line) < 4:
            continue
        parts.append(line)
    if not parts:
        return None
    return " | ".join(parts[:4])


def _build_blocks_from_anchors(lines: Sequence[str]) -> Tuple[Optional[PartyBlock], Optional[PartyBlock], str]:
    product_start = _find_product_start(lines)
    header_lines = list(lines[:product_start])
    supplier_anchor = _find_first_anchor(header_lines, SUPPLIER_ANCHORS)
    customer_anchor = _find_first_anchor(header_lines, CUSTOMER_ANCHORS)

    if supplier_anchor is not None and customer_anchor is not None:
        if supplier_anchor < customer_anchor:
            supplier = _slice_block(header_lines, supplier_anchor, customer_anchor, "supplier")
            customer = _slice_block(header_lines, customer_anchor, min(len(header_lines), customer_anchor + 8), "customer")
        else:
            customer = _slice_block(header_lines, customer_anchor, supplier_anchor, "customer")
            supplier = _slice_block(header_lines, supplier_anchor, min(len(header_lines), supplier_anchor + 8), "supplier")
        return supplier, customer, "anchor_both"

    if customer_anchor is not None:
        supplier = _slice_block(header_lines, 0, customer_anchor, "supplier")
        customer = _slice_block(header_lines, customer_anchor, min(len(header_lines), customer_anchor + 8), "customer")
        return supplier, customer, "anchor_customer_only"

    if supplier_anchor is not None:
        supplier = _slice_block(header_lines, supplier_anchor, min(len(header_lines), supplier_anchor + 8), "supplier")
        customer = _slice_block(header_lines, supplier.end, min(len(header_lines), supplier.end + 8), "customer")
        return supplier, customer, "anchor_supplier_only"

    return None, None, "no_anchor"


def _build_blocks_from_gst(lines: Sequence[str]) -> Tuple[Optional[PartyBlock], Optional[PartyBlock], str]:
    product_start = _find_product_start(lines)
    header_lines = list(lines[:product_start])
    gst_indices = [idx for idx, line in enumerate(header_lines) if GST_PATTERN.search(line.upper())]
    if not gst_indices:
        return None, None, "no_gst"

    block_starts: List[int] = []
    for gst_idx in gst_indices[:2]:
        company_idx = _nearest_company_above(header_lines, gst_idx)
        if company_idx is not None:
            block_starts.append(company_idx)
        else:
            block_starts.append(max(0, gst_idx - 1))

    unique_starts = sorted(set(block_starts))
    if not unique_starts:
        return None, None, "gst_no_company"

    supplier_start = unique_starts[0]
    customer_start = unique_starts[1] if len(unique_starts) > 1 else None
    supplier_end = customer_start if customer_start is not None else min(len(header_lines), supplier_start + 8)
    supplier = _slice_block(header_lines, supplier_start, supplier_end, "supplier")
    customer = (
        _slice_block(header_lines, customer_start, min(len(header_lines), customer_start + 8), "customer")
        if customer_start is not None
        else None
    )
    return supplier, customer, "gst_proximity"


def _build_blocks_by_position(lines: Sequence[str]) -> Tuple[Optional[PartyBlock], Optional[PartyBlock], str]:
    product_start = _find_product_start(lines)
    header_lines = list(lines[:product_start])
    company_indices = [idx for idx, line in enumerate(header_lines) if _is_company_like(line)]
    if not company_indices:
        return None, None, "position_no_company"
    supplier_start = company_indices[0]
    customer_start = company_indices[1] if len(company_indices) > 1 else None
    supplier_end = customer_start if customer_start is not None else min(len(header_lines), supplier_start + 8)
    supplier = _slice_block(header_lines, supplier_start, supplier_end, "supplier")
    customer = (
        _slice_block(header_lines, customer_start, min(len(header_lines), customer_start + 8), "customer")
        if customer_start is not None
        else None
    )
    return supplier, customer, "position_fallback"


def _associate_gst_by_proximity(supplier: PartyBlock, customer: Optional[PartyBlock], lines: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    gst_rows: List[Tuple[int, str]] = []
    for i, line in enumerate(lines):
        for gst in GST_PATTERN.findall(line.upper()):
            gst_rows.append((i, gst))

    if not gst_rows:
        return None, None
    if len(gst_rows) == 1:
        return gst_rows[0][1], None

    def block_center(block: PartyBlock) -> float:
        return (block.start + block.end) / 2.0

    supplier_center = block_center(supplier)
    customer_center = block_center(customer) if customer is not None else None
    sup_gst: Optional[str] = None
    cus_gst: Optional[str] = None

    for idx, gst in gst_rows:
        d_sup = abs(idx - supplier_center)
        if customer_center is None:
            if sup_gst is None:
                sup_gst = gst
            continue
        d_cus = abs(idx - customer_center)
        if d_sup <= d_cus:
            if sup_gst is None:
                sup_gst = gst
        else:
            if cus_gst is None:
                cus_gst = gst

    return sup_gst, cus_gst


def _confidence(method: str, supplier_name: Optional[str], customer_name: Optional[str], supplier_gst: Optional[str], customer_gst: Optional[str]) -> float:
    base = {
        "anchor_both": 0.90,
        "anchor_customer_only": 0.78,
        "anchor_supplier_only": 0.76,
        "gst_proximity": 0.72,
        "position_fallback": 0.60,
    }.get(method, 0.45)
    if supplier_name:
        base += 0.05
    if customer_name:
        base += 0.05
    if supplier_gst:
        base += 0.04
    if customer_gst:
        base += 0.04
    return round(min(0.99, base), 2)


def _null_if_uncertain(value: Optional[str], min_len: int = 2) -> Optional[str]:
    if value is None:
        return None
    value = _clean(value)
    if len(value) < min_len:
        return None
    return value


def segment_parties(structured_data: Dict[str, Any], fused_text: str) -> Dict[str, Any]:
    """
    Segment supplier and customer from preprocessed data + fused OCR text.
    Deterministic heuristics with anchor, positional, and GST proximity rules.
    """
    row_lines = _left_header_lines_from_rows(structured_data.get("_ocr_rows"))
    lines = row_lines if len(row_lines) >= 4 else _split_lines(fused_text)
    supplier_block, customer_block, method = _build_blocks_from_anchors(lines)
    if supplier_block is None:
        supplier_block, customer_block, method = _build_blocks_from_gst(lines)
    if supplier_block is None:
        supplier_block, customer_block, method = _build_blocks_by_position(lines)

    pre_parties = structured_data.get("parties", {}) if isinstance(structured_data.get("parties"), dict) else {}

    if supplier_block is None:
        supplier = {
            "name": _null_if_uncertain(pre_parties.get("SupplierName")),
            "gst": _null_if_uncertain(pre_parties.get("Supplier_GST")),
            "address": _null_if_uncertain(pre_parties.get("Address")),
            "phone": _null_if_uncertain(pre_parties.get("Phone")),
            "email": _null_if_uncertain(pre_parties.get("Email")),
        }
        customer = {
            "name": _null_if_uncertain(pre_parties.get("Customer_Name")),
            "gst": _null_if_uncertain(pre_parties.get("Customer_GSTIN")),
            "address": None,
        }
        return {
            "supplier": supplier,
            "customer": customer,
            "debug": {"supplier_block": "", "customer_block": "", "confidence_score": 0.3, "method": "preprocessed_fallback"},
        }

    supplier_name = _extract_name(supplier_block) or _null_if_uncertain(pre_parties.get("SupplierName"))
    customer_name = (_extract_name(customer_block) if customer_block else None) or _null_if_uncertain(pre_parties.get("Customer_Name"))

    supplier_gst = _extract_gst(supplier_block)
    customer_gst = _extract_gst(customer_block) if customer_block else None
    prox_sup_gst, prox_cus_gst = _associate_gst_by_proximity(supplier_block, customer_block, lines)
    supplier_gst = supplier_gst or prox_sup_gst or _null_if_uncertain(pre_parties.get("Supplier_GST"))
    customer_gst = customer_gst or prox_cus_gst or _null_if_uncertain(pre_parties.get("Customer_GSTIN"))

    supplier = {
        "name": _null_if_uncertain(supplier_name),
        "gst": _null_if_uncertain(supplier_gst, min_len=15),
        "address": _null_if_uncertain(_extract_address(supplier_block, supplier_name)),
        "phone": _null_if_uncertain(_extract_phone(supplier_block)),
        "email": _null_if_uncertain(_extract_email(supplier_block)),
    }
    customer = {
        "name": _null_if_uncertain(customer_name),
        "gst": _null_if_uncertain(customer_gst, min_len=15),
        "address": _null_if_uncertain(_extract_address(customer_block, customer_name)) if customer_block else None,
    }

    confidence = _confidence(method, supplier["name"], customer["name"], supplier["gst"], customer["gst"])

    return {
        "supplier": supplier,
        "customer": customer,
        "debug": {
            "supplier_block": "\n".join(supplier_block.lines),
            "customer_block": "\n".join(customer_block.lines) if customer_block else "",
            "confidence_score": confidence,
            "method": method,
        },
    }
