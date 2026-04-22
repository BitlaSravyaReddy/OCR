import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


GST_RE = re.compile(r"\b[0-9A-Z]{15}\b")
DATE_RE = re.compile(r"(?<!\d)(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}|\d{1,2}[\s\-\/]*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[,\s\-\/]*\d{2,4})(?!\d)", re.I)
INV_RE = re.compile(r"(?:invoice\s*(?:no|number|#)|bill\s*(?:no|number)|inv\s*#?)\s*[:\-]?\s*([A-Za-z0-9\-\/]+)", re.I)
IFSC_RE = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
NUMBER_RE = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?)(?!\d)")
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
HEX_HASH_RE = re.compile(r"\b[a-f0-9]{10,}\b", re.I)

NOISE_LINE_TOKENS = (
    "original",
    "duplicate",
    "triplicate",
    "transport",
    "remarks",
    "reference",
)

GLOBAL_NOISE_PATTERNS = (
    re.compile(r"\birn\b", re.I),
    re.compile(r"\back\s*no\b", re.I),
    re.compile(r"\bmsme\b", re.I),
    re.compile(r"\bqr\s*code\b", re.I),
    re.compile(r"\bterms\s*&?\s*condition", re.I),
    re.compile(r"\bdeclaration\b", re.I),
    re.compile(r"\bthis is a computer generated", re.I),
)

DROP_LINE_PATTERNS = (
    re.compile(r"\birn\b", re.I),
    re.compile(r"\back\b", re.I),
    re.compile(r"\b\d{12,}\b"),
    re.compile(r"\b[a-f0-9]{10,}\b", re.I),
    re.compile(r"^\s*\d{4}\s*[-/]\s*\d{2,4}\s*$"),
)

BLOCK_LABELS = ("header", "supplier", "buyer", "table", "summary", "bank", "other")


def _clean(s: Any) -> str:
    text = str(s or "").replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _to_float(s: Any) -> Optional[float]:
    text = _clean(s).replace(",", "")
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _extract_nums(text: str) -> List[float]:
    out: List[float] = []
    for m in NUMBER_RE.finditer(text):
        v = _to_float(m.group(1))
        if v is not None:
            out.append(v)
    return out


@dataclass
class Line:
    text: str
    x_min: float
    x_max: float
    y: float
    cells: List[Dict[str, Any]]


@dataclass
class Block:
    lines: List[Line]
    label: str = "other"

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def y_top(self) -> float:
        return self.lines[0].y if self.lines else 0.0


class StructureEngine:
    def __init__(self) -> None:
        self._table_stop = ("sub total", "subtotal", "cgst", "sgst", "igst", "g. total", "g total", "grand total", "total")
        self._header_split_tokens = (
            "invoice no",
            "invoice number",
            "dated",
            "date",
            "p.o. no",
            "p.o no",
            "po no",
            "p.o. date",
            "challan no",
            "bill pay status",
            "pay. mode",
            "pay mode",
            "bill credit",
            "due date",
            "delivery by",
            "lr no",
            "last transaction",
            "old balance",
            "adding this invoice amount",
            "new balance",
        )
        self._buyer_anchor_tokens = ("buyer", "bill to", "consignee", "customer", "m/s")
        self._table_anchor_tokens = ("description", "qty", "quantity", "rate", "amount", "hsn", "sac")
        self._totals_anchor_tokens = ("taxable amt", "taxable amount", "cgst", "sgst", "igst", "total gst", "grand total", "g. total", "roundup")

    def build(self, extracted_json: Optional[Dict[str, Any]], extracted_text: str) -> Dict[str, Any]:
        payload = extracted_json if isinstance(extracted_json, dict) else {}
        lines = self._group_lines(payload)
        blocks = self._group_blocks(lines)
        self._classify_blocks(blocks)
        zones = self._build_zones(blocks)

        header = self._extract_header(blocks, extracted_text, zones)
        parties = self._extract_parties(blocks, extracted_text, zones)
        products = self._extract_products(blocks, zones)
        expenses = self._extract_expenses(blocks, zones)
        totals = self._extract_totals(blocks, products, expenses, extracted_text, zones)
        self._validate(parties, products, totals)

        if not products:
            products = [{
                "productName": "Unknown Item",
                "quantity": None,
                "unit": None,
                "rate": None,
                "amount": totals.get("Taxable_Amount"),
                "HSN": None,
                "gst_rate": None,
                "discount_percent": None,
            }]
        if not expenses:
            expenses = [{"name": "Blank", "percentage": 0.0, "amount": 0.0}]

        return {
            "header": header,
            "parties": parties,
            "products": products,
            "expenses": expenses,
            "totals": totals,
            "_ocr_rows": payload.get("rows", []),
            "_ocr_layout": payload.get("layout", []),
            "_structure_debug": {
                "blocks": [
                    {
                        "label": block.label,
                        "line_count": len(block.lines),
                        "preview": [ln.text for ln in block.lines[:3]],
                    }
                    for block in blocks
                ],
                "zones": zones,
            },
        }

    def _build_zones(self, blocks: Sequence[Block]) -> Dict[str, List[str]]:
        lines: List[Line] = []
        for block in blocks:
            lines.extend(block.lines)
        lines.sort(key=lambda ln: ln.y)
        txt_lines = [_clean(ln.text) for ln in lines if _clean(ln.text)]
        if not txt_lines:
            return {
                "header": [],
                "supplier_block": [],
                "buyer_block": [],
                "invoice_meta": [],
                "items_block": [],
                "totals_block": [],
                "footer": [],
            }

        buyer_idx = self._find_idx(txt_lines, self._buyer_anchor_tokens)
        table_idx = self._find_table_idx(txt_lines)
        totals_idx = self._find_idx(txt_lines, self._totals_anchor_tokens, start=(table_idx or 0))
        bank_idx = self._find_idx(txt_lines, ("bank details", "bank name", "a/c", "ifsc", "upi payment"), start=(totals_idx or 0))

        supplier_end = buyer_idx if buyer_idx is not None else (table_idx if table_idx is not None else min(len(txt_lines), 40))
        buyer_start = buyer_idx if buyer_idx is not None else supplier_end
        buyer_end = table_idx if table_idx is not None else len(txt_lines)
        items_start = table_idx if table_idx is not None else buyer_end
        items_end = totals_idx if totals_idx is not None else len(txt_lines)
        totals_start = totals_idx if totals_idx is not None else items_end
        totals_end = bank_idx if bank_idx is not None else len(txt_lines)
        footer_start = bank_idx if bank_idx is not None else totals_end

        supplier_block = [t for t in txt_lines[:supplier_end] if not self._is_global_noise_line(t)]
        buyer_block = [t for t in txt_lines[buyer_start:buyer_end] if not self._is_global_noise_line(t)]
        items_block = [t for t in txt_lines[items_start:items_end] if not self._is_global_noise_line(t)]
        totals_block = [t for t in txt_lines[totals_start:totals_end] if not self._is_global_noise_line(t)]
        footer = [t for t in txt_lines[footer_start:] if not self._is_global_noise_line(t)]
        header = [t for t in txt_lines[: min(len(txt_lines), 20)] if not self._is_global_noise_line(t)]
        invoice_meta = [
            t
            for t in txt_lines[: buyer_end]
            if not self._is_global_noise_line(t)
            and any(k in t.lower() for k in ("invoice no", "invoice number", "invoice date", "dated", "challan", "p.o", "pay mode", "bill credit"))
        ]

        return {
            "header": header,
            "supplier_block": supplier_block,
            "buyer_block": buyer_block,
            "invoice_meta": invoice_meta,
            "items_block": items_block,
            "totals_block": totals_block,
            "footer": footer,
        }

    @staticmethod
    def _find_idx(lines: Sequence[str], tokens: Sequence[str], start: int = 0) -> Optional[int]:
        for i in range(max(0, start), len(lines)):
            low = lines[i].lower()
            if any(tok in low for tok in tokens):
                return i
        return None

    def _find_table_idx(self, lines: Sequence[str]) -> Optional[int]:
        for i, t in enumerate(lines):
            low = t.lower()
            if "description" in low and ("qty" in low or "quantity" in low) and ("rate" in low or "amount" in low):
                return i
            if any(tok in low for tok in ("hsn/sac", "description of goods / service", "billed quantity")):
                return i
        return None

    @staticmethod
    def _is_global_noise_line(text: str) -> bool:
        t = _clean(text)
        if not t:
            return True
        if any(p.search(t) for p in GLOBAL_NOISE_PATTERNS):
            return True
        return False

    @staticmethod
    def _block_from_text_lines(lines: Sequence[str], label: str) -> Optional[Block]:
        if not lines:
            return None
        objs = [Line(text=_clean(t), x_min=0.0, x_max=0.0, y=float(i), cells=[]) for i, t in enumerate(lines) if _clean(t)]
        if not objs:
            return None
        return Block(lines=objs, label=label)

    def _group_lines(self, payload: Dict[str, Any]) -> List[Line]:
        rows = payload.get("rows")
        if not isinstance(rows, list):
            return self._lines_from_text(payload.get("text", ""))

        cells: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_cells = row.get("cells", [])
            if isinstance(row_cells, list):
                for c in row_cells:
                    if not isinstance(c, dict):
                        continue
                    txt = _clean(c.get("text"))
                    if not txt:
                        continue
                    y = c.get("y")
                    x = c.get("x")
                    cells.append({"text": txt, "x": float(x) if isinstance(x, (int, float)) else 0.0, "y": float(y) if isinstance(y, (int, float)) else 0.0})

        if not cells:
            lines: List[Line] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                txt = _clean(row.get("row_text"))
                if not txt:
                    continue
                low = txt.lower()
                if any(tok in low for tok in NOISE_LINE_TOKENS):
                    continue
                y = row.get("y")
                yv = float(y) if isinstance(y, (int, float)) else 0.0
                lines.append(Line(text=txt, x_min=0.0, x_max=0.0, y=yv, cells=[]))
            return sorted(lines, key=lambda l: l.y)

        cells.sort(key=lambda c: (c["y"], c["x"]))
        line_groups: List[List[Dict[str, Any]]] = []
        for cell in cells:
            placed = False
            for grp in line_groups:
                y_avg = sum(x["y"] for x in grp) / max(1, len(grp))
                if abs(cell["y"] - y_avg) <= 10.0:
                    grp.append(cell)
                    placed = True
                    break
            if not placed:
                line_groups.append([cell])

        lines: List[Line] = []
        for grp in line_groups:
            grp.sort(key=lambda c: c["x"])
            text = _clean(" ".join(c["text"] for c in grp))
            if not text:
                continue
            text = self._split_composite_line(text)
            low = text.lower()
            if any(tok in low for tok in NOISE_LINE_TOKENS):
                continue
            if any(p.search(text) for p in DROP_LINE_PATTERNS):
                continue
            lines.append(
                Line(
                    text=text,
                    x_min=min(c["x"] for c in grp),
                    x_max=max(c["x"] for c in grp),
                    y=sum(c["y"] for c in grp) / len(grp),
                    cells=grp,
                )
            )
        lines.sort(key=lambda l: l.y)
        return lines

    def _lines_from_text(self, text: str) -> List[Line]:
        lines: List[Line] = []
        for idx, raw in enumerate((text or "").splitlines()):
            t = _clean(raw)
            if not t:
                continue
            t = self._split_composite_line(t)
            if any(tok in t.lower() for tok in NOISE_LINE_TOKENS):
                continue
            if any(p.search(t) for p in DROP_LINE_PATTERNS):
                continue
            lines.append(Line(text=t, x_min=0.0, x_max=0.0, y=float(idx * 12), cells=[]))
        return lines

    def _split_composite_line(self, text: str) -> str:
        # Remove trailing mixed header tokens that confuse entity selection.
        out = text
        low = out.lower()
        for token in self._header_split_tokens:
            pos = low.find(token)
            if pos > 0:
                out = out[:pos]
                break
        # Strip dangling FY-like suffixes that pollute names: " - 2026 - 27".
        out = re.sub(r"\s*[-|]\s*\d{4}\s*[-/]\s*\d{1,4}\s*$", " ", out)
        return _clean(out)

    def _group_blocks(self, lines: Sequence[Line]) -> List[Block]:
        if not lines:
            return []
        blocks: List[Block] = []
        current: List[Line] = [lines[0]]
        for i in range(1, len(lines)):
            prev = lines[i - 1]
            cur = lines[i]
            y_gap = cur.y - prev.y
            x_overlap = min(prev.x_max, cur.x_max) - max(prev.x_min, cur.x_min)
            if y_gap > 22.0 and x_overlap < 5.0:
                blocks.append(Block(lines=current))
                current = [cur]
            elif y_gap > 28.0:
                blocks.append(Block(lines=current))
                current = [cur]
            else:
                current.append(cur)
        if current:
            blocks.append(Block(lines=current))
        return self._dedupe_blocks(blocks)

    def _dedupe_blocks(self, blocks: Sequence[Block]) -> List[Block]:
        out: List[Block] = []
        seen = set()
        for block in blocks:
            key = re.sub(r"\s+", " ", block.text.lower()).strip()
            if not key:
                continue
            # keep first occurrence only
            if key in seen:
                continue
            seen.add(key)
            out.append(block)
        return out

    def _classify_blocks(self, blocks: Sequence[Block]) -> None:
        for block in blocks:
            txt = block.text.lower()
            scores = {k: 0 for k in BLOCK_LABELS}
            if any(k in txt for k in ("qty", "quantity", "rate", "amount", "description")):
                scores["table"] += 4
            if any(k in txt for k in ("cgst", "sgst", "igst", "sub total", "grand total", "g. total", "total")):
                scores["summary"] += 4
            if any(k in txt for k in ("bank", "a/c", "ifsc", "branch")):
                scores["bank"] += 4
            if any(k in txt for k in ("buyer", "bill to", "consignee", "ship to", "m/s")):
                scores["buyer"] += 4
            if "gstin" in txt and any(k in txt for k in ("invoice", "from", "supplier", "seller")):
                scores["supplier"] += 4
            if any(k in txt for k in ("invoice", "dated", "ack", "irn")):
                scores["header"] += 2
            label = max(scores.items(), key=lambda kv: kv[1])[0]
            block.label = label if scores[label] > 0 else "other"

    def _extract_header(self, blocks: Sequence[Block], extracted_text: str, zones: Optional[Dict[str, List[str]]] = None) -> Dict[str, Optional[str]]:
        zone_lines = []
        if isinstance(zones, dict):
            zone_lines.extend(zones.get("invoice_meta", []))
            zone_lines.extend(zones.get("header", []))
        text = "\n".join(zone_lines) + "\n" + "\n".join(block.text for block in blocks if block.label in {"header", "supplier", "other"}) + "\n" + (extracted_text or "")
        inv = None
        text_lines = [ln for ln in text.splitlines() if _clean(ln)]
        inv_line_idx = None
        for idx, ln in enumerate(text_lines):
            low = ln.lower()
            if "invoice no" in low or "invoice number" in low:
                inv_line_idx = idx
                break
        if inv_line_idx is not None:
            inv_line = text_lines[inv_line_idx]
            mline = re.search(r"invoice\s*(?:no|number)\s*[:#-]?\s*([A-Za-z0-9\/\-]{2,})", inv_line, re.I)
            if mline:
                cand = _clean(mline.group(1))
                if self._is_valid_invoice_token(cand):
                    inv = cand
            if inv is None:
                for j in range(inv_line_idx, min(len(text_lines), inv_line_idx + 4)):
                    for m in re.finditer(r"\b[A-Za-z0-9][A-Za-z0-9\/\-]{2,}\b", text_lines[j]):
                        cand = _clean(m.group(0))
                        if self._is_valid_invoice_token(cand):
                            inv = cand
                            break
                    if inv:
                        break
        dt = None
        for idx, ln in enumerate(text_lines):
            low = ln.lower()
            if "invoice date" in low or re.search(r"\bdated\b", low):
                md = DATE_RE.search(ln)
                if md:
                    dt = _clean(md.group(1))
                    break
                for j in range(idx, min(len(text_lines), idx + 3)):
                    md2 = DATE_RE.search(text_lines[j])
                    if md2:
                        dt = _clean(md2.group(1))
                        break
                if dt:
                    break
        if dt is None:
            md = DATE_RE.search(text)
            if md:
                dt = _clean(md.group(1))
        return {"Invoice_Number": inv, "Invoice_Date": dt}

    @staticmethod
    def _is_valid_invoice_token(token: str) -> bool:
        t = _clean(token)
        if not t:
            return False
        if DATE_RE.search(t):
            return False
        if re.search(r"\b(?:msme|udaym|gstin|tax invoice|invoice)\b", t, re.I):
            return False
        if len(t) < 3:
            return False
        if not re.search(r"\d", t):
            return False
        return True

    def _extract_parties(self, blocks: Sequence[Block], extracted_text: str, zones: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        supplier_block = self._block_from_text_lines((zones or {}).get("supplier_block", []), "supplier")
        buyer_block = self._block_from_text_lines((zones or {}).get("buyer_block", []), "buyer")
        if supplier_block is None:
            supplier_block = self._best_block(blocks, "supplier")
        if buyer_block is None:
            buyer_block = self._best_buyer_block(blocks)

        supplier_clean = self._clean_block_for_party(supplier_block)
        buyer_clean = self._clean_block_for_party(buyer_block)
        supplier_candidates = self._rank_party_candidates(supplier_clean, role="supplier")
        customer_candidates = self._rank_party_candidates(buyer_clean, role="buyer")
        supplier_name = supplier_candidates[0] if supplier_candidates else self._extract_party_name(supplier_block, fallback_anchor="gstin")
        customer_name = customer_candidates[0] if customer_candidates else self._extract_buyer_name(buyer_block)
        if customer_name and customer_name.lower() in {"buyer", "bill to", "consignee", "customer"}:
            customer_name = None
        supplier_gst = self._extract_gst(supplier_block)
        customer_gst = self._extract_gst(buyer_block)
        supplier_addr = self._extract_address_after_name(supplier_block, supplier_name)
        customer_addr = self._extract_address_after_name(buyer_block, customer_name)

        full = (extracted_text or "")
        email = self._first_match(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", full)
        if email and not re.search(r"\.[A-Za-z]{2,}$", email):
            email = None
        phone = self._first_match(r"(?<!\d)(?:\+?91[\s\-]?)?[6-9]\d{9}(?!\d)", full)
        vehicle = self._first_match(r"\b[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}\b", full.upper())
        bank_name, bank_ac, ifsc = self._extract_bank(blocks, full)

        return {
            "SupplierName": supplier_name,
            "Customer_Name": customer_name,
            "Supplier_GST": supplier_gst,
            "Customer_GSTIN": customer_gst if customer_gst != supplier_gst else None,
            "Supplier_Address": supplier_addr,
            "Customer_address": customer_addr,
            "Email": email,
            "Phone": phone,
            "Vehicle_Number": vehicle,
            "Bank_Name": bank_name,
            "bank_account_number": bank_ac,
            "IFSCCode": ifsc,
            "_supplier_candidates": supplier_candidates[:2],
            "_customer_candidates": customer_candidates[:2],
        }

    def _extract_products(self, blocks: Sequence[Block], zones: Optional[Dict[str, List[str]]] = None) -> List[Dict[str, Any]]:
        table_block = self._block_from_text_lines((zones or {}).get("items_block", []), "table")
        if table_block is None:
            table_block = self._best_block(blocks, "table")
        if not table_block:
            return []

        lines = table_block.lines
        header_idx = None
        for i, ln in enumerate(lines):
            low = ln.text.lower()
            if "description" in low and ("qty" in low or "quantity" in low) and "rate" in low:
                header_idx = i
                break
        start = header_idx + 1 if header_idx is not None else 0

        products: List[Dict[str, Any]] = []
        for ln in lines[start:]:
            low = ln.text.lower()
            if any(tok in low for tok in self._table_stop):
                break
            nums = _extract_nums(ln.text)
            if len(nums) < 2:
                if products and ln.text and not NUMBER_RE.search(ln.text):
                    products[-1]["productName"] = _clean(f"{products[-1]['productName']} {ln.text}")
                continue

            qty = None
            rate = None
            amount = None
            gst_rate = None
            disc = None
            hsn = None

            hsn_m = re.search(r"\b\d{4,8}\b", ln.text)
            if hsn_m:
                hsn = hsn_m.group(0)
            pct = [float(m.group(1)) for m in PCT_RE.finditer(ln.text)]
            if pct:
                gst_rate = pct[0]
                if len(pct) > 1:
                    small = [x for x in pct[1:] if x <= 50]
                    disc = small[-1] if small else None

            unit_m = re.search(r"\b(qt|kg|kgs|mt|ton|tons|nos|pcs|piece|box|bag|ltr|litre|unit|doz|bkt)\b", ln.text, re.I)
            unit = unit_m.group(1) if unit_m else "Nos"

            qty_m = re.search(r"\b(\d+(?:\.\d+)?)\s*(qt|kg|kgs|mt|ton|tons|nos|pcs|piece|box|bag|ltr|litre|unit|doz|bkt)\b", ln.text, re.I)
            if qty_m:
                qty = _to_float(qty_m.group(1))

            amount = max(nums) if nums else None
            if qty and len(nums) >= 3:
                mids = [n for n in nums if n != amount]
                rate = max(mids) if mids else None
            elif len(nums) >= 2:
                rate = nums[-2]
                qty = nums[0]

            desc = ln.text
            desc = re.sub(r"\b\d{4,8}\b", " ", desc)
            desc = re.sub(r"\b\d+(?:\.\d+)?\s*(?:qt|kg|kgs|mt|ton|tons|nos|pcs|piece|box|bag|ltr|litre|unit|doz|bkt)\b", " ", desc, flags=re.I)
            desc = NUMBER_RE.sub(" ", desc)
            desc = PCT_RE.sub(" ", desc)
            desc = _clean(desc)
            if not desc:
                desc = "Unknown Item"

            products.append(
                {
                    "productName": desc,
                    "quantity": qty,
                    "unit": unit,
                    "rate": rate,
                    "amount": amount,
                    "HSN": hsn,
                    "gst_rate": gst_rate,
                    "discount_percent": disc,
                }
            )

        return products

    def _extract_expenses(self, blocks: Sequence[Block], zones: Optional[Dict[str, List[str]]] = None) -> List[Dict[str, Any]]:
        summary = self._block_from_text_lines((zones or {}).get("totals_block", []), "summary")
        if summary is None:
            summary = self._best_block(blocks, "summary")
        if not summary:
            return []
        out: List[Dict[str, Any]] = []
        for ln in summary.lines:
            low = ln.text.lower()
            if any(k in low for k in ("cgst", "sgst", "igst", "tax", "g. total", "grand total", "sub total", "total")):
                continue
            if any(k in low for k in ("a/c", "ifsc", "prev due balance", "dr")):
                continue
            if not any(k in low for k in ("round", "freight", "packing", "labour", "commission", "charge", "fee")):
                continue
            nums = _extract_nums(ln.text)
            if not nums:
                continue
            amt = nums[-1]
            pct = None
            pm = PCT_RE.search(ln.text)
            if pm:
                pct = _to_float(pm.group(1))
            name = NUMBER_RE.sub(" ", ln.text)
            name = PCT_RE.sub(" ", name)
            name = _clean(name)
            out.append({"name": name or "Expense", "percentage": pct, "amount": amt})
        return out

    def _extract_totals(
        self,
        blocks: Sequence[Block],
        products: Sequence[Dict[str, Any]],
        expenses: Sequence[Dict[str, Any]],
        extracted_text: str,
        zones: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Optional[float]]:
        zone_text = ""
        if isinstance(zones, dict):
            zone_text = "\n".join(zones.get("totals_block", []) + zones.get("invoice_meta", []))
        text = zone_text + "\n" + "\n".join(block.text for block in blocks) + "\n" + (extracted_text or "")
        sgst = self._amount_from_keyword(text, "sgst")
        cgst = self._amount_from_keyword(text, "cgst")
        igst = self._amount_from_keyword(text, "igst")
        total = self._amount_from_any(text, ("g. total", "g total", "grand total"))
        if total is None:
            total = self._amount_from_any(text, ("total",))
        taxable = self._amount_after_any(text, ("taxable amt", "taxable amount", "sub total"))
        if taxable is None or taxable < 1.0:
            taxable_from_tax = self._taxable_from_tax_lines(text)
            if taxable_from_tax is not None:
                taxable = taxable_from_tax
        if taxable is None:
            vals = [p.get("amount") for p in products if isinstance(p.get("amount"), (int, float))]
            taxable = sum(float(v) for v in vals) if vals else None
        elif taxable is not None and taxable < 1.0:
            vals = [p.get("amount") for p in products if isinstance(p.get("amount"), (int, float))]
            if vals:
                taxable = sum(float(v) for v in vals)
        total_exp = sum(float(e.get("amount") or 0.0) for e in expenses)
        return {
            "SGST_Amount": sgst,
            "CGST_Amount": cgst,
            "IGST_Amount": igst,
            "Taxable_Amount": taxable,
            "Total_Expenses": total_exp,
            "Total_Amount": total,
        }

    def _validate(self, parties: Dict[str, Any], products: Sequence[Dict[str, Any]], totals: Dict[str, Any]) -> None:
        for k in ("Supplier_GST", "Customer_GSTIN"):
            v = parties.get(k)
            if not isinstance(v, str) or not GST_RE.fullmatch(v.strip().upper()):
                parties[k] = None

        sgst = _to_float(totals.get("SGST_Amount")) or 0.0
        cgst = _to_float(totals.get("CGST_Amount")) or 0.0
        igst = _to_float(totals.get("IGST_Amount")) or 0.0
        taxable = _to_float(totals.get("Taxable_Amount")) or 0.0
        total = _to_float(totals.get("Total_Amount")) or 0.0
        exp = _to_float(totals.get("Total_Expenses")) or 0.0

        if total <= 0 and taxable > 0:
            totals["Total_Amount"] = round(taxable + sgst + cgst + igst + exp, 2)

        if parties.get("SupplierName") and parties.get("Bank_Name"):
            if str(parties["SupplierName"]).strip().lower() == str(parties["Bank_Name"]).strip().lower():
                parties["Bank_Name"] = None

        for p in products:
            q = _to_float(p.get("quantity"))
            if q is not None and (q < 0 or q > 1_000_000):
                p["quantity"] = None

    def _best_block(self, blocks: Sequence[Block], label: str) -> Optional[Block]:
        labeled = [b for b in blocks if b.label == label]
        if not labeled:
            return None
        return sorted(labeled, key=lambda b: (len(b.lines), -b.y_top), reverse=True)[0]

    def _best_buyer_block(self, blocks: Sequence[Block]) -> Optional[Block]:
        # Priority: explicit Buyer(Bill to) block, then other buyer-like blocks.
        buyers = [b for b in blocks if b.label == "buyer"]
        if not buyers:
            return None
        buyers_sorted = sorted(
            buyers,
            key=lambda b: (
                1 if re.search(r"buyer\s*\(bill to\)|buyer|bill to|m/s", b.text, re.I) else 0,
                len(b.lines),
            ),
            reverse=True,
        )
        return buyers_sorted[0]

    def _rank_party_candidates(self, block: Optional[Block], role: str) -> List[str]:
        if not block:
            return []
        raw_lines = [ln.text for ln in block.lines]
        if role == "buyer":
            lines = [_clean(self._split_composite_line(t)) for t in raw_lines]
            lines = [t for t in lines if t]
        else:
            lines = self._merge_multiline_entities(raw_lines)
        candidates: List[Tuple[int, str]] = []
        for text in lines:
            score = self._score_party_candidate(text, role)
            if score <= 0:
                continue
            candidates.append((score, text))
        candidates.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        dedup: List[str] = []
        seen = set()
        for _score, text in candidates:
            key = re.sub(r"\s+", " ", text.lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(_clean(text))
        return dedup

    def _clean_block_for_party(self, block: Optional[Block]) -> Optional[Block]:
        if not block:
            return None
        cleaned: List[Line] = []
        for ln in block.lines:
            text = self._split_composite_line(ln.text)
            if not text:
                continue
            low = text.lower()
            if any(tok in low for tok in NOISE_LINE_TOKENS):
                continue
            if any(p.search(text) for p in DROP_LINE_PATTERNS):
                continue
            # Drop pure numeric/id rows and long hash-like fragments.
            if re.fullmatch(r"[A-Fa-f0-9]{10,}", text):
                continue
            if re.fullmatch(r"\d{6,}", text):
                continue
            cleaned.append(
                Line(
                    text=text,
                    x_min=ln.x_min,
                    x_max=ln.x_max,
                    y=ln.y,
                    cells=ln.cells,
                )
            )
        if not cleaned:
            return None
        return Block(lines=cleaned, label=block.label)

    def _score_party_candidate(self, text: str, role: str) -> int:
        t = _clean(text)
        t = re.sub(r"\s*[-|]\s*\d{4}\s*[-/]\s*\d{1,4}\s*$", " ", t)
        if role == "buyer":
            t = re.sub(r"^\s*(buyer|bill to|consignee|customer)\s*[:\-]?\s*", "", t, flags=re.I)
        t = _clean(t)
        low = t.lower()
        score = 0
        if not t:
            return -99
        if low in {"buyer", "bill to", "consignee", "customer"}:
            return -99
        if any(p.search(t) for p in DROP_LINE_PATTERNS):
            return -99
        if HEX_HASH_RE.search(t):
            return -99
        if GST_RE.search(t):
            return -99
        if IFSC_RE.search(t.upper()):
            return -99
        if any(k in low for k in ("original", "duplicate", "triplicate", "copy", "irn", "ack", "invoice", "dated", "bank")):
            score -= 8
        if any(
            k in low
            for k in (
                "contact person",
                "mobile",
                "p.o",
                "challan",
                "bill pay status",
                "pay mode",
                "bill credit",
                "due date",
                "delivery by",
                "lr no",
                "old balance",
                "new balance",
            )
        ):
            score -= 8
        if any(word in low for word in ("corp", "corporation", "pvt", "ltd", "traders", "co.", "company", "motors")):
            score += 5
        if any(word in low for word in ("motor", "parts", "auto", "agencies", "agency", "industries", "enterprises", "stores")):
            score += 4
        if any(word in low for word in ("road", "bazar", "bazaar", "nagar", "state", "district", "pin", "madhya pradesh", "indore")):
            score -= 5
        if len(re.findall(r"[A-Za-z]", t)) >= 5:
            score += 3
        if t.isupper() and len(t.split()) >= 2:
            score += 2
        if any(c.isdigit() for c in t):
            score -= 4
        if role == "buyer" and any(k in low for k in ("buyer", "bill to", "m/s", "consignee")):
            score += 3
        return score

    def _merge_multiline_entities(self, lines: Sequence[str]) -> List[str]:
        merged: List[str] = []
        buf = ""
        for raw in lines:
            line = _clean(raw)
            if not line:
                continue
            if any(p.search(line) for p in DROP_LINE_PATTERNS):
                continue
            if not buf:
                buf = line
                continue
            if self._is_continuation(line):
                buf = _clean(f"{buf} {line}")
            else:
                merged.append(buf)
                buf = line
        if buf:
            merged.append(buf)
        return merged

    @staticmethod
    def _is_continuation(line: str) -> bool:
        low = line.lower()
        if len(line) <= 3:
            return True
        if any(k in low for k in ("road", "street", "nagar", "state", "city", "dist", "malwa", "indore", "compound", "opp", "market")):
            return True
        if line.startswith(",") or line.startswith("-"):
            return True
        return False

    def _extract_party_name(self, block: Optional[Block], fallback_anchor: str) -> Optional[str]:
        if not block:
            return None
        lines = [ln.text for ln in block.lines]
        gst_idx = None
        for i, line in enumerate(lines):
            if fallback_anchor in line.lower():
                gst_idx = i
                break
        scan = lines if gst_idx is None else lines[max(0, gst_idx - 4) : gst_idx + 1]
        for line in scan:
            low = line.lower()
            if any(tok in low for tok in ("invoice", "dated", "gstin", "original", "duplicate", "copy")):
                continue
            if len(re.findall(r"[A-Za-z]", line)) >= 5:
                return _clean(re.sub(r"\(.*?\)", " ", line))
        return None

    def _extract_buyer_name(self, block: Optional[Block]) -> Optional[str]:
        if not block:
            return None
        for i, ln in enumerate(block.lines):
            low = ln.text.lower()
            if any(k in low for k in ("buyer", "bill to", "consignee", "m/s")):
                for j in range(i + 1, min(len(block.lines), i + 5)):
                    t = self._split_composite_line(block.lines[j].text)
                    lt = t.lower()
                    if "gstin" in lt:
                        break
                    if len(re.findall(r"[A-Za-z]", t)) >= 4:
                        return _clean(t)
        for ln in block.lines:
            t = self._split_composite_line(ln.text)
            lt = t.lower()
            if any(k in lt for k in ("buyer", "bill to", "consignee", "gstin")):
                continue
            if len(re.findall(r"[A-Za-z]", t)) >= 4:
                return _clean(t)
        return None

    def _extract_address_after_name(self, block: Optional[Block], name: Optional[str]) -> Optional[str]:
        if not block:
            return None
        lines = [ln.text for ln in block.lines]
        start = 0
        if name:
            for i, t in enumerate(lines):
                if _clean(name).lower() in _clean(t).lower():
                    start = i + 1
                    break
        parts: List[str] = []
        for t in lines[start : min(len(lines), start + 8)]:
            low = t.lower()
            if any(k in low for k in ("gstin", "contact", "description", "qty", "rate", "amount", "invoice", "dated")):
                if "state name" not in low:
                    break
            if any(k in low for k in ("buyer", "bill to", "consignee", "bank")):
                break
            if len(_clean(t)) < 3:
                continue
            cleaned = _clean(t.replace("|", " "))
            if cleaned:
                parts.append(cleaned)
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return f"{parts[0]} {' '.join(parts[1:-1]).strip()}{', ' if len(parts) > 1 else ''}{parts[-1]}".strip(" ,")

    def _extract_gst(self, block: Optional[Block]) -> Optional[str]:
        if not block:
            return None
        txt = block.text.upper()
        found = GST_RE.findall(txt)
        return found[0] if found else None

    def _extract_bank(self, blocks: Sequence[Block], full_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        bank_block = self._best_block(blocks, "bank")
        text = (bank_block.text + "\n" + full_text) if bank_block else full_text
        bank_name = None
        for ln in text.splitlines():
            m = re.search(r"\bbank(?:\s*name)?\s*:\s*(.+)$", ln, re.I)
            if not m:
                continue
            val = _clean(m.group(1))
            val = re.split(r"\b(a/c|account|ifsc)\b", val, flags=re.I)[0].strip(" :-|")
            if val:
                bank_name = val
                break
        ac = self._first_match(r"\b\d{8,20}\b", text)
        ifsc = self._first_match(IFSC_RE.pattern, text.upper())
        return bank_name, ac, ifsc

    def _amount_from_keyword(self, text: str, keyword: str) -> Optional[float]:
        for ln in text.splitlines():
            if keyword not in ln.lower():
                continue
            nums = _extract_nums(ln)
            if nums:
                return nums[-1]
        return None

    def _amount_from_any(self, text: str, keys: Sequence[str]) -> Optional[float]:
        for ln in text.splitlines():
            low = ln.lower()
            if not any(k in low for k in keys):
                continue
            nums = _extract_nums(ln)
            if nums:
                return max(nums)
        return None

    def _amount_after_any(self, text: str, keys: Sequence[str]) -> Optional[float]:
        for ln in text.splitlines():
            low = ln.lower()
            if not any(k in low for k in keys):
                continue
            nums = _extract_nums(ln)
            if nums:
                return nums[0]
        return None

    def _taxable_from_tax_lines(self, text: str) -> Optional[float]:
        candidates: List[float] = []
        for ln in text.splitlines():
            low = ln.lower()
            if not any(k in low for k in ("cgst", "sgst", "igst")):
                continue
            nums = _extract_nums(ln)
            if len(nums) >= 2:
                first = nums[0]
                if first > 1:
                    candidates.append(first)
        if not candidates:
            return None
        return max(candidates)

    @staticmethod
    def _first_match(pattern: str, text: str) -> Optional[str]:
        m = re.search(pattern, text or "", re.I)
        return m.group(0) if m else None


def build_structured_invoice(extracted_json: Optional[Dict[str, Any]], extracted_text: str) -> Dict[str, Any]:
    return StructureEngine().build(extracted_json=extracted_json, extracted_text=extracted_text)
