import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SECTION_HEADER = "HEADER"
SECTION_PARTY = "PARTY"
SECTION_PRODUCT = "PRODUCT"
SECTION_EXPENSE = "EXPENSE"
SECTION_TOTAL = "TOTAL"
SECTION_OTHER = "OTHER"


INVOICE_NUMBER_PATTERNS = [
    re.compile(r"(?:invoice\s*(?:no|number|#)|bill\s*(?:no|number)|inv\s*#?)\s*[:\-]?\s*([A-Za-z0-9\-\/]+)", re.I),
    re.compile(r"\b(?:inv)\s*[:\-]?\s*([A-Za-z0-9\-\/]{2,})\b", re.I),
]
DATE_PATTERN = re.compile(
    r"(?<!\d)(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}|\d{1,2}[\s\-\/]*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[,\s\-\/]*\d{2,4})(?!\d)",
    re.I,
)
GSTIN_PATTERN = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?91[\s\-]?)?[6-9]\d{9}(?!\d)")
VEHICLE_PATTERN = re.compile(r"\b[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}\b", re.I)
NUMBER_PATTERN = re.compile(r"(?<!\d)(-?\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)(?!\d)")
PERCENT_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")


HEADER_KEYWORDS = {"invoice", "dated", "date", "bill no", "inv no", "inv#", "invoice no"}
PARTY_KEYWORDS = {
    "buyer",
    "customer",
    "bill to",
    "ship to",
    "supplier",
    "seller",
    "gstin",
    "gst",
    "address",
    "contact",
    "e-mail",
    "email",
    "phone",
}
PRODUCT_KEYWORDS = {"qty", "quantity", "rate", "amount", "hsn", "sac", "description", "goods"}
EXPENSE_KEYWORDS = {
    "comm",
    "labour",
    "commission",
    "commision",
    "fee",
    "charge",
    "freight",
    "fright",
    "hrdf",
    "loading",
    "market",
    "transport",
    "cartage",
    "handling",
}
TOTAL_KEYWORDS = {"total", "amount chargeable", "grand total", "net amount", "invoice value"}

STOP_WORDS = {
    "qty",
    "quantity",
    "rate",
    "amount",
    "hsn",
    "sac",
    "per",
    "no",
    "of",
    "pkgs",
    "taxable",
    "value",
    "total",
}
NON_PRODUCT_HINTS = {
    "invoice",
    "buyer",
    "supplier",
    "gstin",
    "state name",
    "contact",
    "email",
    "despatch",
    "delivery",
    "order no",
    "vehicle",
    "declaration",
    "signature",
}
UNIT_PATTERN = re.compile(r"\b(qt|kg|kgs|mt|ton|tons|nos|pcs|piece|box|bag|ltr|litre|unit|doz)\b", re.I)
HSN_PATTERN = re.compile(r"\b\d{4,8}\b")


@dataclass(frozen=True)
class NormalizedRow:
    index: int
    y: Optional[int]
    text: str
    section: str


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ")
    text = text.replace("|", " ")
    text = text.replace("•", " ")
    text = re.sub(r"[^\S\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_amount(value: str) -> Optional[float]:
    text = clean_text(value)
    if not text:
        return None
    text = text.replace(",", "")
    text = re.sub(r"[^0-9.\-]", "", text)
    if text.count(".") > 1:
        left, *rest = text.split(".")
        text = left + "." + "".join(rest)
    if text in {"", "-", ".", "-."}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def extract_amounts_with_pos(text: str) -> List[Tuple[float, int, int]]:
    values: List[Tuple[float, int, int]] = []
    for match in NUMBER_PATTERN.finditer(text):
        parsed = parse_amount(match.group(1))
        if parsed is not None:
            values.append((parsed, match.start(), match.end()))
    return values


def normalize_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = clean_text(value).replace(".", "-").replace("/", "-")
    text = re.sub(r"\s+", " ", text)
    formats = [
        "%d-%m-%Y",
        "%d-%m-%y",
        "%m-%d-%Y",
        "%m-%d-%y",
        "%d %b %Y",
        "%d %b %y",
        "%d %B %Y",
        "%d %B %y",
        "%d-%b-%Y",
        "%d-%b-%y",
        "%d-%B-%Y",
        "%d-%B-%y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    match = DATE_PATTERN.search(text)
    if not match:
        return None
    part = clean_text(match.group(1)).replace("/", "-").replace(".", "-")
    part = re.sub(r"\s+", " ", part)
    for fmt in formats:
        try:
            return datetime.strptime(part, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def safe_get(dct: Dict[str, Any], *keys: str) -> Optional[Any]:
    for key in keys:
        if key in dct and dct[key] not in (None, ""):
            return dct[key]
    return None


class InvoicePreprocessor:
    def preprocess(self, extracted_json: Optional[Dict[str, Any]], extracted_text: str) -> Dict[str, Any]:
        payload = extracted_json or {}
        rows = self._normalize_rows(payload)
        if not rows:
            rows = self._rows_from_raw_text(extracted_text)

        header = self._extract_header(payload, rows, extracted_text)
        parties = self._extract_parties(payload, rows, extracted_text)
        products, product_rows = self._extract_products(rows)
        expenses = self._extract_expenses(rows, product_rows)
        totals = self._extract_totals(payload, rows, products, expenses)

        if not products:
            fallback_amount = totals.get("Taxable_Amount")
            products = [
                {
                    "productName": "Unknown Item",
                    "quantity": None,
                    "unit": None,
                    "rate": None,
                    "amount": fallback_amount,
                    "HSN": None,
                }
            ]

        if not expenses:
            expenses = [{"name": "Unspecified Expense", "amount": 0.0, "percentage": None}]

        return {
            "header": header,
            "parties": parties,
            "products": products,
            "expenses": expenses,
            "totals": totals,
            "_ocr_rows": extracted_json.get("rows", []) if isinstance(extracted_json, dict) else [],
            "_ocr_layout": extracted_json.get("layout", []) if isinstance(extracted_json, dict) else [],
        }

    def _normalize_rows(self, payload: Dict[str, Any]) -> List[NormalizedRow]:
        input_rows = payload.get("rows")
        if not isinstance(input_rows, list):
            return []

        normalized: List[NormalizedRow] = []
        for idx, row in enumerate(input_rows, start=1):
            if not isinstance(row, dict):
                continue

            row_text = clean_text(row.get("row_text"))
            if not row_text:
                cells = row.get("cells", [])
                if isinstance(cells, list):
                    parts: List[str] = []
                    for cell in cells:
                        if isinstance(cell, dict):
                            parts.append(clean_text(cell.get("text")))
                        else:
                            parts.append(clean_text(cell))
                    row_text = clean_text(" ".join(part for part in parts if part))

            if not row_text:
                continue

            section = self._classify_row(row_text)
            y_val = row.get("y")
            y_int = int(y_val) if isinstance(y_val, (int, float)) else None
            normalized.append(NormalizedRow(index=idx, y=y_int, text=row_text, section=section))

        deduped: List[NormalizedRow] = []
        seen = set()
        for row in normalized:
            key = (row.text.lower(), row.y)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped

    def _rows_from_raw_text(self, extracted_text: str) -> List[NormalizedRow]:
        lines = [clean_text(line) for line in (extracted_text or "").splitlines()]
        rows: List[NormalizedRow] = []
        for idx, line in enumerate(lines, start=1):
            if not line:
                continue
            rows.append(NormalizedRow(index=idx, y=None, text=line, section=self._classify_row(line)))
        return rows

    def _classify_row(self, row_text: str) -> str:
        text = row_text.lower()
        score = {
            SECTION_HEADER: self._keyword_score(text, HEADER_KEYWORDS),
            SECTION_PARTY: self._keyword_score(text, PARTY_KEYWORDS),
            SECTION_PRODUCT: self._keyword_score(text, PRODUCT_KEYWORDS),
            SECTION_EXPENSE: self._keyword_score(text, EXPENSE_KEYWORDS),
            SECTION_TOTAL: self._keyword_score(text, TOTAL_KEYWORDS),
        }

        if score[SECTION_TOTAL] > 0:
            return SECTION_TOTAL
        if score[SECTION_EXPENSE] > 0 and "amount chargeable" not in text:
            return SECTION_EXPENSE
        if score[SECTION_PARTY] > 0 and score[SECTION_PRODUCT] == 0:
            return SECTION_PARTY

        amount_count = len(extract_amounts_with_pos(text))
        if score[SECTION_PRODUCT] > 0 or amount_count >= 3:
            if score[SECTION_EXPENSE] == 0:
                return SECTION_PRODUCT

        if score[SECTION_HEADER] > 0:
            return SECTION_HEADER
        return SECTION_OTHER

    @staticmethod
    def _keyword_score(text: str, keywords: Iterable[str]) -> int:
        return sum(1 for key in keywords if key in text)

    def _extract_header(
        self,
        payload: Dict[str, Any],
        rows: Sequence[NormalizedRow],
        extracted_text: str,
    ) -> Dict[str, Optional[str]]:
        normalized_fields = payload.get("normalized_fields", {}) if isinstance(payload.get("normalized_fields"), dict) else {}
        key_values = self._flatten_key_values(payload)
        all_text = "\n".join([row.text for row in rows[:25]]) + "\n" + (extracted_text or "")

        invoice_number = self._first_non_empty(
            safe_get(normalized_fields, "invoice_no", "Invoice_Number"),
            safe_get(key_values, "invoice_no", "Invoice Number", "Invoice No"),
            self._extract_invoice_number_from_text(all_text),
        )
        invoice_date = self._first_non_empty(
            safe_get(normalized_fields, "date", "Invoice_Date"),
            safe_get(key_values, "date", "invoice date", "dated"),
            self._extract_date_from_text(all_text),
        )

        return {
            "Invoice_Number": clean_text(invoice_number) or None,
            "Invoice_Date": normalize_date(invoice_date),
        }

    def _extract_parties(
        self,
        payload: Dict[str, Any],
        rows: Sequence[NormalizedRow],
        extracted_text: str,
    ) -> Dict[str, Any]:
        normalized_fields = payload.get("normalized_fields", {}) if isinstance(payload.get("normalized_fields"), dict) else {}
        key_values = self._flatten_key_values(payload)
        full_text = "\n".join(row.text for row in rows) + "\n" + (extracted_text or "")

        supplier_name = self._extract_labeled_value(rows, ["supplier", "seller", "from"])
        customer_name = self._extract_labeled_value(rows, ["buyer", "customer", "bill to", "ship to"])

        gstins = list(dict.fromkeys(GSTIN_PATTERN.findall(full_text.upper())))
        positional_supplier_gst, positional_customer_gst = self._gstins_by_row_order(rows)
        supplier_gst = self._extract_gstin_near_label(rows, ["supplier", "seller"]) or positional_supplier_gst or (gstins[0] if gstins else None)
        customer_gst = self._extract_gstin_near_label(rows, ["buyer", "customer", "bill to"]) or positional_customer_gst or (gstins[1] if len(gstins) > 1 else None)

        parties = {
            "SupplierName": self._first_non_empty(supplier_name, self._extract_supplier_name(rows), safe_get(key_values, "seller_name", "supplier"), safe_get(normalized_fields, "seller_name", "SupplierName")),
            "Customer_Name": self._first_non_empty(customer_name, self._extract_customer_name(rows), safe_get(key_values, "buyer_name", "buyer", "customer"), safe_get(normalized_fields, "buyer_name", "Customer_Name")),
            "Supplier_GST": supplier_gst,
            "Customer_GSTIN": customer_gst or (gstins[1] if len(gstins) > 1 else None),
            "Supplier_Address": self._extract_party_address(rows, ["supplier", "seller", "from"], ["buyer", "customer", "bill to", "ship to", "consignee"]),
            "Customer_address": self._extract_party_address(rows, ["buyer", "customer", "bill to", "ship to", "consignee"], ["description", "hsn", "qty", "quantity", "total"]),
            "Address": self._extract_address(rows),
            "Email": self._first_match(EMAIL_PATTERN, full_text),
            "Phone": self._extract_phone_list(full_text),
            "Vehicle_Number": self._first_match(VEHICLE_PATTERN, full_text.upper()),
        }
        for key, value in list(parties.items()):
            if isinstance(value, str):
                parties[key] = clean_text(value) or None
            elif value is None:
                parties[key] = None
        return parties

    def _extract_products(self, rows: Sequence[NormalizedRow]) -> Tuple[List[Dict[str, Any]], List[int]]:
        products: List[Dict[str, Any]] = []
        product_rows: List[int] = []
        table_start = self._find_product_table_start(rows)
        total_index = self._last_total_row_index(rows)

        if table_start is None:
            table_start = 0
        if total_index is None:
            total_index = len(rows)

        carry_name: Optional[str] = None
        expense_started = False
        for idx in range(table_start, total_index):
            row = rows[idx]
            text = row.text
            lower = text.lower()

            if any(hint in lower for hint in NON_PRODUCT_HINTS):
                continue
            if any(word in lower for word in EXPENSE_KEYWORDS) and products:
                expense_started = True
            if expense_started:
                continue
            if self._classify_row(text) == SECTION_EXPENSE:
                continue
            amounts = extract_amounts_with_pos(text)
            if not amounts:
                name_candidate = self._clean_product_name(text)
                if name_candidate and len(name_candidate) > 2 and self._classify_row(text) != SECTION_TOTAL:
                    carry_name = name_candidate
                continue
            if len(amounts) < 2:
                continue
            numeric_values = [val for val, _, _ in amounts]
            if len(amounts) >= 2 and max(numeric_values) < 50 and not UNIT_PATTERN.search(text):
                continue
            if any(word in lower for word in EXPENSE_KEYWORDS):
                continue
            if "total" in lower and "qty" not in lower:
                continue
            if idx + 1 < total_index:
                next_text = rows[idx + 1].text
                next_amounts = [val for val, _, _ in extract_amounts_with_pos(next_text)]
                if (
                    len(amounts) == 3
                    and not UNIT_PATTERN.search(text)
                    and len(next_amounts) >= 3
                    and UNIT_PATTERN.search(next_text)
                    and max(next_amounts) > 10000
                ):
                    carry_name = self._clean_product_name(text)
                    continue

            parsed = self._parse_product_row(text, carry_name)
            if parsed:
                products.append(parsed)
                product_rows.append(idx)
                carry_name = None

        deduped: List[Dict[str, Any]] = []
        seen = set()
        for product in products:
            key = (product.get("productName"), product.get("quantity"), product.get("rate"), product.get("amount"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(product)

        return deduped, product_rows

    def _parse_product_row(self, text: str, carry_name: Optional[str]) -> Optional[Dict[str, Any]]:
        amounts = [val for val, _, _ in extract_amounts_with_pos(text)]
        if len(amounts) < 2:
            return None

        quantity: Optional[float] = None
        rate: Optional[float] = None
        amount: Optional[float] = None

        if len(amounts) >= 3:
            quantity = amounts[0]
            rate = amounts[1]
            amount = amounts[-1]
        elif len(amounts) == 2:
            quantity = amounts[0]
            amount = amounts[1]

        hsn_match = HSN_PATTERN.search(text)
        hsn = hsn_match.group(0) if hsn_match else None

        unit_match = UNIT_PATTERN.search(text)
        unit = unit_match.group(1).lower() if unit_match else None

        name = self._clean_product_name(text)
        if carry_name and name and carry_name.lower() not in name.lower():
            name = f"{carry_name} {name}".strip()
        elif carry_name and not name:
            name = carry_name
        if not name:
            name = "Unknown Item"

        return {
            "productName": name,
            "quantity": quantity,
            "unit": unit,
            "rate": rate,
            "amount": amount,
            "HSN": hsn,
        }

    def _extract_expenses(self, rows: Sequence[NormalizedRow], product_rows: Sequence[int]) -> List[Dict[str, Any]]:
        expenses: List[Dict[str, Any]] = []
        last_total = self._last_total_row_index(rows) or len(rows)
        start = (max(product_rows) + 1) if product_rows else 0

        section_rows = rows[start:last_total]
        pending_label: Optional[str] = None

        for row in section_rows:
            text = row.text
            lower = text.lower()
            amount_matches = extract_amounts_with_pos(text)
            percent_match = PERCENT_PATTERN.search(text)
            has_expense_hint = any(key in lower for key in EXPENSE_KEYWORDS)

            if has_expense_hint and not amount_matches:
                pending_label = self._clean_expense_label(text)
                continue

            if pending_label and amount_matches and not has_expense_hint:
                amount = max((item[0] for item in amount_matches), key=abs)
                expenses.append(
                    {
                        "name": pending_label,
                        "amount": amount,
                        "percentage": float(percent_match.group(1)) if percent_match else None,
                    }
                )
                pending_label = None
                continue

            if has_expense_hint and amount_matches:
                if percent_match and len(amount_matches) == 1:
                    pending_label = self._clean_expense_label(text)
                    continue
                labels = self._split_expense_labels(text)
                mapped = self._map_amounts_to_labels(text, labels, amount_matches)
                for label, amount in mapped:
                    expenses.append(
                        {
                            "name": self._clean_expense_label(label),
                            "amount": amount,
                            "percentage": self._extract_percentage_for_label(text, label),
                        }
                    )
                pending_label = None

        deduped: List[Dict[str, Any]] = []
        seen = set()
        for expense in expenses:
            name = clean_text(expense.get("name"))
            amount = expense.get("amount")
            key = (name.lower(), amount)
            if key in seen or not name:
                continue
            seen.add(key)
            deduped.append({"name": name, "amount": amount, "percentage": expense.get("percentage")})
        return deduped

    def _map_amounts_to_labels(
        self,
        text: str,
        labels: Sequence[Tuple[str, int, int]],
        amounts: Sequence[Tuple[float, int, int]],
    ) -> List[Tuple[str, float]]:
        if not labels:
            return []
        if len(labels) == 1:
            return [(labels[0][0], max(amounts, key=lambda item: abs(item[0]))[0])]

        label_ranges = [(label, start, end) for label, start, end in labels]
        amount_ranges = [(value, start, end) for value, start, end in amounts]

        all_labels_before_amounts = max(end for _, _, end in label_ranges) < min(start for _, start, _ in amount_ranges)
        if all_labels_before_amounts and len(label_ranges) == len(amount_ranges):
            # OCR commonly prints labels left-to-right and amounts right-aligned. Reverse map is often correct.
            return [(label_ranges[-1 - idx][0], amount_ranges[idx][0]) for idx in range(len(amount_ranges))]

        mapped: List[Tuple[str, float]] = []
        unused_labels = list(range(len(label_ranges)))
        for value, start, end in amount_ranges:
            center = (start + end) / 2.0
            best_label_idx = min(
                unused_labels,
                key=lambda i: abs(center - ((label_ranges[i][1] + label_ranges[i][2]) / 2.0)),
            )
            mapped.append((label_ranges[best_label_idx][0], value))
            if len(unused_labels) > 1:
                unused_labels.remove(best_label_idx)
        return mapped

    def _split_expense_labels(self, text: str) -> List[Tuple[str, int, int]]:
        matches: List[Tuple[str, int, int]] = []
        lower = text.lower()
        for keyword in sorted(EXPENSE_KEYWORDS, key=len, reverse=True):
            for match in re.finditer(rf"\b{re.escape(keyword)}\b", lower):
                left = max(0, match.start() - 15)
                right = min(len(text), match.end() + 20)
                snippet = clean_text(text[left:right])
                if snippet:
                    matches.append((snippet, left, right))
        if not matches:
            return []

        deduped: List[Tuple[str, int, int]] = []
        seen = set()
        for label, start, end in sorted(matches, key=lambda m: (m[1], m[2])):
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append((label, start, end))
        return deduped

    def _extract_percentage_for_label(self, text: str, label: str) -> Optional[float]:
        label_pos = text.lower().find(label.lower())
        if label_pos == -1:
            match = PERCENT_PATTERN.search(text)
            return float(match.group(1)) if match else None
        nearby = text[label_pos : min(len(text), label_pos + 40)]
        match = PERCENT_PATTERN.search(nearby)
        if match:
            return float(match.group(1))
        return None

    def _gstins_by_row_order(self, rows: Sequence[NormalizedRow]) -> Tuple[Optional[str], Optional[str]]:
        ordered: List[str] = []
        for row in rows:
            for gst in GSTIN_PATTERN.findall(row.text.upper()):
                if gst not in ordered:
                    ordered.append(gst)
        supplier = ordered[0] if ordered else None
        customer = ordered[1] if len(ordered) > 1 else None
        return supplier, customer

    def _extract_totals(
        self,
        payload: Dict[str, Any],
        rows: Sequence[NormalizedRow],
        products: Sequence[Dict[str, Any]],
        expenses: Sequence[Dict[str, Any]],
    ) -> Dict[str, Optional[float]]:
        normalized_fields = payload.get("normalized_fields", {}) if isinstance(payload.get("normalized_fields"), dict) else {}
        last_total_idx = self._last_total_row_index(rows)

        candidate_amounts: List[float] = []
        if last_total_idx is not None:
            for idx in range(last_total_idx, min(len(rows), last_total_idx + 3)):
                candidate_amounts.extend([value for value, _, _ in extract_amounts_with_pos(rows[idx].text)])

        if candidate_amounts:
            total_amount = max(candidate_amounts)
        else:
            total_amount = parse_amount(str(safe_get(normalized_fields, "total_amount", "net_amount")))

        product_amounts = [p.get("amount") for p in products if isinstance(p.get("amount"), (int, float))]
        taxable_amount = max(product_amounts) if product_amounts else None
        if taxable_amount is None:
            taxable_amount = parse_amount(str(safe_get(normalized_fields, "taxable_amount")))

        total_expense = sum(float(e.get("amount") or 0.0) for e in expenses)
        return {
            "Taxable_Amount": taxable_amount,
            "Total_Amount": total_amount,
            "Total_Expenses": total_expense,
        }

    def _find_product_table_start(self, rows: Sequence[NormalizedRow]) -> Optional[int]:
        header_idx: Optional[int] = None
        for idx, row in enumerate(rows):
            lower = row.text.lower()
            if ("description" in lower or "goods" in lower) and ("qty" in lower or "quantity" in lower):
                return idx + 1
            if "hsn" in lower and ("qty" in lower or "amount" in lower):
                return idx + 1
            if ("description" in lower or "goods" in lower or "hsn" in lower) and header_idx is None:
                header_idx = idx
            if header_idx is not None and idx <= header_idx + 2 and ("qty" in lower or "quantity" in lower) and ("rate" in lower or "amount" in lower):
                return idx + 1

        for idx, row in enumerate(rows):
            lower = row.text.lower()
            amounts = extract_amounts_with_pos(lower)
            if len(amounts) >= 3 and (UNIT_PATTERN.search(lower) or "hsn" in lower):
                return idx
        return None

    def _last_total_row_index(self, rows: Sequence[NormalizedRow]) -> Optional[int]:
        indexes = []
        for idx, row in enumerate(rows):
            lower = row.text.lower()
            if "total" in lower or "amount chargeable" in lower:
                indexes.append(idx)
        return indexes[-1] if indexes else None

    def _clean_product_name(self, text: str) -> str:
        cleaned = text
        cleaned = NUMBER_PATTERN.sub(" ", cleaned)
        cleaned = PERCENT_PATTERN.sub(" ", cleaned)
        cleaned = UNIT_PATTERN.sub(" ", cleaned)
        cleaned = re.sub(r"\b[A-Z]{1,3}\d{2,}\b", " ", cleaned)
        tokens = [tok for tok in re.split(r"[^A-Za-z0-9&/+\-]+", cleaned) if tok]
        useful = [tok for tok in tokens if tok.lower() not in STOP_WORDS and not tok.isdigit()]
        return clean_text(" ".join(useful))

    def _clean_expense_label(self, text: str) -> str:
        cleaned = NUMBER_PATTERN.sub(" ", text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return clean_text(cleaned)

    def _flatten_key_values(self, payload: Dict[str, Any]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        pages = payload.get("pages")
        if isinstance(pages, list):
            for page in pages:
                if isinstance(page, dict) and isinstance(page.get("key_values"), dict):
                    for key, val in page["key_values"].items():
                        k = clean_text(key).lower()
                        v = clean_text(val)
                        if k and v and (k not in out or len(v) > len(out[k])):
                            out[k] = v
        return out

    def _extract_invoice_number_from_text(self, text: str) -> Optional[str]:
        for pattern in INVOICE_NUMBER_PATTERNS:
            match = pattern.search(text)
            if match:
                return clean_text(match.group(1))
        return None

    def _extract_date_from_text(self, text: str) -> Optional[str]:
        match = DATE_PATTERN.search(text)
        return clean_text(match.group(1)) if match else None

    def _extract_labeled_value(self, rows: Sequence[NormalizedRow], labels: Sequence[str]) -> Optional[str]:
        for idx, row in enumerate(rows[:40]):
            lower = row.text.lower()
            for label in labels:
                if not re.search(rf"\b{re.escape(label)}\b", lower):
                    continue
                if "supplier" in label and "ref" in lower:
                    continue
                pattern = re.compile(rf"\b{re.escape(label)}\b\s*[:\-]?\s*(.+)$", re.I)
                match = pattern.search(row.text)
                if match:
                    value = clean_text(match.group(1))
                    if value and len(value) > 2 and "order no" not in value.lower():
                        return value
                if idx + 1 < len(rows):
                    next_row_text = clean_text(rows[idx + 1].text)
                    if next_row_text and self._classify_row(next_row_text) in {SECTION_PARTY, SECTION_OTHER}:
                        return next_row_text
        return None

    def _extract_supplier_name(self, rows: Sequence[NormalizedRow]) -> Optional[str]:
        for row in rows[:12]:
            lower = row.text.lower()
            if "invoice no" in lower or "dated" in lower:
                left = re.split(r"invoice\s*no\.?|dated", row.text, flags=re.I)[0]
                candidate = clean_text(left)
                if candidate and len(candidate) > 3:
                    return candidate
        for row in rows[:10]:
            lower = row.text.lower()
            if any(token in lower for token in {"invoice", "note", "gstin", "buyer", "date"}):
                continue
            if len(re.findall(r"[A-Za-z]", row.text)) >= 6:
                return clean_text(row.text)
        return None

    def _extract_customer_name(self, rows: Sequence[NormalizedRow]) -> Optional[str]:
        for idx, row in enumerate(rows[:35]):
            lower = row.text.lower()
            if re.search(r"\bbuyer\b|\bcustomer\b|\bbill to\b", lower):
                for scan in range(idx + 1, min(len(rows), idx + 5)):
                    text = clean_text(rows[scan].text)
                    low = text.lower()
                    if any(x in low for x in {"order no", "dated", "gstin", "state name", "destination"}):
                        continue
                    if len(re.findall(r"[A-Za-z]", text)) >= 4:
                        return text
        return None

    def _extract_gstin_near_label(self, rows: Sequence[NormalizedRow], labels: Sequence[str]) -> Optional[str]:
        for idx, row in enumerate(rows[:50]):
            lower = row.text.lower()
            if not any(label in lower for label in labels):
                continue
            if "supplier" in " ".join(labels).lower() and "ref" in lower:
                continue
            span_text = " ".join(r.text for r in rows[idx : min(len(rows), idx + 3)])
            match = GSTIN_PATTERN.search(span_text.upper())
            if match:
                return match.group(0)
        return None

    def _extract_address(self, rows: Sequence[NormalizedRow]) -> Optional[str]:
        parts: List[str] = []
        for row in rows[:25]:
            text = row.text
            lower = text.lower()
            if "address" in lower or ("state" in lower and "code" in lower):
                parts.append(text)
            elif parts and row.section in {SECTION_PARTY, SECTION_OTHER} and len(text) > 8:
                parts.append(text)
                if len(parts) >= 3:
                    break
        value = clean_text(" | ".join(parts))
        return value or None

    def _extract_party_address(
        self,
        rows: Sequence[NormalizedRow],
        anchor_labels: Sequence[str],
        stop_labels: Sequence[str],
    ) -> Optional[str]:
        start_idx: Optional[int] = None
        for idx, row in enumerate(rows[:60]):
            low = row.text.lower()
            if any(label in low for label in anchor_labels):
                start_idx = idx
                break
        if start_idx is None:
            return None

        parts: List[str] = []
        for idx in range(start_idx + 1, min(len(rows), start_idx + 8)):
            text = clean_text(rows[idx].text)
            low = text.lower()
            if any(stop in low for stop in stop_labels):
                break
            if any(k in low for k in {"gstin", "date", "invoice", "order no", "delivery note", "state name"}):
                continue
            if len(text) < 4:
                continue
            if re.fullmatch(r"\d+", text):
                continue
            parts.append(text)

        value = clean_text(" | ".join(parts[:3]))
        return value or None

    def _extract_phone_list(self, text: str) -> Optional[str]:
        phones = list(dict.fromkeys(PHONE_PATTERN.findall(text)))
        if not phones:
            return None
        return " | ".join(phones)

    @staticmethod
    def _first_non_empty(*values: Optional[Any]) -> Optional[str]:
        for value in values:
            text = clean_text(value)
            if text and text.lower() not in {"null", "none", "na", "n/a"}:
                return text
        return None

    @staticmethod
    def _first_match(pattern: re.Pattern[str], text: str) -> Optional[str]:
        match = pattern.search(text or "")
        return match.group(0) if match else None


def preprocess_invoice_data(extracted_json: Optional[Dict[str, Any]], extracted_text: str) -> Dict[str, Any]:
    """
    Deterministic preprocessing entrypoint for OCR invoice data.
    """
    return InvoicePreprocessor().preprocess(extracted_json=extracted_json, extracted_text=extracted_text)
