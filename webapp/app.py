import os
import re
import json
import time
import tempfile
from collections import defaultdict

from flask import Flask, request, render_template
from dotenv import load_dotenv

import pdfplumber

# Load variables from a local .env file, if one exists (never committed to
# git -- see .gitignore). This is what lets you set ANTHROPIC_API_KEY once
# in a file instead of retyping it into the terminal every session.
load_dotenv()

app = Flask(__name__)

# ------------------------------------------------------------------
# API key lives ONLY here, as a server-side environment variable.
# It is never sent to, stored in, or visible from the browser.
# ------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

client = None
if ANTHROPIC_API_KEY:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

MODEL_NAME = "claude-sonnet-4-6"

TARGET_CLAUSE_TYPES = [
    "compensation clause", "termination clause", "confidentiality clause",
    "non-compete clause", "intellectual property clause", "probation clause"
]

# ------------------------------------------------------------------
# Model 1: PDF extraction & clause chunking
# (ported directly from the validated Model1_PDF_Extraction.ipynb)
# ------------------------------------------------------------------

def extract_pages(pdf_path):
    all_lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                lines = page.extract_text_lines()
            except Exception:
                lines = []
            prev_bottom = None
            for l in lines:
                text = l.get('text', '').strip()
                if not text:
                    continue
                gap = (l['top'] - prev_bottom) if prev_bottom is not None else None
                all_lines.append({'page': page_num, 'text': text, 'gap_before': gap})
                prev_bottom = l['bottom']
    return all_lines


_STAMP_PATTERNS = [
    re.compile(r'Updated\s+on\s+\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', re.I),
    re.compile(r'^[A-Z]-\d+$'),
    re.compile(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\s+\d{1,2}:\d{2}(:\d{2})?$'),
]


def strip_boilerplate(all_lines, min_page_repeats=2):
    def norm(t):
        return re.sub(r'\s+', ' ', t.strip().lower())

    page_sets = defaultdict(set)
    for rec in all_lines:
        page_sets[norm(rec['text'])].add(rec['page'])
    repeated = {t for t, pages in page_sets.items() if len(pages) >= min_page_repeats}

    out = []
    for rec in all_lines:
        text = rec['text']
        if norm(text) in repeated:
            continue
        if any(p.search(text) for p in _STAMP_PATTERNS):
            continue
        out.append(rec)
    return out


def clean_line(text):
    text = re.sub(r'^\s*Page\s+\d+(\s+of\s+\d+)?\s*$', '', text)
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text)
    return text.strip()


HEADING_PATTERNS = [
    re.compile(r'^\s*(\d{1,2})\.?\s+([A-Z][A-Za-z0-9,/&\'\-\s]{2,80})\s*$'),
    re.compile(r'^\s*(ARTICLE|Article)\s+([IVXLCDM]+|\d+)\b[\s:.\-]*(.*)$'),
    re.compile(r'^\s*(SECTION|Section)\s+(\d+)\b[\s:.\-]*(.*)$'),
    re.compile(r'^\s*[A-Z][A-Z\s]{3,60}\s*$'),
]
_NON_HEADING_STARTERS = ('PLEASE ', 'NOTE ', 'NOTE:', 'WARNING', 'IMPORTANT',
                          'CAUTION', 'ATTENTION', 'DISCLAIMER')
_CLOSING_MARKERS = re.compile(r'\b(IN WITNESS WHEREOF|SIGNED AT|SIGNATURE[S]?\s*:?\s*$)\b', re.I)


def is_heading(line):
    line = line.strip()
    if not line or len(line) > 140:
        return False
    if _CLOSING_MARKERS.search(line):
        return True
    if len(line) > 100:
        return False
    if line.upper().startswith(_NON_HEADING_STARTERS):
        return False
    return any(pat.match(line) for pat in HEADING_PATTERNS)


def is_new_paragraph(line_record, prev_line_record, gap_threshold=10):
    if prev_line_record is None:
        return True
    if line_record['page'] != prev_line_record['page']:
        prev_text = prev_line_record['text'].strip()
        return prev_text[-1:] in '.!?:;' if prev_text else True
    gap = line_record['gap_before']
    return gap is not None and gap > gap_threshold


def is_placeholder_text(text, alpha_ratio_threshold=0.2):
    letters = sum(1 for c in text if c.isalpha())
    return len(text) > 0 and (letters / len(text)) < alpha_ratio_threshold


def is_likely_fragment(text, min_absolute_len=10):
    text = text.strip()
    if not text:
        return True
    if text[0].islower():
        return True
    if len(text) < min_absolute_len:
        return True
    return False


def chunk_into_clauses(all_lines, min_chunk_chars=30):
    cleaned = []
    for rec in all_lines:
        t = clean_line(rec['text'])
        if t:
            cleaned.append({**rec, 'text': t})

    heading_idx = [i for i, r in enumerate(cleaned) if is_heading(r['text'])]
    chunks = []

    def join_lines(records):
        parts = []
        for r in records:
            t = r['text']
            if parts and parts[-1].endswith('-'):
                parts[-1] = parts[-1][:-1] + t
            else:
                parts.append(t)
        return ' '.join(parts)

    if heading_idx:
        for idx, start in enumerate(heading_idx):
            end = heading_idx[idx + 1] if idx + 1 < len(heading_idx) else len(cleaned)
            heading_text = cleaned[start]['text']
            body_text = join_lines(cleaned[start + 1:end])
            page_num = cleaned[start]['page']
            if len(body_text) >= min_chunk_chars:
                chunks.append({'chunk_id': len(chunks) + 1, 'section_heading': heading_text,
                                'clause_text': body_text, 'page_number': page_num,
                                'char_count': len(body_text)})
    else:
        current, prev, groups = [], None, []
        for r in cleaned:
            if is_new_paragraph(r, prev) and current:
                groups.append(current)
                current = []
            current.append(r)
            prev = r
        if current:
            groups.append(current)
        for i, group in enumerate(groups, start=1):
            text = join_lines(group)
            if len(text) >= min_chunk_chars:
                chunks.append({'chunk_id': i, 'section_heading': None, 'clause_text': text,
                                'page_number': group[0]['page'], 'char_count': len(text)})

    chunks = [c for c in chunks if not is_placeholder_text(c['clause_text'])]

    merged = []
    for c in chunks:
        if merged and is_likely_fragment(c['clause_text']):
            merged[-1]['clause_text'] = merged[-1]['clause_text'] + ' ' + c['clause_text']
            merged[-1]['char_count'] = len(merged[-1]['clause_text'])
        else:
            merged.append(dict(c))
    for i, c in enumerate(merged, start=1):
        c['chunk_id'] = i
    return merged


# ------------------------------------------------------------------
# Model 2 + 3: classification + explanation via the real Claude API,
# called server-side using the securely-held key.
# ------------------------------------------------------------------

PROMPT_TEMPLATE = """You are helping a non-lawyer employee in Singapore understand one clause \
from their employment contract.

Clause text: "{clause_text}"

Decide which ONE of these six categories the clause best fits: {types}. If it genuinely does \
not fit any of them (e.g. it's administrative or procedural, like a commencement date, working \
hours, or a signature block), use "unclassified" instead of forcing a guess.

Respond with ONLY a valid JSON object, no other text, no markdown fences, with exactly these keys:
- "predicted_label": one of the six category strings above, or "unclassified"
- "risk_level": "LOW", "MEDIUM", "HIGH", or "N/A" if unclassified -- based on the ACTUAL terms \
in this specific clause, not a generic guess for the category
- "explanation": one or two plain-English sentences explaining what this specific clause means \
for the employee, referencing concrete details (durations, amounts, conditions) where present
- "watch_for": one short, concrete thing the employee should check or ask about, specific to \
what's actually written here
"""


def classify_and_explain(clause_text, max_retries=2):
    prompt = PROMPT_TEMPLATE.format(
        clause_text=clause_text[:1500].replace('"', "'"),
        types=", ".join(TARGET_CLAUSE_TYPES)
    )
    raw = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.messages.create(
                model=MODEL_NAME, max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = resp.content[0].text
            break
        except Exception:
            if attempt == max_retries:
                raw = None
            else:
                time.sleep(1)

    fallback = {"predicted_label": "unclassified", "risk_level": "N/A",
                "explanation": "This clause couldn't be analysed automatically due to a "
                                "connection issue.",
                "watch_for": "Review this clause manually."}
    if raw is None:
        return fallback
    try:
        cleaned = re.sub(r'^```(json)?|```$', '', raw.strip(), flags=re.M).strip()
        parsed = json.loads(cleaned)
        assert parsed.get("predicted_label") and parsed.get("risk_level")
        assert parsed.get("explanation") and parsed.get("watch_for")
        return parsed
    except Exception:
        return fallback


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", server_ready=client is not None)


@app.route("/analyze", methods=["POST"])
def analyze():
    if client is None:
        return render_template(
            "index.html", server_ready=False,
            error="This server isn't configured with an API key yet. "
                  "(Set the ANTHROPIC_API_KEY environment variable.)")

    file = request.files.get("contract")
    if not file or not file.filename.lower().endswith(".pdf"):
        return render_template("index.html", server_ready=True,
                                error="Please upload a PDF file.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        lines = extract_pages(tmp_path)
        lines = strip_boilerplate(lines)
        chunks = chunk_into_clauses(lines)
    finally:
        os.remove(tmp_path)

    if not chunks:
        return render_template(
            "index.html", server_ready=True,
            error="No readable clause-length text was found in this PDF. If it's a scanned "
                  "document, this deployment doesn't include OCR (the project's notebooks do).")

    for chunk in chunks:
        analysis = classify_and_explain(chunk["clause_text"])
        chunk.update(analysis)

    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "N/A": 3}
    chunks.sort(key=lambda c: order.get(c.get("risk_level", "N/A"), 3))

    stats = {
        "total": len(chunks),
        "high": sum(1 for c in chunks if c.get("risk_level") == "HIGH"),
        "medium": sum(1 for c in chunks if c.get("risk_level") == "MEDIUM"),
        "low": sum(1 for c in chunks if c.get("risk_level") == "LOW"),
        "na": sum(1 for c in chunks if c.get("risk_level") == "N/A"),
    }
    return render_template("results.html", clauses=chunks, stats=stats,
                            filename=file.filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
