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
# Model 3's API key. Held ONLY here, as a server-side environment
# variable -- never sent to, stored in, or visible from the browser.
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
# Pre-pipeline sanity check: does this document even look
# like an employment contract? Catches a wrong upload (a
# resume, an invoice, an unrelated PDF) before wasting any Model 2/3
# calls on it, instead of forcing meaningless output out of the pipeline.
# ------------------------------------------------------------------

EMPLOYMENT_CONTRACT_KEYWORDS = [
    "employ",  # matches employee, employer, employment, employed
    "salary", "remuneration", "wage",
    "terminat",  # matches terminate, terminated, termination
    "probation",  # matches probation, probationary
    "notice period", "duties", "job title", "position",
    "confidential",  # matches confidential, confidentiality
    "non-compete", "noncompete", "cpf", "annual leave", "sick leave",
    "commencement", "basic salary", "resign",  # matches resign, resignation
    "employment agreement", "contract of employment",
]


def looks_like_employment_contract(combined_text_lower, min_keyword_hits=3):
    hits = sum(1 for kw in EMPLOYMENT_CONTRACT_KEYWORDS if kw in combined_text_lower)
    return hits >= min_keyword_hits

CONFIDENCE_THRESHOLD = 0.35

# ------------------------------------------------------------------
# Model 2: Clause Classification (zero-shot DeBERTa NLI)
# Loaded once so the
# app still starts quickly and so importing this module for testing
# doesn't require downloading model weights.
# ------------------------------------------------------------------

_zero_shot_classifier = None


def get_classifier():
    """Lazily loads the zero-shot classification pipeline (same model
    used in Model2's zero-shot prototype notebook: cross-encoder/nli-deberta-v3-small).
    Downloaded from Hugging Face on first call and cached in memory afterward."""
    global _zero_shot_classifier
    if _zero_shot_classifier is None:
        from transformers import pipeline
        print("[Model 2] Loading zero-shot classifier (first request only, "
              "may take a moment)...")
        _zero_shot_classifier = pipeline(
            "zero-shot-classification", model="cross-encoder/nli-deberta-v3-small"
        )
    return _zero_shot_classifier


def classify_clause(clause_text):
    """
    Model 2: classifies a single clause into one of the six target types
    using zero-shot NLI, applying the same 0.35 confidence threshold
    documented in the project report. Returns (label, confidence) --
    label is 'unclassified' if no candidate clears the threshold.
    """
    classifier = get_classifier()
    result = classifier(clause_text[:2000], candidate_labels=TARGET_CLAUSE_TYPES)
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    if top_score < CONFIDENCE_THRESHOLD:
        return "unclassified", top_score
    return top_label, top_score

# ------------------------------------------------------------------
# Model 1: PDF extraction & clause chunking
# (ported directly from the validated Model1_PDF_Extraction.ipynb,
# including the OCR fallback -- Tesseract's LSTM engine is this
# pipeline's third pre-trained model, alongside Model 2's classifier
# and Model 3's LLM.)
# ------------------------------------------------------------------

def needs_ocr(pdf_path, min_chars_per_page=20):
    """Detects pages with no meaningful extractable text layer."""
    pages_needing_ocr = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if len(text.strip()) < min_chars_per_page:
                pages_needing_ocr.append(i)
    return pages_needing_ocr


def ocr_page_to_lines(image, page_num, dpi=200):
    """
    Runs Tesseract's LSTM OCR engine on a page image and groups the
    word-level output into lines with synthetic top/bottom positions,
    rescaled from pixel coordinates (at the given render DPI) into
    points so gaps are directly comparable to pdfplumber's native
    coordinate system regardless of render resolution.
    """
    import pytesseract
    from pytesseract import Output
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    lines = {}
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        if not word:
            continue
        key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
        top, height = data['top'][i], data['height'][i]
        if key not in lines:
            lines[key] = {'words': [], 'top': top, 'bottom': top + height}
        lines[key]['words'].append(word)
        lines[key]['bottom'] = max(lines[key]['bottom'], top + height)

    scale = 72.0 / dpi
    ordered = sorted(lines.values(), key=lambda l: l['top'])
    out, prev_bottom = [], None
    for l in ordered:
        top_pt, bottom_pt = l['top'] * scale, l['bottom'] * scale
        gap = (top_pt - prev_bottom) if prev_bottom is not None else None
        out.append({'page': page_num, 'text': ' '.join(l['words']), 'gap_before': gap})
        prev_bottom = bottom_pt
    return out


def extract_pages(pdf_path, ocr_dpi=200):
    all_lines = []
    ocr_needed_pages = set(needs_ocr(pdf_path))

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            if page_num in ocr_needed_pages:
                continue
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

    if ocr_needed_pages:
        from pdf2image import convert_from_path
        print(f"[Model 1] Pages with no extractable text layer -- running OCR: "
              f"{sorted(ocr_needed_pages)}")
        images = convert_from_path(pdf_path, dpi=ocr_dpi)
        for page_num in sorted(ocr_needed_pages):
            all_lines.extend(ocr_page_to_lines(images[page_num - 1], page_num, dpi=ocr_dpi))
        all_lines.sort(key=lambda l: l['page'])

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
# Model 3: Explanation Generation via the real Claude API, called
# server-side using the securely-held key. Model 3 does NOT decide the
# clause type -- that decision is Model 2's alone (classify_clause,
# above). Model 3 only explains a clause it has already been told the
# type of, and independently assesses risk level from the clause's
# actual text -- matching the pipeline documented in the project report.
# ------------------------------------------------------------------

PROMPT_TEMPLATE = """You are helping a non-lawyer employee in Singapore understand one clause \
from their employment contract.

Clause text: "{clause_text}"

This clause has already been classified as a "{predicted_label}" by a separate classification \
model. Do not reclassify it or question the label -- your job is only to explain it and assess \
its risk.

Respond with ONLY a valid JSON object, no other text, no markdown fences, with exactly these keys:
- "risk_level": "LOW", "MEDIUM", or "HIGH" -- based on the ACTUAL terms in this specific clause \
(durations, amounts, conditions), not a generic guess for the category
- "explanation": one or two plain-English sentences explaining what this specific clause means \
for the employee, referencing concrete details where present
- "watch_for": one short, concrete thing the employee should check or ask about, specific to \
what's actually written here
"""


def explain_clause(clause_text, predicted_label, max_retries=2):
    """
    Model 3: given a clause and the type Model 2 already assigned it,
    generates a plain-English risk assessment and explanation via Claude.
    Never called for 'unclassified' clauses -- see the /analyze route.
    """
    prompt = PROMPT_TEMPLATE.format(
        clause_text=clause_text[:1500].replace('"', "'"),
        predicted_label=predicted_label
    )
    raw = None
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.messages.create(
                model=MODEL_NAME, max_tokens=400,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = resp.content[0].text
            break
        except Exception as e:
            last_error = e
            print(f"[explain_clause] API call failed (attempt {attempt + 1}/"
                  f"{max_retries + 1}): {type(e).__name__}: {e}")
            if attempt < max_retries:
                time.sleep(1)

    fallback = {"risk_level": "ERROR",
                "explanation": "This clause was classified but couldn't be explained "
                                f"automatically due to a connection issue "
                                f"({type(last_error).__name__ if last_error else 'unknown'}).",
                "watch_for": "Review this clause manually."}
    if raw is None:
        return fallback
    try:
        cleaned = re.sub(r'^```(json)?|```$', '', raw.strip(), flags=re.M).strip()
        parsed = json.loads(cleaned)
        assert parsed.get("risk_level") and parsed.get("explanation") and parsed.get("watch_for")
        return parsed
    except Exception as e:
        print(f"[explain_clause] Response wasn't valid JSON: {e}\nRaw response: {raw[:300]}")
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
            error="No readable clause-length text was found in this PDF. It may be corrupted "
                  "or contain no meaningful content even after OCR.")

    combined_text = " ".join(c["clause_text"] for c in chunks).lower()
    if not looks_like_employment_contract(combined_text):
        return render_template(
            "index.html", server_ready=True,
            error="This doesn't look like an employment contract -- please check you've "
                  "uploaded the right file and try again.")

    for chunk in chunks:
        # Model 2: real classification, independent of Model 3
        label, confidence = classify_clause(chunk["clause_text"])
        chunk["predicted_label"] = label
        chunk["confidence"] = confidence

        if label == "unclassified":
            # Matches the documented pipeline: unclassified clauses bypass
            # Model 3 entirely rather than asking an LLM to explain a category
            # it was never confidently assigned to.
            chunk["risk_level"] = "N/A"
            chunk["explanation"] = ("This clause didn't clearly match a tracked category "
                                     "-- it may be administrative or procedural content.")
            chunk["watch_for"] = ("Skim this manually if it looks important; automatic risk "
                                   "assessment wasn't confident enough to be reliable here.")
        else:
            # Model 3: explanation only, given Model 2's label
            analysis = explain_clause(chunk["clause_text"], label)
            chunk.update(analysis)

    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "N/A": 3, "ERROR": 4}
    chunks.sort(key=lambda c: order.get(c.get("risk_level", "N/A"), 3))

    stats = {
        "total": len(chunks),
        "high": sum(1 for c in chunks if c.get("risk_level") == "HIGH"),
        "medium": sum(1 for c in chunks if c.get("risk_level") == "MEDIUM"),
        "low": sum(1 for c in chunks if c.get("risk_level") == "LOW"),
        "na": sum(1 for c in chunks if c.get("risk_level") == "N/A"),
        "error": sum(1 for c in chunks if c.get("risk_level") == "ERROR"),
    }
    return render_template("results.html", clauses=chunks, stats=stats,
                            filename=file.filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)