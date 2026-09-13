# Contract Analysis AI

CM3070 Final Project &mdash; an AI-orchestrated system that helps employees understand their
employment contracts in plain English, built on the CM3020 Artificial Intelligence template
1: *Orchestrating AI Models to Achieve a Goal*.

This README covers how to run the code.

## Repository structure

```
finalProject/
├── report/
│   └── Contract_Analysis_AI_Report.docx   # Full written report
├── notebooks/
│   ├── Model1_PDF_Extraction.ipynb        # PDF text extraction + clause chunking (+ OCR fallback)
│   ├── Model2_LegalBERT_FineTuning.ipynb  # Fine-tuning LEGAL-BERT on the CUAD employment subset
│   ├── Model3_Explanation_Generation.ipynb# Plain-English explanation generation via Claude
│   └── Pipeline_End_to_End.ipynb          # All three models chained together
└── webapp/
    ├── app.py                             # Flask backend (real, runnable web interface)
    ├── requirements.txt
    ├── templates/
    └── static/
```

## Running the notebooks

Each notebook is self-contained and designed for Google Colab.

1. Open [Google Colab](https://colab.research.google.com)
2. File &rarr; Upload notebook &rarr; select the `.ipynb` file from `notebooks/`
3. Run cells top to bottom (`Runtime` &rarr; `Run all`)
4. `Model1_PDF_Extraction.ipynb` and `Pipeline_End_to_End.ipynb` will prompt for a contract PDF
   upload &mdash; skip the upload to fall back to a generated sample contract
5. `Model3_Explanation_Generation.ipynb` and `Pipeline_End_to_End.ipynb` will prompt for an
   Anthropic API key &mdash; leave it blank to run in a deterministic mock mode instead
6. `Model2_LegalBERT_FineTuning.ipynb` needs a GPU runtime (`Runtime` &rarr; `Change runtime
   type` &rarr; GPU) to fine-tune in a reasonable time

## Running the web app locally

The web app is the real, working user interface: upload a contract PDF, get a plain-English,
risk-flagged report back. It uses a Flask backend so the Anthropic API key stays server-side
and is never exposed to whoever is using the site.

```bash
cd webapp
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in a browser.

### Deploying it publicly (optional)

The app is a standard Flask app and deploys as-is to any platform that runs Python (Render,
Railway, Fly.io, etc.). Set the `ANTHROPIC_API_KEY` environment variable in the platform's
dashboard (never commit it to the repo) and set the start command to:

```
gunicorn app:app
```

## System architecture

```
Contract PDF
     |
     v
+-------------+     +----------------------+     +--------------------------+
|   Model 1   | --> |       Model 2        | --> |         Model 3          |
|  Document   |     |  Clause              |     |  Plain-English           |
|  Processing |     |  Classification      |     |  Explanation             |
|             |     |                      |     |                          |
| pdfplumber, |     | Zero-shot DeBERTa    |     | Anthropic Claude API,    |
| heading     |     | NLI (current) /      |     | structured JSON prompt,  |
| detection,  |     | fine-tuned LEGAL-BERT|     | retry + fallback logic   |
| OCR fallback|     | (planned upgrade)    |     |                          |
+-------------+     +----------------------+     +--------------------------+
```

The web app in `webapp/` is a simplified, deployable version of this same pipeline, built for
demonstration: it combines classification and explanation into a single Claude call per clause
for a fast, fully server-rendered experience, rather than running the separate LegalBERT
classifier stage. See the report's Design and Implementation chapters for the full, validated
research pipeline this is based on.

## Disclaimer

This tool is for informational purposes only and does not constitute legal advice.
