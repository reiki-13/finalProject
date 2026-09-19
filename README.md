# Contract Analysis AI

CM3070 Final Project: An AI-orchestrated system that helps employees understand their
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
│   ├── Model3_Explanation_Generation.ipynb# Plain-English explanation generation via Groq
│   └── Pipeline_End_to_End.ipynb          # All three models chained together
└── webapp/
    ├── app.py                             # Flask backend (real, runnable web interface)
    ├── requirements.txt
    ├── Procfile                           # For deploying to Render/Railway/etc.
    ├── .gitignore                         # Keeps your real .env out of version control
    ├── .env.example                       # Template for your API key -- copy to .env                     
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
5. `Model3_Explanation_Generation.ipynb` and `Pipeline_End_to_End.ipynb` will prompt for a
   **Groq** API key (free) &mdash; get one at
   [console.groq.com/keys](https://console.groq.com/keys)), leave it blank to run in a
   clearly-labelled mock mode instead
6. `Model2_LegalBERT_FineTuning.ipynb` needs a GPU runtime (`Runtime` &rarr; `Change runtime
   type` &rarr; GPU) to fine-tune in a reasonable time

**Model 2 status:** the notebooks include both a deployed zero-shot classifier and a separately
fine-tuned LEGAL-BERT alternative. Fine-tuning produces a substantial accuracy improvement (33-68
percentage-point F1 gain) on four of six clause categories against real, held-out CUAD data; two
categories (compensation, probation) could not be evaluated because CUAD's own taxonomy doesn't
meaningfully cover individual-employment concepts (covered more in depth in the report)

## Running the web app locally

The web app is the working user interface: upload a contract PDF, get a plain-English,
risk-flagged report back. It runs a three-model pipeline &mdash; Model 1 (extraction) and
Model 2 (zero-shot classification) run locally and only a clause Model 2 has confidently
classified is ever sent to Model 3 (Groq) for explanation. Model 3 is never permitted to revise
Model 2's classification.

```bash
cd webapp
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to a new file named `.env` in the same folder and put your real key in it:

```
GROQ_API_KEY=your-real-key-here
```

Then run:

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in a browser. `.env` is listed in `.gitignore` and will
never be committed &mdash; only the placeholder `.env.example` is tracked.

### Deploying it publicly (optional)

The app is a standard Flask app and deploys as-is to any platform that runs Python (Render,
Railway, Fly.io, etc.). Set the `GROQ_API_KEY` environment variable in the platform's dashboard
(never commit it to the repo) and set the start command to:

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
| pdfplumber, |     | Zero-shot DeBERTa    |     | Groq API (Llama/GPT-OSS),|
| heading     |     | NLI (deployed) /     |     | structured JSON prompt,  |
| detection,  |     | fine-tuned LEGAL-BERT|     | honest ERROR state on    |
| OCR fallback|     | (validated, not      |     | failure (never a         |
|             |     | deployed -- see      |     | disguised guess)         |
|             |     | report)              |     |                          |
+-------------+     +----------------------+     +--------------------------+
```

This is the same three-model pipeline in both the notebooks and the web app &mdash; the web app
is not a simplified version of it. Model 2's classification decision and Model 3's explanation
are two separately checkable stages throughout, verified directly during testing.

**Known limitation:** the deployed zero-shot classifier has two reproducible
errors on real-document testing &mdash; it sometimes misreads a compensation clause as a
probation clause and can miss an explicit non-compete clause entirely (leaving it
unclassified). Both are already fixed by the validated fine-tuned classifier described above,
which is not yet deployed for the reasons given in the report.

## Testing

- Model 1: a 9-case synthetic adversarial suite (in `Model1_PDF_Extraction.ipynb`), plus testing
  against real, independently authored employment contracts, including two genuine SEC EDGAR
  filings
- Model 2: benchmarked against real, held-out CUAD data for both the zero-shot and fine-tuned
  classifiers
- Model 3: tested against live API output (not just mock mode) across multiple real contracts
- Full pipeline: run end-to-end against real documents in both notebook and web app form
- A small usability evaluation with real, non-expert participants is reported in the Evaluation chapter of the report.

## Disclaimer

This tool is for informational purposes only and does not constitute legal advice.
