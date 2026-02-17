# Cigna OLAM Policy Extraction Pipeline

A production-grade ML pipeline designed to extract structured clinical drug rules from Cigna OLAM HTML documents and their associated attachments.

## 🚀 4-Stage Orchestration Flow

The pipeline uses a conditional logic to ensure high accuracy and resource efficiency.

### 1. Classification & Routing
- **Namespace Analysis**: Detects boundaries between **Medical** and **Pharmacy** sections.
- **Evidence Detection**: Determines if drug rules reside in the source HTML text or external links.
- **Flags**: Returns `source_doc` and `hyperlink` flags to steer subsequent stages.

### 2. Source Text Extraction
- **Trigger**: Runs if `source_doc` flag is TRUE.
- **Task**: Extracts granular `drug_rules` and `scenarios` directly from HTML snippets.
- **Cleanup**: Automatically normalizes drug names and removes noise (HCPCS, J-Codes, Junk).

### 3. Hyperlink & Attachment Extraction
- **Trigger**: Runs if `hyperlink` flag is TRUE.
- **Handlers**: 
    - **CSV/Excel**: Deterministic mapping of clinical lists using LLM-assisted column detection.
    - **DOCX**: Structural parsing of tables and paragraphs.
    - **PDF Vision**: GPT-4o Vision analysis on first 5 pages for precise table extraction.

### 4. Metadata & Consolidation
- **Metadata**: Captures policy-level context (Policy Name, Client, Total Drugs).
- **Consolidation**: Merges identical drug lists across sections and prevents duplicates.

---

## 📂 Directory Structure

```text
├── run.py                 # Main orchestration engine
├── src/
│   ├── html_parser.py     # Structural BeautifulSoup parsing
│   ├── processor.py       # 4-Stage Logic & AI Orchestration
│   ├── handlers.py        # Document Handlers (PDF/DOCX/CSV/Vision)
│   ├── prompts.py         # AI Extraction Prompts
│   ├── schemas.py         # Pydantic models (Data validation)
│   └── utils.py           # Helper utilities
├── files/                 # Input: Provider HTML folders
├── local_path/            # Input: Referenced attachments
├── extraction/            # Output: Final AI-refined Drug Rules (JSON)
└── output/                # Output: Intermediate structural results
```

## 📝 Example Output

The pipeline produces a unified JSON structure like this:

```json
{
  "client": "Boilermakers",
  "extraction": [
    {
      "section_type": "pharmacy",
      "scenario": {
        "rule": "Specialty medications are not covered outside of emergency situations.",
        "routing": "Pharmacy",
        "applied_in": ["Home", "Office"],
        "does_not_applies_in": ["Hospital"]
      },
      "drugs": [
        {
          "drug_name": "TREPROSTINIL SODIUM (GENERIC REMODULIN)",
          "hcpcs_code": "J3285"
        }
      ]
    }
  ],
  "metadata": { "total_drugs_found": 129, "flags": { "source_doc": true, "hyperlink": true } }
}
```

## 🛠 Setup

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Environment**: Populate `.env` with Azure OpenAI credentials.
3. **Run**: `python run.py`
4. **Local Test**: Toggle `USE_LOCAL_TESTING = True` in `run.py` to test specific providers.

---
*Autonomize AI | Cigna Guideline Extraction Project*