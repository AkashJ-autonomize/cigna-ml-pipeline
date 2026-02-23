# Cigna OLAM Policy Extraction Pipeline

A production-grade AI pipeline designed to extract structured clinical drug rules from Cigna OLAM HTML documents and their associated attachments.


## 🚀 4-Stage Orchestration Flow

The pipeline uses a dynamic, conditional logic flow to ensure high accuracy while minimizing unnecessary API calls.

### 1. Classification & Routing
- **Namespace Analysis**: Detects boundaries between **Medical** and **Pharmacy** sections.
- **Evidence Detection**: Uses GPT-5.2 to determine if drug rules reside in the source HTML text, in external linked documents, or both.
- **Flags**: Returns `source_doc` and `hyperlink` flags to steer subsequent stages.

### 2. Source Document Extraction
- **Trigger**: Runs if the `source_doc` flag is TRUE.
- **Task**: Extracts granular `scenario` rules directly from HTML snippets, mapping exactly to clinical Settings of Care (`applied_in` arrays: HOME, OFFICE, HOSPITAL, PHARMACY).
- **Cleanup**: Normalizes drug names, strictly filtering out medical devices, route-of-administration triggers, and HCPCS/J-Codes into distinct fields.

### 3. Hyperlink & Attachment Extraction
- **Trigger**: Runs if the `hyperlink` flag is TRUE.
- **Early Exit Guard**: Intelligently skips extraction if Stage 2 found no valid governing rule in the source text, preventing orphaned drug extraction.
- **Handlers**: 
    - **CSV/Excel**: LLM-assisted column detection paired with rapid deterministic extraction. Supports multi-sheet auto-discovery.
    - **DOCX**: Structural parsing of tables and paragraphs.
    - **PDF Vision**: Batched GPT-5.2 Vision API analysis on the first 5 pages for precise table extraction.

### 4. Cross-Section Consolidation
- **Merger**: Identifies and merges identical drug lists across Medical and Pharmacy sections to prevent duplicates.
- **Metadata**: Captures policy-level context (Policy Name, Client, Flags, Total Drugs).
- **LLM Usage Stats**: Tracks and accumulates API calls, prompt/completion tokens, and wall-clock duration dynamically per file.

---

## 📂 Directory Structure

```text
├── run.py                 # Main orchestration engine & summary generator
├── src/
│   ├── html_parser.py     # Structural BeautifulSoup parsing
│   ├── processor.py       # Core orchestration, prompt management, & cross-section merger
│   ├── handlers.py        # Document Handlers (PDF Vision/DOCX/CSV/TXT)
│   ├── prompts.py         # AI Extraction Prompts (GPT-5.2 optimized)
│   ├── schemas.py         # Pydantic models (Data validation)
│   └── utils.py           # Helper utilities & lazy-downloading mechanics
├── files/                 # Input: Provider HTML folders
├── local_path/            # Input: Referenced attachments
├── extraction/            # Output: Final AI-refined Drug Rules (JSON)
└── pipeline_summary.json  # Output: Global run statistics & per-client API usage
```

## 📝 Example Output

The pipeline produces a unified JSON structure like this:

```json
{
  "client": "Boilermakers",
  "source_type": "source_text",
  "extraction": [
    {
      "section_type": "pharmacy",
      "scenario": {
        "rule": "Specialty medications are not covered outside of emergency situations.",
        "routing": "Pharmacy",
        "applied_in": [],
        "does_not_applies_in": []
      },
      "drugs": [
        {
          "drug_name": "TREPROSTINIL SODIUM",
          "hcpcs_code": "J3285",
          "j_code": "J3285"
        }
      ]
    }
  ],
  "metadata": {
    "total_drugs_found": 141,
    "total_scenarios_found": 1,
    "same_drugs_across_scenarios": true,
    "flags": {
      "source_doc": true,
      "hyperlink": true
    },
    "llm_stats": {
      "llm_calls": 7,
      "input_tokens": 8500,
      "output_tokens": 1705,
      "total_tokens": 10205,
      "duration_seconds": 19.654
    }
  }
}
```

## 🛠 Setup & Execution

1. **Install Dependencies**: `pip install -r requirements.txt` (requires `pdf2image`, `docx`, `pandas`, `openai`, `pydantic`, etc.)
2. **Environment**: Populate `.env` with Azure OpenAI credentials (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, etc.).
3. **Run Pipeline**: `python run.py`
4. **Local Testing**: Toggle `USE_LOCAL_TESTING = True` in `run.py` to test on a limited subset of providers.

---
*Autonomize AI | Cigna Guideline Extraction Project*