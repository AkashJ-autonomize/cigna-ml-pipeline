# Cigna OLAM Policy Extraction Pipeline

A production-grade ML pipeline designed to extract structured clinical drug rules from Cigna OLAM HTML documents and their associated attachments (PDF, DOCX, CSV, XLSX).

## 🚀 Workflow Overview

The pipeline utilizes a sophisticated **4-Stage Conditional Orchestration** logic to ensure accuracy, context-awareness, and resource efficiency.

### Stage 1: Classification & Namespace Detection
- **Namespace Analysis**: Automatically identifies boundaries between **Medical** and **Pharmacy** namespaces.
- **Evidence Detection**: The LLM analyzes the parsed text and structural hyperlink metadata to determine where drug rules reside.
- **Binary Flag Steering**: Returns two boolean flags:
    - `excluded_drug_list_in_source_doc`: TRUE if rules are named directly in the HTML.
    - `excluded_drug_list_in_hyperlink`: TRUE if rules are contained in external attachments (e.g., CSV carve-out lists).

### Stage 2: Source Extraction (Conditional)
- **Trigger**: Runs only if `excluded_drug_list_in_source_doc` is TRUE.
- **Task**: Extracts granular `drug_rules` directly from the HTML text snippets.
- **Strict Logic**: Separates drug names from codes and captures setting-specific usage rules.

### Stage 3: Hyperlink Extraction (Conditional)
- **Trigger**: Runs only if `excluded_drug_list_in_hyperlink` is TRUE.
- **Document Handlers**:
    - **CSV/Excel**: Extracts high-signal clinical lists using `pandas`.
    - **DOCX**: Parses complex formatting/tables using `python-docx`.
    - **PDF Vision**: Converts pages to images and uses **Azure OpenAI GPT-5.2 with Vision** for precise table/list extraction.
- **Exhaustive Mapping**: Maps every individual item in an external list to a separate drug rule object.

### Stage 4: Metadata Extraction
- **Trigger**: Runs if any drug rules are found.
- **Task**: Extracts policy-level metadata such as:
    - `total_carve_out_drugs`
    - `policy_name` & `client_name`
    - `carve_out_applies_to` / `carve_out_excluded_from`
- **Result**: Enriches the final JSON output with high-level policy context.

### Final Step: Result Merging
- Consolidated all rules from Stage 2 and Stage 3 into a single, unified `drug_rules` array.
- Ensures a consistent JSON schema regardless of whether the data was in the source text or an attachment.

---

## 📂 Directory Structure

```text
├── run.py                 # Main orchestration engine (supports --local-test)
├── src/
│   ├── html_parser.py     # Structural BeautifulSoup parsing
│   ├── processor.py       # 4-Stage Logic & Document Handlers
│   ├── handlers.py        # File handlers (PDF/DOCX/CSV/Vision)
│   ├── prompts.py         # Versioned AI Extraction Prompts
│   └── schemas.py         # Pydantic models for extraction consistency
├── files/                 # Input provider HTML folders
├── output/                # Raw Parser logs
├── extraction/            # Final AI-refined Drug Rules
└── cache/                 # PDF Vision PNG snapshots (for verification)
```

## 🛠 Setup & Execution

### Prerequisites
1.  **Python 3.10+**
2.  **Poppler**: Required for PDF processing.
    - Mac: `brew install poppler`
    - Linux: `sudo apt-get install poppler-utils`
    - Windows: Download binary and add to PATH.

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
1.  **Environment Config**:
    Populate `.env` with Azure OpenAI credentials (requires GPT-5.2 / Vision support).
    ```ini
    AZURE_OPENAI_API_KEY=your_key
    AZURE_OPENAI_ENDPOINT=your_endpoint
    AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment
    ```

2.  **Run Full Pipeline**:
    ```bash
    python run.py
    ```

3.  **Local Testing (Mocking)**:
    Modify `run.py` to limit processing to specific provider folders:
    ```python
    LOCAL_TEST_PROVIDERS = ["ProviderName"]
    USE_LOCAL_TESTING = True
    ```

---
*Developed for Autonomize | Cigna Guideline Extraction Project*
