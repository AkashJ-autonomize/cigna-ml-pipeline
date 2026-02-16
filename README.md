# Cigna Guideline Extraction Pipeline

A production-grade ML pipeline for extracting structured clinical data from Cigna guideline documents. This repository contains two specialized extraction flows.

## Repository Structure

This repository has two independent extraction pipelines in separate feature branches:

| Pipeline | Branch | Purpose | Input Format |
|----------|--------|---------|--------------|
| **CGM Pipeline** | `cgm-pipeline` | Extracts CGM device component data from HTML tables | HTML |
| **OLAM Pipeline** | `olam-pipeline` | Extracts clinical drug rules from policy documents | HTML + PDF/DOCX/CSV/XLSX |

## CGM Pipeline

**Branch**: `cgm-pipeline`

Processes Continuous Glucose Monitoring (CGM) device data from HTML documents.

### Features
- Extracts therapeutic and non-therapeutic component tables
- Parses device information (brand names, NDC codes, HCPC codes, dosages)
- Batch processing with data validation

### Usage
```bash
git checkout cgm-pipeline
pip install -r requirements.txt
python run.py
```

### Output
```json
{
  "tables": {
    "therapeutic_table": [...],
    "non_therapeutic_table": [...]
  }
}
```

## OLAM Pipeline

**Branch**: `olam-pipeline`

Extracts structured clinical drug rules using a 4-stage conditional orchestration.

### Features
- **Stage 1**: Classification & namespace detection
- **Stage 2**: Source text extraction
- **Stage 3**: Attachment extraction (PDF/DOCX/CSV/XLSX)
- **Stage 4**: Metadata enrichment
- Azure OpenAI GPT-5.2 with Vision support

### Prerequisites
- Python 3.10+
- Poppler for PDF processing
- Azure OpenAI API credentials

### Usage
```bash
git checkout olam-pipeline
pip install -r requirements.txt
# Configure .env with Azure OpenAI credentials
python run.py
```

### Output
```json
{
  "drug_rules": [...],
  "metadata": {
    "total_carve_out_drugs": 45,
    "policy_name": "..."
  }
}
```

## Getting Started

1. Clone the repository
2. Checkout the required branch (`cgm-pipeline` or `olam-pipeline`)
3. Install dependencies: `pip install -r requirements.txt`
4. Place input files in `files/` directory
5. Run: `python run.py`

---

**Maintained by**: Autonomize Team  
**Last Updated**: February 2026