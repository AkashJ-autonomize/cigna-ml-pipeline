# CGM Table Extraction Pipeline

A Python-based HTML table extraction pipeline for processing Continuous Glucose Monitoring (CGM) device data from Cigna guideline documents. This tool parses HTML files to extract structured information about therapeutic and non-therapeutic CGM components.

## 📋 Overview

This pipeline extracts structured data from HTML documents containing CGM device information. It identifies and parses two specific table types:

- **Therapeutic Components Table**: Devices and components approved for therapeutic use
- **Non-Therapeutic Components Table**: Devices and components for non-therapeutic use

The extracted data includes component types, brand names, NDC codes, HCPC codes, and standard dosage information.

## 🏗️ Project Structure

```
cigna-ml-pipeline/
├── files/              # Input HTML files
│   └── kx.html        # Sample input file
├── output/            # Generated JSON output files
│   └── kx.json        # Sample output file
├── src/               # Source code
│   ├── html_parser.py # HTML parsing logic
│   └── schemas.py     # Pydantic data models
├── run.py             # Main execution script
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Place HTML files** in the `files/` directory

2. **Run the extraction pipeline:**
   ```bash
   python run.py
   ```

3. **View results** in the `output/` directory as JSON files

## 📊 Data Schema

### CGMRow

Each extracted row contains:

| Field | Type | Description |
|-------|------|-------------|
| `component` | `str` | Component type (e.g., Sensor, Receiver, Transmitter) |
| `brand_name` | `str` | Brand name of the device |
| `ndc` | `str` (optional) | National Drug Code |
| `hcpc_code` | `List[str]` | List of Healthcare Common Procedure Coding System codes |
| `standard_dose` | `str` (optional) | Standard dose or quantity per year |

### Output Structure

```json
{
  "document_name": "kx.html",
  "total_pages": 1,
  "tables": {
    "therapeutic_table": [
      {
        "component": "Sensor",
        "brand_name": "Dexcom G6 Sensor",
        "ndc": "08627005303",
        "hcpc_code": ["A9276", "A4239"],
        "standard_dose": "365 appliances per year"
      }
    ],
    "non_therapeutic_table": [...]
  },
  "metrics": {
    "duration_seconds": 0.012,
    "total_table_count": 2,
    "table_row_counts": {
      "therapeutic_table": 18,
      "non_therapeutic_table": 45
    }
  }
}
```

## 🔧 Technical Details

### HTML Parsing Strategy

The pipeline uses **BeautifulSoup** to:

1. **Locate tables** by searching for specific heading patterns:
   - Therapeutic: "Which components are therapeutic"
   - Non-therapeutic: "Which components are non-therapeutic"

2. **Extract headers** from the first table row and map them to schema fields using a predefined mapping:
   ```python
   {
       "component": "component",
       "brand name": "brand_name",
       "ndc": "ndc",
       "hcpc code": "hcpc_code",
       "standard dose": "standard_dose"
   }
   ```

3. **Parse data rows** with automatic data cleaning:
   - Removes non-breaking spaces (`\u00a0`) and zero-width spaces (`\u200b`)
   - Normalizes whitespace
   - Splits comma-separated HCPC codes into lists

4. **Validate data** using Pydantic models for type safety

### Key Components

#### `HTMLTableParser` Class

Located in `src/html_parser.py`:

- **`parse_file(file_path)`**: Main entry point for parsing HTML documents
  - Returns a dictionary with `therapeutic_table` and `non_therapeutic_table` keys
  - Each table is a list of `CGMRow` objects

- **`_extract_table_data(table)`**: Extracts and validates data from a single HTML table
  - Maps headers to schema fields
  - Cleans and normalizes cell data
  - Handles HCPC code splitting

#### `process_html_document(file_path)`

Orchestrates the entire extraction process:
- Calls the parser
- Calculates metrics
- Returns a `FinalDocumentOutput` object

### Schema Validation

All extracted data is validated against Pydantic models defined in `src/schemas.py`:

- **`CGMRow`**: Individual table row schema with field validation
- **`FinalDocumentOutput`**: Complete output structure with metadata
- **`UsageMetrics`**: Performance and extraction statistics

## 📈 Metrics & Monitoring

Each extraction generates metrics including:

- **Total table count**: Number of tables extracted (always 2 for CGM documents)
- **Table row counts**: Rows extracted per table type
- **Duration**: Processing time in seconds

## 🛠️ Error Handling

The pipeline includes robust error handling:

- **Invalid rows**: Logged with detailed error messages, processing continues
- **Missing tables**: Gracefully handled with empty arrays
- **Malformed HTML**: Continues processing other files in batch mode

## � Example

### Input
Place an HTML file like `kx.html` in the `files/` directory containing CGM device tables.

### Output
A JSON file `kx.json` in the `output/` directory with:
- 18 therapeutic component entries
- 45 non-therapeutic component entries
- Processing metrics

See [`output/kx.json`](output/kx.json) for a complete example.

## � Features

✅ **Batch Processing**: Processes multiple HTML files in one run  
✅ **UTF-8 Support**: Handles special characters and international text  
✅ **Automatic Cleaning**: Removes HTML entities and normalizes whitespace  
✅ **Type Safety**: Pydantic validation ensures data integrity  
✅ **Performance Metrics**: Tracks processing time and extraction statistics  
✅ **Error Resilience**: Continues processing even if individual rows fail  

## 📄 Dependencies

- **beautifulsoup4** (>=4.12.0): HTML parsing
- **pydantic** (>=2.7.0): Data validation and schema modeling

## 🔄 Workflow

1. HTML files are read from the `files/` directory
2. BeautifulSoup parses the HTML structure
3. Tables are identified by heading text patterns
4. Headers are mapped to schema fields
5. Data rows are extracted, cleaned, and validated
6. Results are serialized to JSON with metrics
7. Output files are saved to `output/` directory

---

**Project**: Cigna Guideline Management (CGM) Pipeline - Phase 3 HTML Extraction  
**Last Updated**: February 2026  
**Maintained by**: Autonomize Team
