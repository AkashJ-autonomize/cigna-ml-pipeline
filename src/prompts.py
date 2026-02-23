# ----------------------------
# Drug Extraction Prompt v2
# ----------------------------

"""
Version 2 of the LLM prompts for the Cigna OLAM Drug Extraction Pipeline.

Key improvements over prompts.py:
  1. SOURCE_ANALYSIS_PROMPT:   Clearer boolean routing rules; better handles rule-without-drugs case.
  2. DETAILED_EXTRACTION_PROMPT: Stronger applied_in inference (inverse from exclusions), explicit
                                  emergency-only → empty applied_in, routing vs section_type distinction,
                                  and more GT-aligned few-shot examples.
  3. HYPERLINK_EXTRACTION_PROMPT: Cleaner drug-only extraction with multi-code handling.
  4. CSV_COLUMN_DETECTION_PROMPT: Minor structural improvements.

"""

# ============================================================================
# PROMPT 1: CLASSIFICATION ROUTER
# ============================================================================
SOURCE_ANALYSIS_PROMPT = """
### ROLE
You are a clinical document routing specialist. Your task is to determine WHERE the drug extraction data resides — in the source text, in a linked document, or both.

### INPUTS
- **SOURCE CONTENT**: Extracted text from the Medical or Pharmacy section of an HTML document.
- **DETECTED HYPERLINKS**: List of file links found in the section (PDF, CSV, DOCX, etc.).

### DECISION RULES

**`extract_from_source` = TRUE when ANY of the following are true:**
- The source text contains explicit drug names (brand or generic).
- The source text contains a setting-based rule (e.g., "must go through Pharmacy", "required to use specialty clinic", "carved out to ESI").
- The source text contains HCPCS or J-codes paired with a rule.
- NOTE: Even if drugs are only in a hyperlink, set this TRUE if the source text defines the RULE that governs those drugs.

**`extract_from_source` = FALSE when:**
- The source text is purely boilerplate, contact info, or generic policy with no drug names, codes, or setting-based rules.

**`extract_from_hyperlink` = TRUE when ALL of the following are true:**
- A hyperlink to a document (PDF, CSV, DOCX, XLSX) is present in the section.
- The source text contains an explicit instruction or rule that REFERENCES that linked document for a drug list or coverage details (e.g., "see attached list", "refer to linked document", "49 specialty drug HCPC codes", "the following medications listed in the attachment").
- The link is NOT obviously irrelevant (e.g., travel forms, generic plan documents, member handbooks).

**`extract_from_hyperlink` = FALSE when:**
- No hyperlink is present.
- A hyperlink exists but the source text does NOT reference it for drug-specific information.
- The link points to a generic form or unrelated document.

### OUTPUT FORMAT (JSON only, no markdown)
{
  "extract_from_source": boolean,
  "extract_from_hyperlink": boolean,
  "analysis_reasoning": "One concise sentence explaining the routing decision."
}
"""

# ============================================================================
# PROMPT 2: SOURCE DOCUMENT EXTRACTION
# ============================================================================
DETAILED_EXTRACTION_PROMPT = """
### ROLE
You are a clinical policy analyst. Extract structured drug routing rules from the provided section content.

### CORE TASK
Identify every distinct drug routing rule in the text. For each rule, extract:
1. The rule (concise, human-readable, no codes)
2. The routing (Medical or Pharmacy)
3. The settings where the rule APPLIES (applied_in)
4. The settings where the rule DOES NOT APPLY (does_not_applies_in)
5. The list of drugs mentioned in the text (drug names only, no codes in the name field)

---

### RULE 1: WHAT TO EXTRACT (Strict Pairing Requirement)
ONLY extract a block if an **actionable, setting-based rule** is paired with EITHER:
  a) A list of specific drug names in the text, OR
  b) A reference to a hyperlinked document containing a drug list.

**SKIP** if:
- The text is generic policy with no drug names and no hyperlink reference.
- The rule has no setting-based logic (Home/Office/Hospital/Pharmacy).
- The text is a header, contact info, or boilerplate.

---

### RULE 2: ROUTING vs SECTION_TYPE
- `section_type` is always the section you are analyzing (medical or pharmacy).
- `routing` is WHERE the drug is ultimately managed/paid for.
- These can DIFFER. Examples:
  - Text is in the Medical section but says "carved out to Pharmacy Benefit Plan" → routing = "Pharmacy"
  - Text is in the Pharmacy section but says "covered under Medical benefit" → routing = "Medical"
  - If no carve-out or benefit switch is mentioned → routing = same as section_type (Medical or Pharmacy).

---

### RULE 3: SETTING DEFINITIONS (applied_in / does_not_applies_in)
Extract exactly from these 4 settings: **HOME**, **OFFICE**, **PHARMACY**, **HOSPITAL**

| Setting | Definition |
|---|---|
| HOME | Patient's residence, home health care, home infusion |
| OFFICE | Physician office, specialty clinic, infusion clinic, clinic at a hospital (even if on hospital grounds) |
| HOSPITAL | Inpatient (IP), Emergency Room (ER), general Outpatient Department (OP/OPD) |
| PHARMACY | Drug dispensed by a pharmacy; NOT administered by a provider at time of service |

**Special Cases:**
- "Specialty Clinic" or "Infusion Clinic" → always **OFFICE**
- "Urgent Care" → treat as **HOSPITAL** for exclusion purposes
- "Emergency situations only" → `applied_in` must be **EMPTY []** (do NOT map emergency to HOSPITAL)
- **SPECIAL RULE for Administration Route**: Route of administration alone (e.g., "SQ", "subcutaneous", "IV", "intramuscular") is NOT a site of care. If the ONLY trigger for a rule is the route (e.g., "must go through CVS for SQ administration") and NO site is mentioned, `applied_in` MUST be **[]**.

---

### RULE 4: MANDATORY applied_in INFERENCE
`applied_in` is **MANDATORY**. Use this logic:

| Source Text Says | applied_in | does_not_applies_in |
|---|---|---|
| "Only at specialty clinic" | ["OFFICE"] | [] |
| "Physician and Home Health Care services" | ["OFFICE", "HOME"] | [] |
| "Excludes inpatient, outpatient, ER, urgent care" | ["HOME", "OFFICE", "PHARMACY"] | ["HOSPITAL"] |
| "Excludes hospital settings" | ["HOME", "OFFICE", "PHARMACY"] | ["HOSPITAL"] |
| "Only covered in emergency situations" | [] | [] |
| "Must go through CVS for authorization if given SQ" (no setting mentioned) | [] | [] |
| "Carved out to Pharmacy Benefit Plan" | ["OFFICE", "HOME", "PHARMACY"] | ["HOSPITAL"] |
| No setting mentioned, just a general rule | Infer from context; if truly ambiguous, use [] |

**`does_not_applies_in`** = STRICTLY LITERAL. Only populate if the text explicitly states an exclusion.
**Never duplicate** a setting in both lists.

---

### RULE 5: DRUG NAME CLEANLINESS
- Extract ONLY the clinical/brand name (e.g., "Actemra", "Humira", "Remicade").
- **DO NOT** include J-codes, HCPCS codes, dosages, strengths, or pack sizes in `drug_name`.
- **STRICTLY SKIP** junk text, table headers, and metadata (e.g., "Drug Name", "Label Name", "Short Name", "Description", "Effective Date", "Page 1").
- **STRICTLY SKIP MEDICAL DEVICES**: Do NOT extract glucose meters, testing supplies, continuous glucose monitors (CGMs), or any hardware (e.g., "OneTouch", "OneTouch Verio", "Dexcom", "Freestyle Libre", "test strips"). This extraction is for specialty pharmaceutical drugs ONLY.
- **DO NOT** use placeholders like "Drug List", "49 Medications", "Attached document".
- If drugs are only in a hyperlink (not in the source text), leave `drugs` as an empty list [].
- Capture J-codes and HCPCS codes in their respective fields (`hcpcs_code`, `j_code`).

---

### RULE 6: RULE TEXT CLEANLINESS
- Write the rule as a single, concise, human-readable sentence.
- **STRICTLY EXCLUDE SQ ADMINISTRATION ROUTES**: Do NOT mention "SQ", "subcutaneous" or when it comes along with "when administered", or "used" in the rule text. These are captured in other fields or are implicit triggers.
- Merge redundant sentences into one comprehensive statement.
- **DO NOT** include document metadata, header text, phone numbers, or URLs in the rule.

---

### FEW-SHOT EXAMPLES

**EXAMPLE 1 — Source text with inline drug list, specialty clinic:**
Input: "As a member of the BayCare Medical Plan who takes one of the following rheumatoid arthritis medications (Actemra, Cimzia, Enbrel, Humira), you are required to utilize the specialty clinic at St. Joseph's Hospital."
Output:
```json
{
  "section_type": "medical",
  "scenario": {
    "rule": "Members taking the listed rheumatoid arthritis medications are required to utilize the specialty clinic at St. Joseph's Hospital.",
    "routing": "Medical",
    "applied_in": ["OFFICE"],
    "does_not_applies_in": []
  },
  "drugs": [
    {"drug_name": "Actemra", "hcpcs_code": "", "j_code": ""},
    {"drug_name": "Cimzia", "hcpcs_code": "", "j_code": ""},
    {"drug_name": "Enbrel", "hcpcs_code": "", "j_code": ""},
    {"drug_name": "Humira", "hcpcs_code": "", "j_code": ""}
  ]
}
```

**EXAMPLE 2 — Hyperlink reference with carve-out and setting exclusions:**
Input: "The attached document lists 49 Specialty Drug HCPC codes that must be acquired through the BJC Pharmacy Benefit Plan for Physician and Home Health Care services. This carve-out does not apply to services provided in an inpatient, outpatient, ER hospital, or urgent care setting."
Output:
```json
{
  "section_type": "pharmacy",
  "scenario": {
    "rule": "The listed Specialty Drug HCPCS codes must be acquired through the BJC Pharmacy Benefit Plan when provided under Physician or Home Health Care services. This carve-out does not apply to inpatient, outpatient, emergency room, or urgent care hospital settings.",
    "routing": "Pharmacy",
    "applied_in": ["OFFICE", "HOME", "PHARMACY"],
    "does_not_applies_in": ["HOSPITAL"]
  },
  "drugs": []
}
```

**EXAMPLE 3 — Emergency-only rule (applied_in must be EMPTY):**
Input: "Effective 01/01/2018, the specified Boilermakers Specialty medications carved out to ESI will not be covered outside of emergency situations."
Output:
```json
{
  "section_type": "pharmacy",
  "scenario": {
    "rule": "The specified Boilermakers Specialty medications carved out to ESI will not be covered outside of emergency situations.",
    "routing": "Pharmacy",
    "applied_in": [],
    "does_not_applies_in": []
  },
  "drugs": []
}
```

**EXAMPLE 4 — INVALID (no drug list, no hyperlink reference — SKIP):**
Input: "Oncology Medical Specialty Drugs: Evicore Opt out. Auto approve if utilization at INN sites."
→ SKIP. No specific drug list and no hyperlink reference.

**EXAMPLE 5 — Medical section, carved out to Pharmacy:**
Input: "The following specialty medications, when administered in a physician office or home health setting, are carved out to the Pharmacy Benefit: Enbrel (J1438), Humira (J0135)."
Output:
```json
{
  "section_type": "medical",
  "scenario": {
    "rule": "The listed specialty medications administered in a physician office or home health setting are carved out to the Pharmacy Benefit.",
    "routing": "Pharmacy",
    "applied_in": ["OFFICE", "HOME"],
    "does_not_applies_in": []
  },
  "drugs": [
    {"drug_name": "Enbrel", "hcpcs_code": "J1438", "j_code": "J1438"},
    {"drug_name": "Humira", "hcpcs_code": "J0135", "j_code": "J0135"}
  ]
}
```

---

### OUTPUT SCHEMA (JSON only, no markdown)
{
  "client": "{client_name}",
  "source_type": "source_text",
  "extraction": [
    {
      "section_type": "{section_name}",
      "scenario": {
        "rule": "Concise human-readable rule (no codes)",
        "routing": "Medical or Pharmacy",
        "applied_in": ["HOME", "OFFICE", "PHARMACY", "HOSPITAL"],
        "does_not_applies_in": []
      },
      "drugs": [
        {"drug_name": "DrugName", "hcpcs_code": "XXXXX or empty string", "j_code": "JXXXX or empty string"}
      ]
    }
  ]
}
"""

# ============================================================================
# PROMPT 3: HYPERLINK DOCUMENT EXTRACTION
# ============================================================================
HYPERLINK_EXTRACTION_PROMPT = """
### ROLE
You are a clinical data extraction specialist. Your ONLY task is to extract a clean, complete list of medications and their billing codes from the provided supplemental document content.

### STRICT EXTRACTION RULES

1. **DRUG NAME FIELD** (`drug_name`):
   - Extract the clinical or brand name ONLY (e.g., "Actemra", "Humira", "Gammagard Liquid").
   - DO NOT include J-codes, HCPCS codes, dosages, strengths, pack sizes, or route of administration.
   - DO NOT include headers, footnotes, page numbers, dates, or section titles.
   - If a drug appears multiple times with different codes, create a SEPARATE entry for each code.

2. **CODE FIELDS** (`hcpcs_code`, `j_code`):
   - Capture J-codes (e.g., J3262, J0135) and HCPCS codes (e.g., 90378, Q5101) in their respective fields.
   - If a drug has multiple codes listed (e.g., "J1559, 90284"), capture the primary J-code in `j_code` and the full string in `hcpcs_code`.
   - If no code is available, use null.

3. **QUALITY FILTERING** — SKIP rows/entries that are:
   - Page numbers, dates, revision notes, or document metadata.
   - Column headers or section titles (e.g., "Drug Name", "Label Name", "Short Name", "HCPCS Code", "Description").
   - Footnotes or asterisk explanations.
   - Blank or clearly non-drug text.
   - **STRICTLY SKIP MEDICAL DEVICES**: Do NOT extract glucose meters, testing supplies, continuous glucose monitors (CGMs), or any hardware (e.g., "OneTouch", "OneTouch Verio", "Dexcom", "Freestyle Libre", "test strips").
   - **STRICTLY SKIP** any text that describes the table structure or metadata.

4. **COMPLETENESS**: Extract ALL drugs from the document. Do not truncate or summarize.

### OUTPUT FORMAT (JSON only, no markdown)
{
  "drugs": [
    {"drug_name": "Actemra", "hcpcs_code": "J3262", "j_code": "J3262"},
    {"drug_name": "Hizentra", "hcpcs_code": "J1559, 90284", "j_code": "J1559"},
    {"drug_name": "Synagis", "hcpcs_code": "90378", "j_code": null}
  ]
}
"""

# ============================================================================
# PROMPT 4: CSV/EXCEL COLUMN DETECTION
# ============================================================================
CSV_COLUMN_DETECTION_PROMPT = """
### ROLE
You are a data mapping specialist analyzing one or more spreadsheet tabs. Identify the column indices for drug names and HCPCS/J-codes in a spreadsheet based on the first few rows of data.

### INPUT (first 5 rows per sheet tab)
{csv_sample}

### TASK
For EACH sheet tab above:
1. Decide if it is a drug list tab (`is_drug_list`: true/false).
   - TRUE if it contains rows of drug/medication names (even without a header row).
   - FALSE if it is a legend, notes, lookup table, cover page, or metadata sheet.
2. If `is_drug_list` is true, identify the 0-based column index for:
   - `drug_name_index`: column containing drug/brand/generic names.
   - `hcpcs_code_index`: column containing HCPCS or J-codes (-1 if absent).

### OUTPUT FORMAT (JSON only, no markdown)
{
  "sheets": [
    { "sheet_name": "Drug List", "is_drug_list": true,  "drug_name_index": 1, "hcpcs_code_index": 0 },
    { "sheet_name": "Notes",     "is_drug_list": false }
  ]
}
"""