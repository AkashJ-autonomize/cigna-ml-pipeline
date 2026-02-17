"""
This module contains the prompts used by the DrugProcessor to analyze clinical text.
Keeping prompts in a separate file improves maintainability and allows for versioning of the LLM instructions independent of the processing logic.
"""

# PROMPT 1: CLASSIFICATION ROUTER
SOURCE_ANALYSIS_PROMPT = """
### ROLE
You are a smart clinical document analyzer. Your goal is to determine WHERE the drug extraction data resides based on the provided CONTENT and HYPERLINKS.

### INPUT
- **SOURCE CONTENT**: Extracted text from the HTML document.
- **DETECTED HYPERLINKS**: List of file links found in the text.

### YOUR TASK
Set two boolean flags to guide the extraction pipeline:

1. **`extract_from_source`**:
   - **TRUE if**: The source text contains ANY setting-based rules OR explicit drug names.
   - **NOTE**: If the text defines a RULE (e.g., "Must go through Pharmacy Plan") but the drugs are in a link, this must be TRUE to capture the rule.

2. **`extract_from_hyperlink`**:
   - **TRUE if**: A hyperlink is present AND there is an explicit instruction or RULE in the source text pointing to that link for a list of drugs or further coverage details.
   - **FALSE if**: No rule in the source text references the link, or the link is obviously irrelevant (e.g., travel forms, generic homepages).

### OUTPUT FORMAT (JSON)
{
  "extract_from_source": boolean,
  "extract_from_hyperlink": boolean,
  "analysis_reasoning": "Concise explanation of rule-link correlation."
}
"""

# PROMPT 2: SOURCE DOCUMENT EXTRACTION
DETAILED_EXTRACTION_PROMPT = """
### GOAL
Extract clinical "Routing Logic" (rules) from the provided section. 

### 1. CORE EXTRACTION PRINCIPLE: RULE-DRUG PAIRING
- **STRICT REQUIREMENT**: ONLY extract a data block (Scenario + Drugs) if an **actionable, setting-based rule** is explicitly paired with a **drug list** OR a **hyperlink reference**.
- **RULE DEFINITION**: A rule must specify HOW or WHERE a drug list is managed based on clinical setting (Home, Office, Hospital).
- **PAIRING DEFINITION**: The rule must be **immediately adjacent** (above/below) to:
    a) A list of specific drug names.
    b) A hyperlink reference (e.g., "see attached link", "refer to linked document").

### 2. SURGICAL FILTERING (DO NOT EXTRACT IF...)
- **NO DRUGS OR LINKS**: If the text contains a policy but NO drug names and NO hyperlinks are nearby, **SKIP IT**.
- **LINK-ONLY CASE**: If a rule points to a link but lists no drugs in text, extract the rule and leave the `drugs` list empty.
- **GENERIC TEXT**: Ignore document headers, contact info, or boilerplate that doesn't change coverage logic.
- **NO SETTING**: If the rule doesn't vary by setting (Home/Office/Hospital), it is likely not a "Routing Rule" for this pipeline. **SKIP IT**.

### 3. DRUG CLEANLINESS (STRICT DRUG NAMES)
- **EXTRACT ONLY**: The clinical/brand name (e.g., "Actemra").
- **EXCLUDE ALL**: J-codes, HCPCS, dosages, packaging, or random identifiers.
- **NO PLACEHOLDERS**: Do NOT extract "Drug List", "Attached document", or "49 Medications" as drug names.

### 4. RULE CLEANLINESS (HUMAN-READABLE)
- **CLEAN**: Remove all J-codes, HCPCS, and diagnosis codes from the rule text.
- **CRISP**: Merge redundant sentences into one comprehensive coverage scenario.

### 5. SETTING DEFINITIONS
Extract exactly 4 settings: **HOME**, **OFFICE**, **PHARMACY**, **HOSPITAL**.
- **Clinics**: Always classify "Specialty Clinics" or "Infusion Clinics" or "Clinics at [Hospital]" as **OFFICE** (even if on hospital grounds).
- **Hospital**: Reserved for Inpatient, ER, or general Outpatient Departments.
- **HOME**: Patient residence / Home health.
- **Pharmacy**: Medication is dispensed by a pharmacy and not administered by a provider at time of service. 
- **Emergency Situations**: DO NOT map to any of the above. If a drug is ONLY covered in emergency situations, its `applied_in` list must be EMPTY.

The settings should be clearly understood based on the exclusion logic and should be added in `applied_in` list.
For Example:
`Medications does not apply to services provided in an inpatient, outpatient, or ER hospital setting, or in an urgent care setting.`
Result:
applied_in: ["HOME", "OFFICE", "PHARMACY"]
does_not_applies_in: ["HOSPITAL"]

### 6. INTELLIGENT INFERENCE
- **`applied_in`**: MANDATORY. If the rule says "Excluding Hospital", `applied_in` must be ["HOME", "OFFICE", "PHARMACY"]. If a rule says "only covered in emergency", `applied_in` must be EMPTY [].
- **`does_not_applies_in`**: STRICTLY LITERAL. Only populate if explicit exclusions are stated.

### 7. FEW-SHOT EXAMPLES (GUIDANCE)
**EXAMPLE 1 (VALID)**: 
*Text*: "As a member of the Plan who takes one of the following medications (Actemra, Cimzia, Enbrel), you are required to utilize the specialty clinic at St. Joseph’s Hospital..."
*Reasoning*: VALID rule because it contains a specific setting (Clinic/Hospital) paired with explicit drug names.

**EXAMPLE 2 (INVALID)**:
*Text*: "Oncology Medical Specialty Drugs: Evicore Opt out. Auto approve if utilization at INN sites; both OP and IP."
*Reasoning*: INVALID because it does not provide a specific drug list or a clear hyperlink reference for one.

**EXAMPLE 3 (INVALID)**:
*Text*: "Non-Oncology Medical Specialty Drugs: Administered IP or OP at a Facility – Reviewed by Cigna."
*Reasoning*: INVALID because it is generic text without a paired drug list or hyperlink.

**EXAMPLE 4 (VALID)**:
*Text*: "The attached document lists 49 Specialty Drug HCPC codes that must be acquired through the Pharmacy Benefit Plan... This does not apply to outpatient or ER hospital settings."
*Reasoning*: VALID rule because it explicitly points to a supplemental document (hyperlink) for a specific list and defines setting exclusions (applied_in: ["HOME", "OFFICE"], does_not_applies_in: ["HOSPITAL"]).

### 8. OUTPUT SCHEMA (JSON)
{
  "client": "{client_name}",
  "source_type": "source_text",
  "extraction": [
    {
      "section_type": "{section_name}",
      "scenario": { 
          "rule": "Primary coverage rule (concise, NO codes)", 
          "routing": "Medical or Pharmacy", 
          "applied_in": ["HOME", "OFFICE", "PHARMACY", "HOSPITAL"], 
          "does_not_applies_in": [] 
      },
      "drugs": [
        { "drug_name": "Actemra", "hcpcs_code": "...", "j_code": "..." }
      ]
    }
  ]
}
"""

# PROMPT 3: HYPERLINK DOCUMENT EXTRACTION
HYPERLINK_EXTRACTION_PROMPT = """
### ROLE
You are a clinical data assistant. Your goal is to extract a pristine list of medications and their codes from a supplemental document.

### EXTRACTION PRINCIPLES
1. **STRICT DRUG NAMES ONLY**: 
   - Extract clinical or brand names.
   - **DO NOT** include J-codes, HCPCS, dosages, pack sizes, or strengths in the `drug_name` field.
2. **CODE CAPTURE**:
   - Capture J-codes and HCPCS codes in their respective fields if provided.
3. **QUALITY FILTERING**:
   - Skip page numbers, dates, random text, or technical metadata.

### OUTPUT FORMAT (JSON)
{
  "drugs": [
    { "drug_name": "Actemra", "hcpcs_code": "J3262", "j_code": "J3262" }
  ]
}
"""

# PROMPT 4: CSV/EXCEL COLUMN DETECTION
CSV_COLUMN_DETECTION_PROMPT = """
### ROLE
You are a technical data mapper. Your goal is to identify the index of the columns representing 'Drug Name' and 'HCPCS/J-Code' in a spreadsheet based on the first few rows.

### INPUT (First 2 Rows)
{csv_sample}

### TARGET COLUMNS
1. **Drug Name**: Usually labeled as "Drug Name", "Brand Name", "Medication", "Name", or similar.
2. **HCPCS/J-Code**: Usually labeled as "HCPCS", "J-Code", "Current HCPC", "HCPCS Code", or similar.

### YOUR TASK
1. Analyze the sample data.
2. Provide the integer index (starting from 0) for each column.
3. If a column is missing, return -1.

### OUTPUT FORMAT (JSON)
{
  "drug_name_index": integer,
  "hcpcs_code_index": integer,
  "analysis": "Short reasoning for index choice."
}
"""
