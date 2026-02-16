"""
This module contains the prompts used by the DrugProcessor to analyze clinical text.
Keeping prompts in a separate file improves maintainability and allows for versioning
of the LLM instructions independent of the processing logic.
"""

# PROMPT 1: CLASSIFICATION
SOURCE_ANALYSIS_PROMPT = """
Analyze the provided CONTENT and STRUCTURAL HYPERLINKS to set dual boolean flags.

STRICT CRITERIA:

1. EXCLUSION DETECTION (Source):
   - Set 'excluded_drug_list_in_source_doc' to TRUE if the text context itself clearly names SPECIFIC drugs (e.g., "Actemra", "Procrit") and their associated exclusion rules, or requirements like "must use specialty clinic".
   - If the text only gives general hints but no names/rules in-line, set to FALSE.

2. EXCLUSION DETECTION (Hyperlink):
   - Set 'excluded_drug_list_in_hyperlink' to TRUE *ONLY* if:
     a) There is a STRUCTURAL HYPERLINK provided in the metadata with a relevant file extension (.pdf, .csv, .xls, .xlsx, .docx).
     b) The sentence/meaning conveying the link includes EXPLICIT EVIDENCE of exclusion criteria. Evidence includes:
        - Specific **Drug Names** (e.g., Actemra, Procrit).
        - **J-codes** or **HCPCS codes** (e.g., J0885, J3490).
        - Explicit keywords like **"Exclusion list"**, **"Carve-out rule"**, **"Authorization required"**, or **"Non-covered"**.
   - If the sentence only says "MCM Drug List" or "2023 Formulary" (general titles with no rule/code context), set to FALSE.
   - If a link exists but the meaning is general (Benefits, Enrollment, Expense), set to FALSE.

3. DUAL PRESENCE:
   - If both in-source names/rules AND a relevant external link exist, set BOTH to TRUE.

OUTPUT FORMAT (JSON):
{
  "excluded_drug_list_in_source_doc": true | false,
  "excluded_drug_list_in_hyperlink": true | false,
  "analysis_reasoning": "Crisp reasoning for the boolean flags based on semantic relevance."
}
"""

###########################################################################################################################

# PROMPT 2: SOURCE EXTRACTION
DETAILED_EXTRACTION_PROMPT = """
Analyze the clinical content provided and extract structured DRUG RULES from the SOURCE TEXT with extreme precision.

STRICT RULES:
1. EXPLICIT SOURCE EXTRACTION: ONLY extract rules that are clearly stated in the provided text. Do not hallucinate or use external knowledge.
2. CLEAN DRUG NAMES: Extract ONLY the brand or generic name (e.g., "Humira", "Actemra"). No HCPCS or J-codes in the 'drug_name' field.
3. DETAIL CAPTURE: For each drug, extract:
   - hcpcs_code / j_code: Capture if explicitly present.
   - classification: Describe the drug classification (e.g., "Specialty Drug - Carve-Out", "Medical", "Pharmacy").
   - summary: Ultra-crisp consolidated summary. State ONLY the rule and core reason. EXPLICITLY FORBID: specific facility names, specific pharmacy names, regional names (East/West), and identification numbers (TIN, NPI, etc.).
   
4. SETTING-SPECIFIC EXTRACTION (OPTIONAL):
   - If the text describes different rules for different care settings (e.g., physician office, hospital, emergency room), extract setting-specific data.
   - Standard care settings: physician_office, home_health_care, inpatient_hospital, outpatient_hospital, emergency_room, urgent_care
   - For prior_auth_required: If rules vary by setting, provide an object with setting-specific booleans. If uniform, use a single boolean. If not mentioned, omit this field.
   - For pharmacy_benefit_plan_required: Provide setting-specific booleans indicating if pharmacy benefit plan routing is required. If not mentioned, omit this field.
   
5. SCENARIOS EXTRACTION (OPTIONAL - ONLY IF DETAILS ARE AVAILABLE):
   - **IMPORTANT**: Only create scenarios if the source text provides SPECIFIC details about care settings, routing, or benefit types.
   - If the text only mentions drugs with a general requirement (e.g., "must use specialty clinic") WITHOUT specifying settings or routing details, DO NOT create scenarios. Just provide the summary.
   - Each scenario should only be created when you have ACTUAL information from the source for:
     * setting: The care setting (use snake_case: physician_office, inpatient_hospital, etc.) - ONLY if explicitly mentioned
     * routing: Where to obtain the drug - ONLY if explicitly stated
     * benefit_type: Either "pharmacy" or "medical" - ONLY if clearly indicated
     * prior_auth: Boolean indicating if prior authorization is required - ONLY if explicitly stated
     * notes: Any exceptions, special conditions, or additional context
   - If you cannot determine these details from the source, leave scenarios as an empty array [].
   
6. CONSOLIDATION RULE:
   - Provide only ONE object per unique drug/code. 
   - Combine all location-based rules into that single object's 'summary', 'scenarios', and setting-specific fields.

OUTPUT FORMAT (JSON):
{
  "drug_rules": [
    {
      "drug_name": "...",
      "hcpcs_code": "...",
      "j_code": "...",
      "classification": "...",
      "prior_auth_required": {
        "physician_office": false,
        "home_health_care": false,
        "inpatient_hospital": true,
        "outpatient_hospital": true,
        "emergency_room": false,
        "urgent_care": false
      },
      "pharmacy_benefit_plan_required": {
        "physician_office": true,
        "home_health_care": true,
        "inpatient_hospital": false,
        "outpatient_hospital": false,
        "emergency_room": false,
        "urgent_care": false
      },
      "summary": "Consolidated summary of rules across all settings.",
      "scenarios": [
        {
          "setting": "physician_office",
          "routing": "Pharmacy Benefit Plan",
          "benefit_type": "pharmacy",
          "prior_auth": false,
          "notes": "Must acquire through specialty pharmacy"
        },
        {
          "setting": "inpatient_hospital",
          "routing": "Direct facility dispensing",
          "benefit_type": "medical",
          "prior_auth": true,
          "notes": "Carve-out does not apply; standard medical benefit rules"
        }
      ]
    }
  ]
}


EXAMPLE WITH LIMITED INFORMATION (NO DETAILED SCENARIOS):
{
  "drug_rules": [
    {
      "drug_name": "Humira",
      "classification": "RA/Psoriatic Arthritis Drug",
      "summary": "Members taking this drug are required to utilize the specialty clinic.",
      "scenarios": []
    }
  ]
}

NOTE: Prioritize accuracy over completeness. If information is not explicitly stated, omit the field or use an empty array rather than guessing or creating null values. If prior auth or pharmacy benefit requirements are uniform across all settings, you may use a simple boolean instead of the object format.
"""


###########################################################################################################################

# PROMPT 3: HYPERLINK EXTRACTION
HYPERLINK_EXTRACTION_PROMPT = """
Analyze the clinical content provided (hyperlink document results) and extract structured DRUG RULES.

STRICT RULES:
1. EXHAUSTIVE EXTRACTION: Extract EVERY individual drug/code item mentioned as a separate rule. Do not summarize list contents.
2. CLEAN DRUG NAMES: 
   - Extract ONLY the name (e.g., "Actemra"). No codes in this field.
   - If only a code like "J3262" is provided, use the common name for that code if known, or as a last resort use the code itself but mark it clearly.
3. DETAIL CAPTURE:
   - hcpcs_code / j_code: Capture if explicitly provided.
   - classification: Describe the drug classification.
   - summary: Ultra-crisp consolidated summary. Strip ALL specific entity names, facility/pharmacy identifiers, geographical regions, and numeric codes (TIN, etc.). Focus strictly on the clinical logic.
   
4. SETTING-SPECIFIC EXTRACTION:
   - If the document describes different rules for different care settings, extract setting-specific data.
   - Standard care settings: physician_office, home_health_care, inpatient_hospital, outpatient_hospital, emergency_room, urgent_care
   - For prior_auth_required: If rules vary by setting, provide an object with setting-specific booleans. If uniform, use a single boolean.
   - For pharmacy_benefit_plan_required: Provide setting-specific booleans indicating if pharmacy benefit plan routing is required.
   
5. SCENARIOS EXTRACTION:
   - Extract detailed scenarios for each care setting mentioned.
   - Each scenario must include:
     * setting: The care setting (use snake_case)
     * routing: Where to obtain the drug
     * benefit_type: Either "pharmacy" or "medical"
     * prior_auth: Boolean indicating if prior authorization is required
     * notes: Any exceptions or special conditions (strip all specific identifiers)
   
6. CONSOLIDATION RULE:
   - Provide only ONE object per unique drug/code. Do not create separate objects for different scenarios of the same drug.
   - Combine all location-based rules into the 'summary', 'scenarios', and setting-specific fields of that single object.

OUTPUT FORMAT (JSON):
{
  "drug_rules": [
    {
      "drug_name": "...",
      "hcpcs_code": "...",
      "j_code": "...",
      "classification": "...",
      "prior_auth_required": {...} or true/false,
      "pharmacy_benefit_plan_required": {...},
      "summary": "Consolidated summary of rules from the document.",
      "scenarios": [
        {
          "setting": "...",
          "routing": "...",
          "benefit_type": "pharmacy" or "medical",
          "prior_auth": true/false,
          "notes": "..."
        }
      ]
    }
  ]
}
"""

# PROMPT 4: METADATA EXTRACTION
METADATA_EXTRACTION_PROMPT = """
Analyze the provided content and extract POLICY-LEVEL METADATA about the drug rules.

EXTRACTION GUIDELINES:
1. total_carve_out_drugs: Count or extract the total number of drugs mentioned in carve-out lists
2. source_document: Extract the path or reference to the source document (e.g., "local_path/BJCHealthcare/BJC Healthcare Specialty Drugs.docx")
3. policy_name: Extract the policy name or description (e.g., "BJC Specialty Drug Pharmacy Benefit Plan Carve-Out")
4. carve_out_applies_to: List of care settings where the carve-out applies (e.g., ["physician_office", "home_health_care"])
5. carve_out_excluded_from: List of care settings excluded from the carve-out (e.g., ["inpatient_hospital", "outpatient_hospital", "emergency_room", "urgent_care"])
6. client_name: Extract the client/organization name
7. specialty_pharmacy_vendor: Extract the specialty pharmacy vendor name if mentioned (e.g., "Vivio", "Express Scripts")
8. additional_info: Any other relevant policy-level information

OUTPUT FORMAT (JSON):
{
  "total_carve_out_drugs": 49,
  "source_document": "local_path/BJCHealthcare/BJC Healthcare Specialty Drugs.docx",
  "policy_name": "BJC Specialty Drug Pharmacy Benefit Plan Carve-Out",
  "carve_out_applies_to": ["physician_office", "home_health_care"],
  "carve_out_excluded_from": ["inpatient_hospital", "outpatient_hospital", "emergency_room", "urgent_care"],
  "client_name": "BJC Healthcare",
  "specialty_pharmacy_vendor": "BJC Pharmacy Benefit Plan",
  "additional_info": {}
}

NOTE: All fields are optional. Only extract what is clearly stated in the content.
"""

