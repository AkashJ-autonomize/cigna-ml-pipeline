import os
import json
import re
from typing import List, Dict, Optional, Any, Tuple
from openai import AzureOpenAI
from src.schemas import OLAMExtraction, ExtractionSection, RuleScenario, DrugItem, RefinedExtraction, ExtractionMetadata
from src.prompts import SOURCE_ANALYSIS_PROMPT, DETAILED_EXTRACTION_PROMPT, HYPERLINK_EXTRACTION_PROMPT, CSV_COLUMN_DETECTION_PROMPT
from src.handlers import DocumentHandler
from src.utils import lazy_download_documents, build_context_string

class DrugProcessor:
    """
    Orchestrates the filtering of parsed text and the extraction of drug rules using LLMs.
    Refactored to support granular Medical/Pharmacy parsing and Pharmacy/Hospital/Home/Office classification.
    """
    def __init__(self):
        """Initializes the Azure OpenAI client and document handlers."""
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.handler = DocumentHandler(self.client, self.deployment_name)

    def filter_relevant_points(self, extraction: OLAMExtraction) -> Dict[str, List[str]]:
        """
        Filters the raw parsed text for high-signal clinical drug keywords.
        Returns a dict with 'medical' and 'pharmacy' keys.
        """
        drug_keywords = {
            "prior authorization", "pre-authorization", "specialty drug", "carve-out",
            "j-code", "hcpcs", "injection", "infusion", "oncology", "chemotherapy",
            "clinical criteria", "quantity limit", "step therapy", "requires approval",
            "medication", "pharmacy", "clinic", "covered", "not covered", "exclusion",
            "esi", "special handling", "exclusion list", "carveout", "carved-out", "formularies",
            "self-administered", "medical benefit", "pharmacy benefit",
            "auth", "authorization", "cvs", "caremark", "optum", "drug list", "specialty list"
        }
        
        relevant = {"medical": [], "pharmacy": []}
        
        def has_drug_code(text: str) -> bool:
            return bool(re.search(r'\b[JQSCA][0-9]{4}\b', text))

        for section in ["medical", "pharmacy"]:
            points = getattr(extraction, section).text
            for p in points:
                p_lower = p.lower()
                if any(kw in p_lower for kw in drug_keywords) or \
                   any(suf in p_lower for suf in ["mab", "mib", "nib", "tinib", "cept"]) or \
                   has_drug_code(p):
                    relevant[section].append(p)
        
        print(f"    [FILTER] {len(relevant['medical'])} med points, {len(relevant['pharmacy'])} pharm points")
        return relevant

    def process_with_llm(self, relevant_points: Dict[str, List[str]], full_extraction: OLAMExtraction, client_name: str = "Unknown", output_dir: Optional[str] = None) -> RefinedExtraction:
        """
        Main 3-Stage Orchestration Flow.
        Updated to support the nested 'extraction' structure and literal rule enforcement.
        """
        all_sections: List[ExtractionSection] = []
        flags = {"source_doc": False, "hyperlink": False}
        analysis_reasoning = []
        
        sections_to_process = []
        if relevant_points["medical"]: sections_to_process.append("medical")
        if relevant_points["pharmacy"]: sections_to_process.append("pharmacy")
        
        if not sections_to_process:
            metadata = ExtractionMetadata(
                analysis_reasoning="No relevant Medical or Pharmacy sections detected in the source document.",
                total_drugs_found=0,
                total_scenarios_found=0
            )
            return RefinedExtraction(client=client_name, extraction=[], metadata=metadata)
        
        if output_dir:
             lazy_download_documents(full_extraction, output_dir)

        for sec in sections_to_process:
            print(f"    [PROCESS] Analyzing section: {sec.upper()}")
            
            sec_text = "\n".join(relevant_points[sec])
            sec_links_obj = getattr(full_extraction, sec).hyperlinks
            sec_links_str = self._format_links_for_prompt(sec_links_obj)
            
            # --- STAGE 1: Classification (Router) ---
            print(f"      [STAGE 1] Routing logic for {sec}...")
            router_prompt = SOURCE_ANALYSIS_PROMPT.replace("{section_name}", sec) + \
                            f"\n\nSOURCE CONTENT ({sec}):\n{sec_text}\n\nDETECTED HYPERLINKS:\n{sec_links_str}"
            
            routing_decision = self._call_llm_json(router_prompt)
            
            extract_source = routing_decision.get("extract_from_source", False)
            extract_hyperlink = routing_decision.get("extract_from_hyperlink", False)
            if extract_source: flags["source_doc"] = True
            if extract_hyperlink: flags["hyperlink"] = True
            if "analysis_reasoning" in routing_decision:
                analysis_reasoning.append(f"{sec}: {routing_decision['analysis_reasoning']}")

            # --- STAGE 2: Source Extraction ---
            current_sec_extractions = []
            if extract_source:
                print(f"      [STAGE 2] Extracting Scenarios & Drugs from Source ({sec})...")
                s_prompt = DETAILED_EXTRACTION_PROMPT.replace("{section_name}", sec).replace("{client_name}", client_name) + \
                           f"\nSOURCE CONTENT:\n{sec_text}\n\nDETECTED HYPERLINKS:\n{sec_links_str}"
                
                s_data = self._call_llm_json(s_prompt)
                extracted_sections = self._parse_extraction_sections(s_data)
                
                # FORCE section_type override to prevent LLM mislabeling
                for ex in extracted_sections:
                    ex.section_type = sec
                    # Ensure we only have one scenario (LLM might still return a list in some cases)
                    # This logic will be handled inside _parse_extraction_sections or here
                    
                current_sec_extractions = extracted_sections
                
            # --- STAGE 3: Hyperlink Extraction ---
            if extract_hyperlink:
                print(f"      [STAGE 3] Extracting Drugs ONLY from Hyperlinks ({sec})...")
                hyperlink_text, pre_extracted_drugs = self._gather_hyperlink_content(sec_links_obj)
                
                h_drugs = pre_extracted_drugs
                if hyperlink_text:
                    h_prompt = HYPERLINK_EXTRACTION_PROMPT + f"\n\nDOCUMENT CONTENT:\n{hyperlink_text[:15000]}"
                    
                    h_data = self._call_llm_json(h_prompt)
                    for d_raw in h_data.get("drugs", []):
                        try:
                            h_drugs.append(DrugItem(**d_raw))
                        except:
                            continue
                    
                if h_drugs:
                    if current_sec_extractions:
                         for ex in current_sec_extractions:
                             # SURGICAL: Only apply hyperlink drugs if the rule references a list/document/link
                             has_link_ref = False
                             if ex.scenario:
                                 has_link_ref = any(kw in ex.scenario.rule.lower() for kw in ["list", "document", "link", "attached", "provision", "refer to"])
                             
                             if has_link_ref:
                                 existing_sigs = {(d.drug_name.lower(), d.hcpcs_code) for d in ex.drugs}
                                 for hd in h_drugs:
                                     sig = (hd.drug_name.lower(), hd.hcpcs_code)
                                     if sig not in existing_sigs:
                                         ex.drugs.append(hd)
                                         existing_sigs.add(sig)
                                 print(f"      [STAGE 3] Applied {len(h_drugs)} hyperlink drugs to scenario referencing external list.")
                             else:
                                 print(f"      [STAGE 3] Skipped applying {len(h_drugs)} hyperlink drugs - rule does not reference external list.")
                    else:
                        print(f"      [STAGE 3] Found drugs but NO specific rule/scenario found in source for {sec}. Skipping.")

            # --- VALIDATION: RULE-DRUG PAIRING & SPECIAL HANDLING ---
            valid_sec_extractions = []
            for ex in current_sec_extractions:
                # 1. Must have drugs
                if not ex.drugs:
                    print(f"      [VALIDATION] Dropping {sec} section - no associated drugs found in source or hyperlinks.")
                    continue
                
                # 2. Must have actionable scenario with 'special handling' (settings)
                if not ex.scenario:
                    print(f"      [VALIDATION] Dropping {sec} section - no scenario found.")
                    continue

                # Stricter: must have either applied_in or does_not_applies_in, 
                # OR a rule that isn't just a placeholder like "No specific rule"
                is_useless_placeholder = any(kw in ex.scenario.rule.lower() for kw in ["no specific rule", "not found", "n/a"])
                has_settings = bool(ex.scenario.applied_in or ex.scenario.does_not_applies_in)
                has_link_keywords = any(kw in ex.scenario.rule.lower() for kw in ["list", "document", "link", "attached", "provision", "refer to"])
                
                # We keep the scenario if:
                # 1. It has explicit settings (Home/Office/Hospital)
                # 2. OR it is a long, descriptive rule (> 40 chars)
                # 3. OR it explicitly references an external list/link
                if not is_useless_placeholder and (has_settings or len(ex.scenario.rule) > 40 or has_link_keywords):
                    valid_sec_extractions.append(ex)
                else:
                    print(f"      [VALIDATION] Dropping section - scenario missing special handling or too generic: {ex.scenario.rule[:50]}...")
            
            all_sections.extend(valid_sec_extractions)

        # --- STAGE 4: Consolidation (Cross-Section) ---
        all_sections = self._consolidate_sections_cross_section(all_sections)

        # Metadata
        total_drugs = sum(len(ex.drugs) for ex in all_sections)
        total_scenarios = sum(1 for ex in all_sections if ex.scenario) # Count scenarios
        same_drugs = len(all_sections) == 1
        
        metadata = ExtractionMetadata(
            total_drugs_found=total_drugs,
            total_scenarios_found=total_scenarios,
            same_drugs_across_scenarios=same_drugs,
            analysis_reasoning=" | ".join(analysis_reasoning),
            flags=flags
        )
        if all_sections and not metadata.policy_name:
             metadata.policy_name = f"Extracted Rules ({total_drugs} drugs)"
        
        print(f"    [COMPLETE] Extracted {len(all_sections)} unique drug lists with {total_scenarios} scenarios.")
        return RefinedExtraction(client=client_name, source_type="source_text", extraction=all_sections, metadata=metadata)

    def _consolidate_sections_cross_section(self, sections: List[ExtractionSection]) -> List[ExtractionSection]:
        """
        Merges ExtractionSections that have identical (or very similar) drug lists.
        Only merges within the same section_type (medical/pharmacy) to preserve category separation.
        """
        if not sections: return []

        def normalize_name(n: str) -> str:
            clean = re.sub(r'[^a-zA-Z0-9 ]', ' ', n).lower()
            return " ".join(sorted(clean.split()))

        processed = []
        for s in sections:
            name_words = set()
            for d in s.drugs:
                name_words.update(normalize_name(d.drug_name).split())
            
            codes = {d.hcpcs_code for d in s.drugs if d.hcpcs_code} | \
                    {d.j_code for d in s.drugs if d.j_code}
            
            processed.append({
                "sec": s,
                "name_words": name_words,
                "codes": codes,
                "merged": False
            })

        final_consolidated = []
        
        for i in range(len(processed)):
            if processed[i]["merged"]: continue
            
            base = processed[i]
            base["merged"] = True
            
            for j in range(i + 1, len(processed)):
                if processed[j]["merged"]: continue
                
                target = processed[j]
                
                # REFINEMENT: Only merge if section types match
                if base["sec"].section_type != target["sec"].section_type:
                    continue
                
                match = False
                if base["codes"] and target["codes"]:
                    shared_codes = base["codes"] & target["codes"]
                    if len(shared_codes) / max(1, min(len(base["codes"]), len(target["codes"]))) > 0.5:
                        match = True
                elif base["name_words"] and target["name_words"]:
                    shared_words = base["name_words"] & target["name_words"]
                    if len(shared_words) / max(1, min(len(base["name_words"]), len(target["name_words"]))) > 0.7:
                        match = True
                
                if match:
                    target["merged"] = True
                    # Merge scenario if target has a rule and base doesn't, or if target's rule is more specific
                    if target["sec"].scenario and not base["sec"].scenario:
                        base["sec"].scenario = target["sec"].scenario
                    elif target["sec"].scenario and base["sec"].scenario:
                        # If both have scenarios, combine the rules to ensure "most important" info is kept
                        if target["sec"].scenario.rule.lower().strip() not in base["sec"].scenario.rule.lower().strip():
                            base["sec"].scenario.rule = f"{base['sec'].scenario.rule} | {target['sec'].scenario.rule}"
                            # Merge settings
                            base["sec"].scenario.applied_in = list(set(base["sec"].scenario.applied_in + target["sec"].scenario.applied_in))
                            base["sec"].scenario.does_not_applies_in = list(set(base["sec"].scenario.does_not_applies_in + target["sec"].scenario.does_not_applies_in))
                    
                    # Union drugs
                    existing_drugs = {(d.hcpcs_code, d.j_code, d.drug_name.lower()) for d in base["sec"].drugs}
                    for d in target["sec"].drugs:
                        if (d.hcpcs_code, d.j_code, d.drug_name.lower()) not in existing_drugs:
                            base["sec"].drugs.append(d)
                            existing_drugs.add((d.hcpcs_code, d.j_code, d.drug_name.lower()))
            
            final_consolidated.append(base["sec"])

        return final_consolidated

    def _consolidate_scenarios(self, scenarios: List[RuleScenario]) -> Optional[RuleScenario]:
        """
        Picks the single most important scenario from a list.
        Merges redundant rules into a single crisp rule.
        """
        if not scenarios: return None
        if len(scenarios) == 1: return scenarios[0]
        
        # Merge all distinct setting-based rules into one
        base_scenario = scenarios[0]
        for s in scenarios[1:]:
            if s.rule.lower().strip() not in base_scenario.rule.lower().strip():
                base_scenario.rule = f"{base_scenario.rule} | {s.rule}"
            base_scenario.applied_in = list(set(base_scenario.applied_in + s.applied_in))
            base_scenario.does_not_applies_in = list(set(base_scenario.does_not_applies_in + s.does_not_applies_in))
            
        return base_scenario

    def _format_links_for_prompt(self, hyperlinks: List[Any]) -> str:
        valid_extensions = {".pdf", ".csv", ".xlsx", ".xls", ".docx", ".doc"}
        formatted = []
        for l in hyperlinks:
            ext = os.path.splitext(l.url)[1].lower()
            if ext in valid_extensions:
                formatted.append(f"- {l.url} (Context text: '{l.text}')")
        return "\n".join(formatted) if formatted else "None relevant."

    def _gather_hyperlink_content(self, hyperlinks: List[Any]) -> Tuple[str, List[DrugItem]]:
        """
        Reads content from downloaded documents.
        Returns (content_string, list_of_preextracted_drugs).
        """
        contents = []
        pre_extracted_drugs = []
        
        for link in hyperlinks:
            if not (link.local_path and os.path.exists(link.local_path)):
                continue
                
            ext = os.path.splitext(link.local_path)[1].lower()
            file_name = os.path.basename(link.local_path)
            
            try:
                if ext == ".txt":
                    with open(link.local_path, 'r', encoding='utf-8', errors='ignore') as f:
                        contents.append(f"--- FILE: {file_name} ---\n{f.read()[:5000]}")
                
                elif ext in [".csv", ".xls", ".xlsx"]:
                    # 1. Detect Columns via LLM (Small Sample)
                    csv_sample = self.handler.get_csv_sample(link.local_path)
                    detect_prompt = CSV_COLUMN_DETECTION_PROMPT.replace("{csv_sample}", csv_sample)
                    
                    mapping = self._call_llm_json(detect_prompt)
                    drug_idx = mapping.get("drug_name_index", 0)
                    hcpcs_idx = mapping.get("hcpcs_code_index")
                    
                    if not isinstance(drug_idx, int) or drug_idx < 0: drug_idx = 0
                    if hcpcs_idx is not None and (not isinstance(hcpcs_idx, int) or hcpcs_idx < 0): hcpcs_idx = None
                    
                    # 2. Deterministic Extraction in Python
                    rows = self.handler.process_csv(link.local_path, drug_col_idx=drug_idx, hcpcs_col_idx=hcpcs_idx)
                    for r in rows:
                        # Parse "Drug: Name | Code: 123" back into DrugItem
                        parts = r.split(" | ")
                        d_name = parts[0].replace("Drug: ", "").strip()
                        c_code = parts[1].replace("Code: ", "").strip() if len(parts) > 1 else None
                        
                        # Apply specialized cleaning even for CSV sources
                        d_name = self._clean_drug_name(d_name)
                        
                        if d_name and self._is_valid_drug(d_name):
                            pre_extracted_drugs.append(DrugItem(
                                drug_name=d_name,
                                hcpcs_code=c_code,
                                j_code=c_code # Duplicate for consistency
                            ))
                    print(f"      [CSV/EXCEL] Directly extracted {len(pre_extracted_drugs)} drugs using mapped columns.")

                elif ext in [".docx", ".doc"]:
                    text = self.handler.process_docx(link.local_path)
                    contents.append(f"--- FILE: {file_name} ---\n{text[:10000]}")
                elif ext == ".pdf":
                    vision_text = self.handler.process_pdf_vision(link.local_path)
                    contents.append(f"--- FILE: {file_name} ---\n{vision_text}")
            except Exception as e:
                print(f"      [DOC ERROR] Failed to read {file_name}: {e}")
                
        return "\n\n".join(contents), pre_extracted_drugs

    def _clean_drug_name(self, drug_name: str) -> str:
        """Cleans drug names by removing codes and extra whitespace."""
        # 1. CLEAN: Remove codes in parentheses like "Actemra (J3262)" -> "Actemra"
        clean_name = re.sub(r'\s*\([A-Z0-9.\-]+\)\s*', '', drug_name).strip()
        # 2. CLEAN: Remove leading/trailing codes or junk like "J3262 - Actemra" -> "Actemra"
        clean_name = re.sub(r'^[A-Z]\d{4,5}[^a-zA-Z]+', '', clean_name).strip()
        clean_name = re.sub(r'[^a-zA-Z]+\b[A-Z]\d{4,5}$', '', clean_name).strip()
        # 3. CLEAN: Normalize internal whitespace (e.g. "NAME    (OTHER)" -> "NAME (OTHER)")
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()
        return clean_name

    def _is_valid_drug(self, drug_name: str) -> bool:
        """Determines if a drug name is valid and not a placeholder or junk."""
        name_lower = drug_name.lower()
        placeholder_patterns = [
            r"\d+\s+specialty\s+drug",
            r"attached\s+document",
            r"linked\s+list",
            r"see\s+attached",
            r"hcpc\s+codes",
            r"various\s+medications",
            r"drug\s+list",
            r"specialty\s+drug\s+list",
            r"refer\s+to\s+attached"
        ]
        
        # Skip if drug name is just a code or too short/non-clinical
        if re.match(r'^[A-Z]\d{4,5}$', drug_name.strip()) or len(drug_name) < 3:
            return False
            
        # Skip common junk text that LLMs might hallucinate as drugs
        if any(kw in name_lower for kw in ["drug list", "specialty list", "attached", "document", "form", "provision"]):
            return False
            
        # Broad exclusion for known noise
        if "non-oncology" in name_lower or "formulary" in name_lower:
            return False
            
        # Regex filter for placeholders
        if any(re.search(pat, name_lower) for pat in placeholder_patterns):
            return False
            
        return True


    def _parse_extraction_sections(self, data: Dict) -> List[ExtractionSection]:
        """Safely parses extraction sections from LLM JSON response."""
        sections = []
        # Support both 'extraction' (new) and 'groups' (legacy/staging)
        raw_list = data.get("extraction", data.get("groups", []))
        if isinstance(raw_list, dict): 
            raw_list = [raw_list]
            
        for item in raw_list:
            try:
                # Field normalization for legacy models if LLM returns old names
                if "applies_in" in item: item["applied_in"] = item.pop("applies_in")
                if "does_not_apply_in" in item: item["does_not_applies_in"] = item.pop("does_not_applies_in")
                
                # Handle singular scenario vs plural scenarios
                sc_raw_list = item.get("scenarios", [])
                singular_scenario_dict = item.get("scenario")
                
                # Convert list of scenarios to a single one if necessary
                if not singular_scenario_dict and sc_raw_list:
                    singular_scenario_dict = self._consolidate_scenarios([RuleScenario(**s) for s in sc_raw_list]).dict() if sc_raw_list else None
                
                if singular_scenario_dict:
                    sc = singular_scenario_dict
                    if "applies_in" in sc: sc["applied_in"] = sc.pop("applies_in")
                    if "does_not_apply_in" in sc: sc["does_not_applies_in"] = sc.pop("does_not_applies_in")
                    
                    # Ensure mutual exclusivity as a safeguard
                    applied = set(sc.get("applied_in", []))
                    not_applied = set(sc.get("does_not_applies_in", []))
                    sc["does_not_applies_in"] = list(not_applied - applied)
                    
                    # --- RULE CLEANUP: Remove J-Codes, HCPCS, and Diagnosis codes ---
                    rule_text = sc.get("rule", "")
                    # Remove codes like J3262, HCPCS J3262, ICD-10 M05.1
                    rule_text = re.sub(r'\b[A-Z]\d{4,5}\b', '', rule_text) # J3262
                    rule_text = re.sub(r'\b(HCPCS|J-CODE|ICD-10|ICD)\s*[A-Z]?\d+[.\d]*\b', '', rule_text, flags=re.IGNORECASE)
                    # Clean up double spaces
                    rule_text = re.sub(r'\s+', ' ', rule_text).strip()
                    sc["rule"] = rule_text
                    
                    item["scenario"] = RuleScenario(**sc)
                
                # Clear plural field to satisfy pydantic
                if "scenarios" in item: del item["scenarios"]

                # --- V2 ADDITION: PRE-VALIDATION FILTER ---
                # Remove any drugs missing names BEFORE Pydantic instantiation to prevent crashes
                if "drugs" in item and isinstance(item["drugs"], list):
                    item["drugs"] = [d for d in item["drugs"] if isinstance(d, dict) and d.get("drug_name")]

                sec = ExtractionSection(**item)
                
                # De-duplicate and FILTER placeholder drugs
                unique_drugs = []
                seen_drugs = set()
                
                for d in sec.drugs:
                    d.drug_name = self._clean_drug_name(d.drug_name)
                    
                    if not self._is_valid_drug(d.drug_name):
                        if any(kw in d.drug_name.lower() for kw in ["drug list", "specialty drug"]):
                             print(f"      [DRUG FILTER] Removing placeholder: {d.drug_name}")
                        continue
                    
                    # De-duplication
                    key = (d.hcpcs_code or "", d.j_code or "", d.drug_name.lower())
                    if key not in seen_drugs:
                        unique_drugs.append(d)
                        seen_drugs.add(key)
                
                sec.drugs = unique_drugs
                
                if sec.scenario:
                    sections.append(sec)
            except Exception as e:
                print(f"      [SECTION SKIP] Validation error: {e}")
        return sections

    def _call_llm_json(self, prompt: str) -> Dict:
        """Helper for making structured JSON calls to Azure OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a specialized medical policy analyzer."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )
            content = response.choices[0].message.content
            if not content: return {}
            return json.loads(content)
        except Exception as e:
            print(f"    [LLM ERROR] {e}")
            return {}
