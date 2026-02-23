import os
import json
import re
import time
from typing import List, Dict, Optional, Any, Tuple

from openai import AzureOpenAI

from src.schemas import (
    OLAMExtraction, ExtractionSection, RuleScenario,
    DrugItem, RefinedExtraction, ExtractionMetadata,
)
from src.prompts import (
    SOURCE_ANALYSIS_PROMPT, DETAILED_EXTRACTION_PROMPT,
    HYPERLINK_EXTRACTION_PROMPT, CSV_COLUMN_DETECTION_PROMPT,
)
from src.handlers import DocumentHandler
from src.utils import cache_download_documents


class DrugProcessor:
    """
    Orchestrates the 3-stage LLM extraction pipeline for each HTML section:
      Stage 1 — Routing:    classify whether to extract from source text, hyperlinks, or both.
      Stage 2 — Source:     extract rules and associated drugs from section text.
      Stage 3 — Hyperlink:  extract drugs from linked CSV/DOCX/PDF documents.
    Results are validated and cross-section consolidated before final output.
    """

    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.handler         = DocumentHandler(self.client, self.deployment_name)
        self._stats: Dict[str, Any] = self._fresh_stats()

    @staticmethod
    def _fresh_stats() -> Dict[str, Any]:
        """Returns a blank LLM usage stats dict for one file run."""
        return {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "duration_seconds": 0.0}

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def filter_relevant_points(self, extraction: OLAMExtraction) -> Dict[str, List[str]]:
        """Filters parsed HTML text to keep only lines with drug-relevant clinical keywords."""
        drug_keywords = {
            "prior authorization", "pre-authorization", "specialty drug", "carve-out", "j-code", "hcpcs", "injection", "infusion", "oncology", "chemotherapy", "authorization",
            "clinical criteria", "quantity limit", "step therapy", "requires approval", "medication", "pharmacy", "covered", "not covered", "exclusion", "caremark", "optum",
            "esi", "exclusion list", "carveout", "carved-out", "formularies", "self-administered", "medical benefit", "pharmacy benefit", "drug list", "specialty list",
        }
        drug_suffixes  = {"mab", "mib", "nib", "tinib", "cept"}
        hcpcs_re       = re.compile(r'\b[JQSCA][0-9]{4}\b')

        relevant = {"medical": [], "pharmacy": []}
        for section in ("medical", "pharmacy"):
            for point in getattr(extraction, section).text:
                p_lower = point.lower()
                if (any(kw in p_lower for kw in drug_keywords)
                        or any(suf in p_lower for suf in drug_suffixes)
                        or hcpcs_re.search(point)):
                    relevant[section].append(point)

        print(f"   Filter  medical={len(relevant['medical'])}  pharmacy={len(relevant['pharmacy'])}")
        return relevant

    def process_with_llm(
        self,
        relevant_points: Dict[str, List[str]],
        full_extraction: OLAMExtraction,
        client_name: str = "Unknown",
        output_dir: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> RefinedExtraction:
        """Main orchestration: runs all 3 stages, then validates and consolidates results."""
        all_sections: List[ExtractionSection] = []
        flags              = {"source_doc": False, "hyperlink": False}
        analysis_reasoning = []
        self._stats        = self._fresh_stats()   # reset per file
        t_file_start       = time.time()

        sections_to_process = [s for s in ("medical", "pharmacy") if relevant_points[s]]
        if not sections_to_process:
            metadata = ExtractionMetadata(
                analysis_reasoning="No relevant Medical or Pharmacy points detected.",
                total_drugs_found=0,
                total_scenarios_found=0,
            )
            return RefinedExtraction(client=client_name, extraction=[], metadata=metadata)

        if output_dir:
            cache_download_documents(full_extraction, output_dir, base_dir=base_dir)

        for sec in sections_to_process:
            print(f"   ┌─ {sec.upper()}")
            sec_text      = "\n".join(relevant_points[sec])
            sec_links     = getattr(full_extraction, sec).hyperlinks
            sec_links_str = self._format_links_for_prompt(sec_links)

            # ── Stage 1: Routing ──────────────────────────────────────────────
            print(f"   │  Stage 1  routing")
            router_prompt = (
                SOURCE_ANALYSIS_PROMPT.replace("{section_name}", sec)
                + f"\n\nSOURCE CONTENT ({sec}):\n{sec_text}\n\nDETECTED HYPERLINKS:\n{sec_links_str}"
            )
            routing           = self._call_llm_json(router_prompt)
            extract_source    = routing.get("extract_from_source", False)
            extract_hyperlink = routing.get("extract_from_hyperlink", False)
            if extract_source:    flags["source_doc"] = True
            if extract_hyperlink: flags["hyperlink"]  = True
            if "analysis_reasoning" in routing:
                analysis_reasoning.append(f"{sec}: {routing['analysis_reasoning']}")

            # ── Stage 2: Source Extraction ────────────────────────────────────
            current_extractions = []
            if extract_source:
                print(f"   │  Stage 2  source extraction")
                s_prompt = (
                    DETAILED_EXTRACTION_PROMPT
                    .replace("{section_name}", sec)
                    .replace("{client_name}", client_name)
                    + f"\nSOURCE CONTENT:\n{sec_text}\n\nDETECTED HYPERLINKS:\n{sec_links_str}"
                )
                s_data = self._call_llm_json(s_prompt)
                parsed = self._parse_extraction_sections(s_data)
                for ex in parsed:
                    ex.section_type = sec      # Force correct section_type
                current_extractions = parsed

            # ── Stage 3: Hyperlink Extraction ─────────────────────────────────
            if extract_hyperlink:
                # Early exit: no scenario from Stage 2 means nothing to attach hyperlink drugs to
                if not current_extractions:
                    print(f"   │  Skip     Stage 3 — no source scenario found for {sec}")
                else:
                    print(f"   │  Stage 3  hyperlink extraction")
                    hyperlink_text, pre_drugs = self._gather_hyperlink_content(sec_links)
                    h_drugs = pre_drugs

                    if hyperlink_text:
                        h_prompt = HYPERLINK_EXTRACTION_PROMPT + f"\n\nDOCUMENT CONTENT:\n{hyperlink_text[:15000]}"
                        h_data   = self._call_llm_json(h_prompt)
                        for d_raw in h_data.get("drugs", []):
                            try:
                                h_drugs.append(DrugItem(**d_raw))
                            except Exception:
                                continue

                    if h_drugs:
                        for ex in current_extractions:
                            has_link_ref = self._rule_references_external_list(ex, sec_links)
                            if has_link_ref:
                                existing = {(d.drug_name.lower(), d.hcpcs_code) for d in ex.drugs}
                                for hd in h_drugs:
                                    sig = (hd.drug_name.lower(), hd.hcpcs_code)
                                    if sig not in existing:
                                        ex.drugs.append(hd)
                                        existing.add(sig)
                                print(f"   │  Applied  {len(h_drugs)} hyperlink drugs")
                            else:
                                print(f"   │  Skip     hyperlink drugs (rule has no list reference)")

            # ── Validation ────────────────────────────────────────────────────
            valid = []
            for ex in current_extractions:
                if not ex.drugs:
                    print(f"   │  Drop     {sec} — no drugs found")
                    continue
                if not ex.scenario:
                    print(f"   │  Drop     {sec} — no scenario found")
                    continue
                is_placeholder = any(kw in ex.scenario.rule.lower() for kw in ["no specific rule", "not found", "n/a"])
                has_settings   = bool(ex.scenario.applied_in or ex.scenario.does_not_applies_in)
                has_link_kws   = any(kw in ex.scenario.rule.lower() for kw in [
                    "list", "document", "link", "attached", "provision", "refer to", "carve", "specialty", "following drugs", 
                    "esi", "express scripts", "caremark", "optum", "medication", "drug", "following medications",
                ])
                if not is_placeholder and (has_settings or len(ex.scenario.rule) > 40 or has_link_kws):
                    valid.append(ex)
                else:
                    print(f"   │  Drop     {sec} — rule too generic: '{ex.scenario.rule[:60]}'")

            total_sec_drugs = sum(len(ex.drugs) for ex in valid)
            print(f"   └─ Done    {len(valid)} section(s) · {total_sec_drugs} drugs")
            all_sections.extend(valid)

        # ── Stage 4: Cross-Section Consolidation ─────────────────────────────
        all_sections    = self._consolidate_sections_cross_section(all_sections)
        total_drugs     = sum(len(ex.drugs) for ex in all_sections)
        total_scenarios = sum(1 for ex in all_sections if ex.scenario)

        self._stats["duration_seconds"] = round(time.time() - t_file_start, 3)

        metadata = ExtractionMetadata(
            total_drugs_found=total_drugs,
            total_scenarios_found=total_scenarios,
            same_drugs_across_scenarios=(len(all_sections) == 1),
            analysis_reasoning=" | ".join(analysis_reasoning),
            flags=flags,
            llm_stats=dict(self._stats),
        )
        if all_sections and not metadata.policy_name:
            metadata.policy_name = f"Extracted Rules ({total_drugs} drugs)"

        return RefinedExtraction(
            client=client_name,
            source_type="source_text",
            extraction=all_sections,
            metadata=metadata,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Stage Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _rule_references_external_list(self, ex: ExtractionSection, sec_links: List[Any]) -> bool:
        """
        Returns True if the scenario rule contains carve-out / PBM / list keywords,
        or if the router already confirmed hyperlink relevance (sec_links is non-empty).
        """
        if not ex.scenario:
            return bool(sec_links)     # Fallback: router confirmed hyperlink relevance

        rule_lower = ex.scenario.rule.lower()
        keywords   = [
            "list", "document", "link", "attached", "provision", "refer to", "carve", "carve-out", "carved", "specialty", "following drugs", "drug",
            "following medications", "the following", "below", "xlsx", "csv", "esi", "express scripts", "caremark", "optum", "pbm", "medication", 
        ]
        return any(kw in rule_lower for kw in keywords) or bool(sec_links)

    def _format_links_for_prompt(self, hyperlinks: List[Any]) -> str:
        valid_exts = {".pdf", ".csv", ".xlsx", ".xls", ".docx", ".doc"}
        lines = [
            f"- {l.url} (Context: '{l.text}')"
            for l in hyperlinks
            if os.path.splitext(l.url)[1].lower() in valid_exts
        ]
        return "\n".join(lines) if lines else "None relevant."

    def _gather_hyperlink_content(self, hyperlinks: List[Any]) -> Tuple[str, List[DrugItem]]:
        """Reads downloaded document files and returns (text_content, pre_extracted_drugs)."""
        contents, pre_drugs = [], []

        for link in hyperlinks:
            if not (link.local_path and os.path.exists(link.local_path)):
                continue

            ext  = os.path.splitext(link.local_path)[1].lower()
            name = os.path.basename(link.local_path)

            try:
                if ext == ".txt":
                    with open(link.local_path, "r", encoding="utf-8", errors="ignore") as f:
                        contents.append(f"--- FILE: {name} ---\n{f.read()[:5000]}")

                elif ext in (".csv", ".xls", ".xlsx"):
                    sample  = self.handler.get_csv_sample(link.local_path)
                    mapping = self._call_llm_json(CSV_COLUMN_DETECTION_PROMPT.replace("{csv_sample}", sample))
                    sheets  = mapping.get("sheets", [])

                    # Fallback: old single-sheet response format (backward-compat)
                    if not sheets:
                        sheets = [{
                            "sheet_name":      None,
                            "is_drug_list":    True,
                            "drug_name_index": mapping.get("drug_name_index", 0),
                            "hcpcs_code_index": mapping.get("hcpcs_code_index"),
                        }]

                    seen_keys    = {(d.drug_name.lower(), d.hcpcs_code) for d in pre_drugs}
                    sheets_added = 0

                    for sheet_info in sheets:
                        if not sheet_info.get("is_drug_list", False):
                            continue

                        s_name    = sheet_info.get("sheet_name")
                        drug_idx  = sheet_info.get("drug_name_index", 0)
                        hcpcs_idx = sheet_info.get("hcpcs_code_index")

                        if not isinstance(drug_idx, int)  or drug_idx  < 0: drug_idx  = 0
                        if not isinstance(hcpcs_idx, int) or hcpcs_idx < 0: hcpcs_idx = None

                        rows = self.handler.process_csv(link.local_path, drug_col_idx=drug_idx, hcpcs_col_idx=hcpcs_idx, sheet_name=s_name)
                        added = 0
                        for r in rows:
                            parts  = r.split(" | ")
                            d_name = self._clean_drug_name(parts[0].replace("Drug: ", "").strip())
                            c_code = parts[1].replace("Code: ", "").strip() if len(parts) > 1 else None
                            key    = (d_name.lower(), c_code)
                            if d_name and self._is_valid_drug(d_name) and key not in seen_keys:
                                pre_drugs.append(DrugItem(drug_name=d_name, hcpcs_code=c_code, j_code=c_code))
                                seen_keys.add(key)
                                added += 1
                        sheets_added += added
                        tab_label = f" [{s_name}]" if s_name else ""
                        print(f"   │    CSV    {added} drugs ← col {drug_idx}{tab_label}")

                    print(f"   │    CSV    {sheets_added} new drug(s) total from {name}")


                elif ext in (".docx", ".doc"):
                    text = self.handler.process_docx(link.local_path)
                    contents.append(f"--- FILE: {name} ---\n{text[:10000]}")

                elif ext == ".pdf":
                    vision_text = self.handler.process_pdf_vision(link.local_path, stats=self._stats)
                    contents.append(f"--- FILE: {name} ---\n{vision_text}")

            except Exception as e:
                print(f"   │  Error    Failed to read {name}: {e}")

        return "\n\n".join(contents), pre_drugs

    # ─────────────────────────────────────────────────────────────────────────
    # Data Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _clean_drug_name(self, drug_name: str) -> str:
        """Strips HCPCS codes and normalises whitespace from a drug name string."""
        # Remove single-token parentheticals (codes like J3262); keeps multi-word ones like "(ALL FORMS)"
        name = re.sub(r'\s*\([^ ]+\)\s*', '', drug_name).strip()
        # Remove bare leading/trailing HCPCS codes
        name = re.sub(r'^[A-Z]\d{4,5}[^a-zA-Z]+', '', name).strip()
        name = re.sub(r'[^a-zA-Z]+\b[A-Z]\d{4,5}$', '', name).strip()
        return re.sub(r'\s+', ' ', name).strip()

    def _is_valid_drug(self, drug_name: str) -> bool:
        """Returns True if the string is a real drug name, not a header, code, or placeholder."""
        name_lower = drug_name.lower()

        # Patterns that indicate spreadsheet metadata rows or document placeholders
        junk_patterns = [
            r"drug\s+name", r"label\s+name", r"short\s+name", r"description",
            r"effective\s+date", r"revision\s+date", r"page\s+\d+", r"footer",
            r"cigna\s+confidential", r"proprietary", r"not\s+a\s+covered",
            r"billing\s+code", r"hcpcs\s+code", r"j-code", r"modifier",
            r"dosage", r"strength", r"route", r"pkg\s+size", r"national\s+drug\s+code",
            r"\d+\s+specialty\s+drug", r"attached\s+document", r"linked\s+list",
            r"see\s+attached", r"hcpc\s+codes", r"various\s+medications",
            r"drug\s+list", r"specialty\s+drug\s+list", r"refer\s+to\s+attached",
        ]
        # Word-boundary exclusions — avoids false positives (e.g. "form" vs "(ALL FORMS)")
        exclusion_keywords = [
            r"\bform\b", r"\bdocument\b", r"\bdrug\s+list\b", r"\bspecialty\s+list\b",
            r"\battached\b", r"\bprovision\b", r"\bappendix\b", r"\bexhibit\b",
        ]

        if re.match(r'^[A-Z]\d{4,5}$', drug_name.strip()) or len(drug_name) < 3:
            return False
        if any(re.search(p, name_lower) for p in junk_patterns):
            return False
        if any(re.search(p, name_lower) for p in exclusion_keywords):
            return False
        if "non-oncology" in name_lower or "formulary" in name_lower:
            return False
        return True

    def _parse_extraction_sections(self, data: Dict) -> List[ExtractionSection]:
        """Parses and validates raw LLM JSON into typed ExtractionSection objects."""
        sections = []
        raw_list = data.get("extraction", data.get("groups", []))
        if isinstance(raw_list, dict):
            raw_list = [raw_list]

        for item in raw_list:
            try:
                # Normalise legacy field aliases
                for old, new in [("applies_in", "applied_in"), ("does_not_apply_in", "does_not_applies_in")]:
                    if old in item:
                        item[new] = item.pop(old)

                # Resolve singular scenario from either 'scenario' or 'scenarios' list
                sc_dict = item.get("scenario")
                if not sc_dict and item.get("scenarios"):
                    sc_dict = self._consolidate_scenarios(
                        [RuleScenario(**s) for s in item["scenarios"]]
                    ).dict()
                item.pop("scenarios", None)

                if sc_dict:
                    for old, new in [("applies_in", "applied_in"), ("does_not_apply_in", "does_not_applies_in")]:
                        if old in sc_dict:
                            sc_dict[new] = sc_dict.pop(old)

                    # Ensure applied_in / does_not_applies_in are mutually exclusive
                    applied     = set(sc_dict.get("applied_in", []))
                    not_applied = set(sc_dict.get("does_not_applies_in", []))
                    sc_dict["does_not_applies_in"] = list(not_applied - applied)

                    # Strip raw HCPCS / ICD codes from the rule text
                    rule = sc_dict.get("rule", "")
                    rule = re.sub(r'\b[A-Z]\d{4,5}\b', '', rule)
                    rule = re.sub(r'\b(HCPCS|J-CODE|ICD-10|ICD)\s*[A-Z]?\d+[.\d]*\b', '', rule, flags=re.IGNORECASE)
                    sc_dict["rule"] = re.sub(r'\s+', ' ', rule).strip()
                    item["scenario"] = RuleScenario(**sc_dict)

                # Pre-filter drug entries to prevent Pydantic errors on missing names
                if isinstance(item.get("drugs"), list):
                    item["drugs"] = [d for d in item["drugs"] if isinstance(d, dict) and d.get("drug_name")]

                sec = ExtractionSection(**item)

                # De-duplicate and validate drugs
                unique_drugs, seen = [], set()
                for d in sec.drugs:
                    d.drug_name = self._clean_drug_name(d.drug_name)
                    if not self._is_valid_drug(d.drug_name):
                        continue
                    key = (d.hcpcs_code or "", d.j_code or "", d.drug_name.lower())
                    if key not in seen:
                        unique_drugs.append(d)
                        seen.add(key)
                sec.drugs = unique_drugs

                if sec.scenario:
                    sections.append(sec)

            except Exception as e:
                print(f"   │  Skip     section parse error: {e}")

        return sections

    def _consolidate_scenarios(self, scenarios: List[RuleScenario]) -> Optional[RuleScenario]:
        """Merges a list of RuleScenarios into a single representative scenario."""
        if not scenarios:
            return None
        base = scenarios[0]
        for s in scenarios[1:]:
            if s.rule.lower().strip() not in base.rule.lower().strip():
                base.rule = f"{base.rule} | {s.rule}"
            base.applied_in          = list(set(base.applied_in + s.applied_in))
            base.does_not_applies_in = list(set(base.does_not_applies_in + s.does_not_applies_in))
        return base

    def _consolidate_sections_cross_section(self, sections: List[ExtractionSection]) -> List[ExtractionSection]:
        """
        Merges ExtractionSections that share a significant overlap of drugs or HCPCS codes.
        Only merges within the same section_type (medical / pharmacy).
        """
        if not sections:
            return []

        def normalize(name: str) -> str:
            return " ".join(sorted(re.sub(r'[^a-zA-Z0-9 ]', ' ', name).lower().split()))

        processed = [
            {
                "sec":        s,
                "name_words": {w for d in s.drugs for w in normalize(d.drug_name).split()},
                "codes":      {d.hcpcs_code for d in s.drugs if d.hcpcs_code} | {d.j_code for d in s.drugs if d.j_code},
                "merged":     False,
            }
            for s in sections
        ]

        final = []
        for i, base in enumerate(processed):
            if base["merged"]:
                continue
            base["merged"] = True

            for target in processed[i + 1:]:
                if target["merged"] or base["sec"].section_type != target["sec"].section_type:
                    continue

                match = False
                if base["codes"] and target["codes"]:
                    shared = base["codes"] & target["codes"]
                    match  = len(shared) / max(1, min(len(base["codes"]), len(target["codes"]))) > 0.5
                elif base["name_words"] and target["name_words"]:
                    shared = base["name_words"] & target["name_words"]
                    match  = len(shared) / max(1, min(len(base["name_words"]), len(target["name_words"]))) > 0.7

                if not match:
                    continue

                target["merged"] = True

                # Merge scenarios
                if target["sec"].scenario and not base["sec"].scenario:
                    base["sec"].scenario = target["sec"].scenario
                elif target["sec"].scenario and base["sec"].scenario:
                    t_rule = target["sec"].scenario.rule.lower().strip()
                    if t_rule not in base["sec"].scenario.rule.lower().strip():
                        base["sec"].scenario.rule = f"{base['sec'].scenario.rule} | {target['sec'].scenario.rule}"
                    base["sec"].scenario.applied_in = list(
                        set(base["sec"].scenario.applied_in + target["sec"].scenario.applied_in)
                    )
                    base["sec"].scenario.does_not_applies_in = list(
                        set(base["sec"].scenario.does_not_applies_in + target["sec"].scenario.does_not_applies_in)
                    )

                # Union drugs
                existing = {(d.hcpcs_code, d.j_code, d.drug_name.lower()) for d in base["sec"].drugs}
                for d in target["sec"].drugs:
                    key = (d.hcpcs_code, d.j_code, d.drug_name.lower())
                    if key not in existing:
                        base["sec"].drugs.append(d)
                        existing.add(key)

            final.append(base["sec"])

        return final

    # ─────────────────────────────────────────────────────────────────────────
    # LLM Call
    # ─────────────────────────────────────────────────────────────────────────

    def _call_llm_json(self, prompt: str) -> Dict:
        """Calls Azure OpenAI with JSON mode and returns the parsed response dict."""
        try:
            t0       = time.time()
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a specialized medical policy analyzer. Always respond with valid JSON only."},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            elapsed = time.time() - t0
            if hasattr(response, "usage") and response.usage:
                self._stats["llm_calls"]        += 1
                self._stats["input_tokens"]     += response.usage.prompt_tokens
                self._stats["output_tokens"]    += response.usage.completion_tokens
                self._stats["total_tokens"]     += response.usage.total_tokens
                self._stats["duration_seconds"]  = round(self._stats["duration_seconds"] + elapsed, 3)
            content = response.choices[0].message.content
            return json.loads(content) if content else {}
        except Exception as e:
            print(f"   │  LLM Error  {e}")
            return {}