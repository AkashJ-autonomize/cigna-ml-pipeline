import os
import json
import re
from typing import List, Dict, Optional
from openai import AzureOpenAI
from src.schemas import OLAMExtraction, DrugRule, RefinedExtraction
from src.prompts import SOURCE_ANALYSIS_PROMPT, DETAILED_EXTRACTION_PROMPT, HYPERLINK_EXTRACTION_PROMPT
from src.handlers import DocumentHandler
from src.utils import lazy_download_documents, build_context_string

class DrugProcessor:
    """
    Orchestrates the filtering of parsed text and the extraction of drug rules using LLMs.
    Supports 3-stage analysis: 1. Classification, 2. Source Extraction, 3. Hyperlink Extraction.
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
        """Filters the raw parsed text for high-signal clinical drug keywords."""
        drug_keywords = {
            "prior authorization", "pre-authorization", "specialty drug", "carve-out",
            "j-code", "hcpcs", "injection", "infusion", "oncology", "chemotherapy",
            "clinical criteria", "quantity limit", "step therapy", "requires approval",
            "medication", "pharmacy", "clinic", "covered", "not covered", "exclusion",
            "esi", "special handling", "exclusion list", "carveout", "carved-out", "formularies"
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

    def process_with_llm(self, relevant_points: Dict[str, List[str]], full_extraction: OLAMExtraction, output_dir: Optional[str] = None) -> RefinedExtraction:
        """Main 3-Stage Orchestration Flow."""
        if not relevant_points["medical"] and not relevant_points["pharmacy"]:
            return RefinedExtraction()

        text_context = build_context_string(relevant_points)
        links_str = self._get_hyperlink_metadata(full_extraction)

        # --- STAGE 1: Classification ---
        print("    [STAGE 1] Classifying content source...")
        analysis_prompt = SOURCE_ANALYSIS_PROMPT + f"\n\nSTRUCTURAL HYPERLINKS (METADATA):\n{links_str}\n\nCONTENT:\n{text_context}"
        analysis_data = self._call_llm_json(analysis_prompt)
        
        refined = RefinedExtraction(
            excluded_drug_list_in_hyperlink=analysis_data.get("excluded_drug_list_in_hyperlink", False),
            excluded_drug_list_in_source_doc=analysis_data.get("excluded_drug_list_in_source_doc", False)
        )
        # Store analysis reasoning in additional_info
        refined.metadata.additional_info["analysis_reasoning"] = analysis_data.get("analysis_reasoning", "No reasoning provided.")
        
        # --- STAGE 2: Source Extraction (Conditional) ---
        if refined.excluded_drug_list_in_source_doc:
            print("    [STAGE 2] Extracting granular rules from source text...")
            e_prompt = DETAILED_EXTRACTION_PROMPT + f"\nCONTENT:\n{text_context}"
            e_data = self._call_llm_json(e_prompt)
            refined.drug_rules.extend(self._parse_drug_rules(e_data))

        # --- STAGE 3: Hyperlink Extraction (Conditional) ---
        if refined.excluded_drug_list_in_hyperlink:
            print("    [STAGE 3] Extracting granular rules from hyperlinks...")
            if output_dir:
                lazy_download_documents(full_extraction, output_dir)
            
            h_content = self._gather_hyperlink_content(full_extraction)
            if h_content:
                h_prompt = HYPERLINK_EXTRACTION_PROMPT + f"\nCONTENT FROM DOCUMENTS:\n{h_content}"
                h_data = self._call_llm_json(h_prompt)
                refined.drug_rules.extend(self._parse_drug_rules(h_data))
        
        # --- STAGE 4: Metadata Extraction (Optional) ---
        if refined.drug_rules:
            print("    [STAGE 4] Extracting policy metadata...")
            from src.prompts import METADATA_EXTRACTION_PROMPT
            
            # Combine all available context for metadata extraction
            full_context = f"SOURCE TEXT:\n{text_context}\n\nHYPERLINKS:\n{links_str}"
            if refined.excluded_drug_list_in_hyperlink and h_content:
                full_context += f"\n\nDOCUMENT CONTENT:\n{h_content[:3000]}"
            
            meta_prompt = METADATA_EXTRACTION_PROMPT + f"\n\nCONTENT:\n{full_context}"
            meta_data = self._call_llm_json(meta_prompt)
            
            # Update metadata fields from LLM response
            if meta_data:
                from src.schemas import PolicyMetadata
                try:
                    extracted_meta = PolicyMetadata(**meta_data)
                    # Merge with existing metadata
                    for field, value in extracted_meta.model_dump(exclude_none=True).items():
                        if value and field != "additional_info":
                            setattr(refined.metadata, field, value)
                    # Merge additional_info
                    if extracted_meta.additional_info:
                        refined.metadata.additional_info.update(extracted_meta.additional_info)
                except Exception as e:
                    print(f"      [METADATA SKIP] {e}")
            
        print(f"    [ORCHESTRATION] Flags: Source={refined.excluded_drug_list_in_source_doc}, Hyperlink={refined.excluded_drug_list_in_hyperlink}")
        print(f"    [LLM] Total Consolidated Rules: {len(refined.drug_rules)}")
        return refined

    def _get_hyperlink_metadata(self, extraction: OLAMExtraction) -> str:
        """Gathers basic metadata about detected links for Stage 1 classification."""
        valid_extensions = {".pdf", ".csv", ".xlsx", ".xls", ".docx", ".doc"}
        links = []
        for l in (extraction.medical.hyperlinks + extraction.pharmacy.hyperlinks):
            if os.path.splitext(l.url)[1].lower() in valid_extensions:
                links.append(f"- URL: {l.url} (Context: {l.text})")
        return "\n".join(links) if links else "None relevant detected (PDF/CSV/XLSX/DOCX only)."

    def _gather_hyperlink_content(self, extraction: OLAMExtraction) -> str:
        """Reads text from downloaded documents associated with hyperlinks."""
        contents = []
        all_links = extraction.medical.hyperlinks + extraction.pharmacy.hyperlinks
        for link in all_links:
            if not (link.local_path and os.path.exists(link.local_path)):
                continue
                
            ext = os.path.splitext(link.local_path)[1].lower()
            file_name = os.path.basename(link.local_path)
            
            if ext == ".txt":
                with open(link.local_path, 'r', encoding='utf-8', errors='ignore') as f:
                    contents.append(f"FILE: {file_name}\n{f.read()[:5000]}")
            elif ext in [".csv", ".xls", ".xlsx"]:
                rows = self.handler.process_csv(link.local_path)
                contents.append(f"FILE: {file_name}\n" + "\n".join(rows))
            elif ext in [".docx", ".doc"]:
                text = self.handler.process_docx(link.local_path)
                contents.append(f"FILE: {file_name}\n{text[:5000]}")
            elif ext == ".pdf":
                vision_text = self.handler.process_pdf_vision(link.local_path)
                contents.append(f"FILE: {file_name}\nVISION EXTRACTION:\n{vision_text}")
        return "\n\n".join(contents)

    def _parse_drug_rules(self, data: Dict) -> List[DrugRule]:
        """Safely parses a list of drug rules from LLM JSON response."""
        rules = []
        for r in data.get("drug_rules", []):
            try:
                rules.append(DrugRule(**r))
            except Exception as e:
                print(f"      [RULE SKIP] {e}")
        return rules

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
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"    [LLM ERROR] {e}")
            return {}
