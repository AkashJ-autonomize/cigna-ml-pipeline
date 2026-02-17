from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, Field

# ============================================================================
# Stage 1: OLAM HTML Extraction Models
# ============================================================================

class Hyperlink(BaseModel):
    text: str
    url: str
    local_path: Optional[str] = None

class SectionResult(BaseModel):
    text: List[str] = Field(default_factory=list)
    hyperlinks: List[Hyperlink] = Field(default_factory=list)
    has_link: bool = False

class OLAMExtraction(BaseModel):
    medical: SectionResult = Field(default_factory=SectionResult)
    pharmacy: SectionResult = Field(default_factory=SectionResult)

# ============================================================================
# Stage 2: Generalized Drug Rule Extraction Models
# ============================================================================

class DrugItem(BaseModel):
    """Represents the specific drug and its codes"""
    drug_name: str = Field(..., description="Brand or generic name of the drug ONLY (no headers, no codes)")
    hcpcs_code: Optional[str] = Field(None, description="HCPCS code if available")
    j_code: Optional[str] = Field(None, description="J-code if available")
    original_text: Optional[str] = Field(None, description="The specific text line deriving this drug")

class RuleScenario(BaseModel):
    """
    Represents a specific routing scenario for a drug list.
    """
    rule: str = Field(..., description="The specific rule or exclusion logic for this scenario")
    routing: str = Field(..., description="Inferred routing logic (e.g. 'Medical', 'Pharmacy')")
    applied_in: List[str] = Field(default_factory=list, description="Settings where this rule APPLIES")
    does_not_applies_in: List[str] = Field(default_factory=list, description="Settings where this rule DOES NOT APPLY")

class ExtractionSection(BaseModel):
    """
    Represents a group of scenarios and drugs for a specific section (Medical/Pharmacy).
    """
    section_type: str = Field(..., description="'medical' or 'pharmacy'")
    scenario: Optional[RuleScenario] = Field(None, description="The primary rule scenario for these drugs")
    drugs: List[DrugItem] = Field(default_factory=list, description="List of drugs applying to this scenario")

class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process and policy"""
    total_drugs_found: int = 0
    total_scenarios_found: int = 0
    same_drugs_across_scenarios: bool = False
    policy_name: Optional[str] = None
    source_document: Optional[str] = None
    analysis_reasoning: Optional[str] = None
    flags: Dict[str, bool] = Field(default_factory=dict)

class RefinedExtraction(BaseModel):
    """Final output model for the pipeline"""
    client: str = Field("Unknown", description="Name of the main client")
    source_type: str = Field("source_text", description="Origin of info")
    extraction: List[ExtractionSection] = Field(default_factory=list)
    metadata: ExtractionMetadata = Field(default_factory=ExtractionMetadata)
