from typing import List, Dict, Optional, Any
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
    """Parsed output from an OLAM HTML file (Medical + Pharmacy sections)."""
    medical:  SectionResult = Field(default_factory=SectionResult)
    pharmacy: SectionResult = Field(default_factory=SectionResult)


# ============================================================================
# Stage 2: Generalized Drug Rule Extraction Models
# ============================================================================


class DrugItem(BaseModel):
    drug_name:     str           = Field(..., description="Brand or generic drug name (no codes, no headers)")
    hcpcs_code:    Optional[str] = None
    j_code:        Optional[str] = None
    original_text: Optional[str] = None


class RuleScenario(BaseModel):
    rule:                str       = Field(..., description="The coverage/exclusion rule for this drug group")
    routing:             str       = Field(..., description="Routing logic (e.g. 'Medical', 'Pharmacy')")
    applied_in:          List[str] = Field(default_factory=list, description="Settings where this rule applies")
    does_not_applies_in: List[str] = Field(default_factory=list, description="Settings where this rule does NOT apply")


class ExtractionSection(BaseModel):
    section_type: str                    = Field(..., description="'medical' or 'pharmacy'")
    scenario:     Optional[RuleScenario] = None
    drugs:        List[DrugItem]         = Field(default_factory=list)


class ExtractionMetadata(BaseModel):
    total_drugs_found:           int             = 0
    total_scenarios_found:       int             = 0
    same_drugs_across_scenarios: bool            = False
    policy_name:                 Optional[str]   = None
    source_document:             Optional[str]   = None
    analysis_reasoning:          Optional[str]   = None
    flags:                       Dict[str, bool] = Field(default_factory=dict)
    llm_stats:                   Dict[str, Any]  = Field(default_factory=dict)


class RefinedExtraction(BaseModel):
    """Final pipeline output for a single client file."""
    client:      str                    = Field("Unknown")
    source_type: str                    = Field("source_text")
    extraction:  List[ExtractionSection] = Field(default_factory=list)
    metadata:    ExtractionMetadata     = Field(default_factory=ExtractionMetadata)
