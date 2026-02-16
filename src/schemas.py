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

class PriorAuthBySetting(BaseModel):
    """Setting-specific prior authorization requirements"""
    physician_office: Optional[bool] = None
    home_health_care: Optional[bool] = None
    inpatient_hospital: Optional[bool] = None
    outpatient_hospital: Optional[bool] = None
    emergency_room: Optional[bool] = None
    urgent_care: Optional[bool] = None

class PharmacyBenefitBySetting(BaseModel):
    """Setting-specific pharmacy benefit plan requirements"""
    physician_office: Optional[bool] = None
    home_health_care: Optional[bool] = None
    inpatient_hospital: Optional[bool] = None
    outpatient_hospital: Optional[bool] = None
    emergency_room: Optional[bool] = None
    urgent_care: Optional[bool] = None

class Scenario(BaseModel):
    """Detailed scenario describing drug administration rules for a specific setting"""
    setting: Optional[str] = Field(None, description="Care setting (e.g., physician_office, inpatient_hospital)")
    routing: Optional[str] = Field(None, description="Where to obtain the drug (e.g., 'BJC Pharmacy Benefit Plan', 'Direct facility dispensing')")
    benefit_type: Optional[str] = Field(None, description="Type of benefit: 'pharmacy' or 'medical'")
    prior_auth: Optional[bool] = Field(None, description="Whether prior authorization is required in this setting")
    notes: Optional[str] = Field(None, description="Additional context or exceptions for this scenario")

class DrugRule(BaseModel):
    """Comprehensive drug rule with setting-specific requirements and scenarios"""
    drug_name: str = Field(..., description="Brand or generic name of the drug")
    hcpcs_code: Optional[str] = Field(None, description="HCPCS code if available")
    j_code: Optional[str] = Field(None, description="J-code if available")
    classification: str = Field(..., description="Drug classification (e.g., 'Specialty Drug - Carve-Out', 'Medical', 'Pharmacy')")
    
    # Setting-specific fields (optional for flexibility)
    prior_auth_required: Optional[Union[bool, PriorAuthBySetting]] = Field(
        None, 
        description="Prior auth requirement: simple boolean or setting-specific object"
    )
    pharmacy_benefit_plan_required: Optional[PharmacyBenefitBySetting] = Field(
        None,
        description="Whether pharmacy benefit plan is required by setting"
    )
    
    # Summary and scenarios
    summary: str = Field(..., description="Consolidated summary of all rules across settings")
    scenarios: List[Scenario] = Field(
        default_factory=list,
        description="Detailed scenarios for each care setting"
    )

class PolicyMetadata(BaseModel):
    """Policy-level metadata about the drug rules"""
    total_carve_out_drugs: Optional[int] = Field(None, description="Total number of drugs in carve-out list")
    source_document: Optional[str] = Field(None, description="Path to source document")
    policy_name: Optional[str] = Field(None, description="Name of the policy")
    carve_out_applies_to: Optional[List[str]] = Field(None, description="Settings where carve-out applies")
    carve_out_excluded_from: Optional[List[str]] = Field(None, description="Settings excluded from carve-out")
    client_name: Optional[str] = Field(None, description="Client name")
    specialty_pharmacy_vendor: Optional[Union[str, List[str]]] = Field(None, description="Specialty pharmacy vendor name(s)")
    additional_info: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Any additional metadata")

class RefinedExtraction(BaseModel):
    """Final extraction result with drug rules and metadata"""
    drug_rules: List[DrugRule] = Field(default_factory=list)
    excluded_drug_list_in_hyperlink: bool = False
    excluded_drug_list_in_source_doc: bool = False
    metadata: PolicyMetadata = Field(default_factory=PolicyMetadata)
