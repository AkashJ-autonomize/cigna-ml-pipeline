from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class CGMRow(BaseModel):
    """Represents a single row of CGM device data."""
    component: str = Field(..., description="The type of component (e.g., Sensor, Receiver, Transmitter)")
    brand_name: str = Field(..., description="The brand name of the device")
    ndc: Optional[str] = Field(None, description="The NDC (National Drug Code) number")
    hcpc_code: List[str] = Field(default_factory=list, description="List of associated HCPC codes")
    standard_dose: Optional[str] = Field(None, description="Standard dose or quantity per year")


class UsageMetrics(BaseModel):
    """Metrics for tracking extraction performance."""
    duration_seconds: float = 0.0
    total_table_count: int = 0
    table_row_counts: Dict[str, int] = Field(default_factory=dict, description="Row count for each categorized table")


class FinalDocumentOutput(BaseModel):
    """Final output structure for CGM extraction results."""
    document_name: str
    total_pages: int
    tables: Dict[str, List[CGMRow]] = Field(
        description="Dictionary containing 'therapeutic_table' and 'non_therapeutic_table' keys"
    )
    metrics: UsageMetrics = Field(default_factory=UsageMetrics)
