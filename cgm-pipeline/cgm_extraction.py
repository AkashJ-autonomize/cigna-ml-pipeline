import os
import json
import time
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

# =================================================================
# Data Models
# =================================================================

class CGMRow(BaseModel):
    """Schema for a single row in the CGM extraction tables."""
    component: str = Field(..., description="The type of component (e.g., Sensor, Receiver)")
    brand_name: str = Field(..., description="The brand name of the device")
    ndc: Optional[str] = Field(None, description="The NDC number")
    hcpc_code: List[str] = Field(default_factory=list, description="List of associated HCPC codes")
    standard_dose: Optional[str] = Field(None, description="Standard dose or quantity per year")

class UsageMetrics(BaseModel):
    """Schema for processing metrics."""
    duration_seconds: float = 0.0
    total_table_count: int = 0
    table_row_counts: Dict[str, int] = Field(default_factory=dict)

class FinalDocumentOutput(BaseModel):
    """Final structure for the extracted document data."""
    document_name: str
    total_pages: int = 1
    tables: Dict[str, List[CGMRow]]
    metrics: UsageMetrics = Field(default_factory=UsageMetrics)

# =================================================================
# HTML Parser Logic
# =================================================================

class HTMLTableParser:
    """Handles the identification and extraction of CGM tables from HTML content."""
    
    def __init__(self):
        self.headers_mapping = {
            "component": "component",
            "brand name": "brand_name",
            "ndc": "ndc",
            "hcpc code": "hcpc_code",
            "standard dose": "standard_dose"
        }

    def parse_file(self, file_path: str) -> Dict[str, List[CGMRow]]:
        """Parses an HTML file to extract therapeutic and non-therapeutic tables."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        results = {
            "therapeutic_table": [],
            "non_therapeutic_table": []
        }

        # Strategy: Find headings containing 'therapeutic' and 'which components are'
        def find_heading(pattern_include: str, pattern_exclude: str = None):
            return soup.find(lambda tag: tag.name in ["p", "strong", "b", "span", "td"] and 
                           pattern_include in tag.get_text().lower() and 
                           (pattern_exclude is None or pattern_exclude not in tag.get_text().lower()) and
                           "which components are" in tag.get_text().lower())

        # Extract Therapeutic Table
        therapeutic_heading = find_heading("therapeutic", "non-therapeutic")
        if therapeutic_heading:
            table = therapeutic_heading.find_next("table")
            if table:
                results["therapeutic_table"] = self._extract_table_data(table)

        # Extract Non-Therapeutic Table
        non_therapeutic_heading = find_heading("non-therapeutic")
        if non_therapeutic_heading:
            table = non_therapeutic_heading.find_next("table")
            if table:
                results["non_therapeutic_table"] = self._extract_table_data(table)

        return results

    def _extract_table_data(self, table) -> List[CGMRow]:
        """Extracts and validates rows from a specific HTML table."""
        rows = table.find_all("tr")
        if not rows:
            return []

        # Map headers to schema fields
        header_row = rows[0]
        headers = [td.get_text(strip=True).lower() for td in header_row.find_all(["td", "th"])]
        
        column_indices = {}
        for idx, h in enumerate(headers):
            for key, attr in self.headers_mapping.items():
                if key in h:
                    column_indices[attr] = idx
                    break

        extracted_data = []
        for row in rows[1:]:
            cells = row.find_all("td")
            if len(cells) < len(column_indices):
                continue
            
            data = {}
            for attr, idx in column_indices.items():
                val = cells[idx].get_text(strip=True)
                # Clean and normalize whitespace
                val = val.replace("\u00a0", " ").replace("\u200b", "").strip()
                val = " ".join(val.split())
                
                if attr == "hcpc_code":
                    # Parse comma-separated codes into a list
                    data[attr] = [code.strip() for code in val.split(",") if code.strip()]
                else:
                    data[attr] = val

            try:
                extracted_data.append(CGMRow(**data))
            except Exception as e:
                print(f"[Warning] Error parsing row: {e}")

        return extracted_data

# =================================================================
# Execution Logic
# =================================================================

def process_document(file_path: str) -> FinalDocumentOutput:
    """Coordinates the parsing and metric calculation for a single HTML document."""
    parser = HTMLTableParser()
    extracted_tables = parser.parse_file(file_path)
    
    table_row_counts = {k: len(v) for k, v in extracted_tables.items()}
    
    metrics = UsageMetrics(
        total_table_count=len([v for v in extracted_tables.values() if v]),
        table_row_counts=table_row_counts
    )
    
    return FinalDocumentOutput(
        document_name=os.path.basename(file_path),
        tables=extracted_tables,
        metrics=metrics
    )

def main():
    """Main entry point to process all HTML files in the input directory."""
    input_dir = "files"
    output_dir = "output"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    html_files = [f for f in os.listdir(input_dir) if f.endswith(".html")]
    
    if not html_files:
        print(f"No HTML files found in '{input_dir}' directory.")
        return

    for html_file in html_files:
        print(f"[*] Processing {html_file}...")
        start_time = time.time()
        file_path = os.path.join(input_dir, html_file)
        
        try:
            result = process_document(file_path)
            result.metrics.duration_seconds = time.time() - start_time
            
            # Save results to output directory
            output_name = html_file.replace(".html", ".json")
            output_path = os.path.join(output_dir, output_name)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result.model_dump(), f, indent=4)
                
            print(f"[+] Success: Results saved to {output_path}")
            
        except Exception as e:
            print(f"[!] Error processing {html_file}: {e}")

if __name__ == "__main__":
    main()
