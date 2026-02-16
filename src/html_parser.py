import os
import json
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from src.schemas import CGMRow, FinalDocumentOutput, UsageMetrics

class HTMLTableParser:
    def __init__(self):
        self.headers_mapping = {
            "component": "component",
            "brand name": "brand_name",
            "ndc": "ndc",
            "hcpc code": "hcpc_code",
            "standard dose": "standard_dose"
        }

    def parse_file(self, file_path: str) -> Dict[str, List[CGMRow]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        results = {
            "therapeutic_table": [],
            "non_therapeutic_table": []
        }

        # Find therapeutic table
        # Using a more robust search that handles non-breaking spaces and case sensitivity
        therapeutic_heading = soup.find(lambda tag: tag.name in ["p", "strong", "b", "span", "td"] and 
                                      "therapeutic" in tag.get_text().lower() and 
                                      "non-therapeutic" not in tag.get_text().lower() and
                                      "which components are" in tag.get_text().lower())
        if therapeutic_heading:
            table = therapeutic_heading.find_next("table")
            if table:
                results["therapeutic_table"] = self._extract_table_data(table)

        # Find non-therapeutic table
        non_therapeutic_heading = soup.find(lambda tag: tag.name in ["p", "strong", "b", "span", "td"] and 
                                          "non-therapeutic" in tag.get_text().lower() and
                                          "which components are" in tag.get_text().lower())
        if non_therapeutic_heading:
            table = non_therapeutic_heading.find_next("table")
            if table:
                results["non_therapeutic_table"] = self._extract_table_data(table)

        return results

    def _extract_table_data(self, table) -> List[CGMRow]:
        rows = table.find_all("tr")
        if not rows:
            return []

        # Get headers from first row
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
                # Clean up non-breaking spaces and redundant whitespaces/newlines
                val = val.replace("\u00a0", " ").replace("\u200b", "").strip()
                val = " ".join(val.split()) # Normalize whitespace
                
                if attr == "hcpc_code":
                    # Split by comma and clean
                    data[attr] = [code.strip() for code in val.split(",") if code.strip()]
                else:
                    data[attr] = val

            try:
                extracted_data.append(CGMRow(**data))
            except Exception as e:
                print(f"Error parsing row: {e}")
                print(f"Row data: {data}")

        return extracted_data

def process_html_document(file_path: str) -> FinalDocumentOutput:
    parser = HTMLTableParser()
    extracted_tables = parser.parse_file(file_path)
    
    # Calculate metrics
    total_rows = sum(len(rows) for rows in extracted_tables.values())
    table_row_counts = {k: len(v) for k, v in extracted_tables.items()}
    
    metrics = UsageMetrics(
        total_table_count=2,
        table_row_counts=table_row_counts,
        duration_seconds=0.0 # Placeholder
    )
    
    return FinalDocumentOutput(
        document_name=os.path.basename(file_path),
        total_pages=1, # HTML is usually 1 doc
        tables=extracted_tables,
        metrics=metrics
    )
