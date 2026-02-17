import os
import io
import base64
import pandas as pd
from typing import List, Optional
from docx import Document
from pdf2image import convert_from_path
from PIL import Image

class DocumentHandler:
    """Handles parsing of various document formats (CSV, DOCX, PDF)."""
    
    def __init__(self, openai_client, deployment_name: str):
        self.client = openai_client
        self.deployment_name = deployment_name

    def get_csv_sample(self, file_path: str) -> str:
        """Returns the first 2 rows of a CSV or Excel file as a string for LLM analysis."""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            # Read first 5 rows to ensure we catch the header if it's not on row 0
            if ext in [".xls", ".xlsx"]:
                df = pd.read_excel(file_path, nrows=5, header=None)
            else:
                df = pd.read_csv(file_path, nrows=5, header=None)
            
            # Convert to string representation
            return df.to_string()
        except Exception as e:
            return f"Error reading sample: {e}"

    def process_csv(self, file_path: str, drug_col_idx: int = 0, hcpcs_col_idx: Optional[int] = None) -> List[str]:
        """Extracts values from relevant columns of a CSV or Excel file."""
        print(f"      [CSV/EXCEL] Analyzing: {os.path.basename(file_path)}")
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".xls", ".xlsx"]:
                df = pd.read_excel(file_path, header=None)
            else:
                df = pd.read_csv(file_path, header=None)
            
            if df.empty:
                return []
            
            max_cols = len(df.columns)
            drug_idx = drug_col_idx if drug_col_idx < max_cols else 0
            
            extracted_rows = []
            print(f"      [CSV/EXCEL] Extracting drug names from column {drug_idx}")
            
            for index, row in df.iterrows():
                drug_val = str(row.iloc[drug_idx]).strip()
                
                # Check for useless content or headers
                if not drug_val or drug_val.lower() in ["nan", "null", "none", "drug name", "brand name", "name", "label"]:
                    continue
                
                entry = f"Drug: {drug_val}"
                
                # Optionally add HCPCS code if index is valid
                if hcpcs_col_idx is not None and 0 <= hcpcs_col_idx < max_cols:
                    hcpcs_val = str(row.iloc[hcpcs_col_idx]).strip()
                    if hcpcs_val and hcpcs_val.lower() not in ["nan", "null", "none", "hcpcs", "j-code", "jcode"]:
                        entry += f" | Code: {hcpcs_val}"
                
                extracted_rows.append(entry)
            
            return list(set(extracted_rows)) # Unique entries
        except Exception as e:
            print(f"      [CSV ERROR] {e}")
            return [f"Error parsing CSV: {os.path.basename(file_path)}"]

    def process_docx(self, file_path: str) -> str:
        """Extracts text from paragraphs and tables in a DOCX file."""
        print(f"      [DOCX] Extracting text from: {os.path.basename(file_path)}")
        try:
            doc = Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text)
            
            for table in doc.tables:
                for row in table.rows:
                    row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_text:
                        full_text.append(" | ".join(row_text))
            
            return "\n".join(full_text)
        except Exception as e:
            print(f"      [DOCX ERROR] {e}")
            return f"Error parsing DOCX: {os.path.basename(file_path)}"

    def process_pdf_vision(self, file_path: str) -> str:
        """Converts PDF pages to images and uses Vision model to extract drug info."""
        print(f"      [PDF] Running Vision analysis on: {os.path.basename(file_path)}")
        try:
            # Convert first 5 pages to images to save tokens/time
            images = convert_from_path(file_path, first_page=1, last_page=5)
            
            all_page_results = []
            for i, image in enumerate(images):
                # Ensure cache directory exists
                os.makedirs("cache", exist_ok=True)
                
                cache_path = f"cache/{os.path.basename(file_path)}_page_{i+1}.png"
                image.save(cache_path, "PNG")
                
                # Convert image to base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                # Call GPT-Vision
                vision_content = self._call_vision_model(img_str)
                if vision_content:
                    all_page_results.append(f"--- PAGE {i+1} ---\n{vision_content}")
            
            return "\n\n".join(all_page_results)
        except Exception as e:
            print(f"      [PDF ERROR] Failed to process {os.path.basename(file_path)} via vision: {e}")
            return f"Error parsing PDF via Vision: {os.path.basename(file_path)}"

    def _call_vision_model(self, base64_image: str) -> str:
        """Helper to call Azure OpenAI with an image."""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a specialized medical document analyzer. Extract drug names, HCPCS codes, and exclusion lists from the provided image."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this page from a medical policy. List any drugs or HCPCS codes mentioned, especially if they are part of an exclusion or 'carve-out' list."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_completion_tokens=1000
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"    [VISION ERROR] {e}")
            return ""
