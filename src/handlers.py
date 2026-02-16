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

    def process_csv(self, file_path: str) -> List[str]:
        """Extracts values from the first column of a CSV or Excel file."""
        print(f"      [CSV] Extracting column[0] from: {os.path.basename(file_path)}")
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".xls", ".xlsx"]:
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
            
            if df.empty:
                return []
            
            # Extract first column and drop NaNs
            col0_values = df.iloc[:, 0].dropna().astype(str).unique().tolist()
            return [f"Drug/Code: {v}" for v in col0_values if v.strip()]
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
                # Save to cache for debugging
                if not os.path.exists("cache"):
                    os.makedirs("cache")
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
            print(f"      [PDF ERROR] {e}")
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
            return response.choices[0].message.content
        except Exception as e:
            print(f"    [VISION ERROR] {e}")
            return ""
