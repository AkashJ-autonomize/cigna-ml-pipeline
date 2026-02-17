import os
import shutil
import re
from typing import List, Dict
from src.schemas import OLAMExtraction

def lazy_download_documents(extraction: OLAMExtraction, output_dir: str):
    """
    Copies referenced local documents to output dir only when needed.
    Sanitizes filenames and truncates long paths to avoid OS limits.
    """
    fetched_dir = os.path.join(output_dir, "fetched_files")
    os.makedirs(fetched_dir, exist_ok=True)
        
    all_docs = extraction.medical.hyperlinks + extraction.pharmacy.hyperlinks
    base_proj_dir = os.getcwd() 
    
    print(f"      [FETCH] Checking {len(all_docs)} hyperlinks...")
    
    for doc in all_docs:
        # Skip if already downloaded/fetched
        if doc.local_path and os.path.exists(doc.local_path):
            continue 
            
        local_src = None
        # Possible locations for the file relative to base project
        search_paths = [
            os.path.join(base_proj_dir, doc.url),
            os.path.join(os.path.dirname(base_proj_dir), doc.url),
            os.path.join(base_proj_dir, doc.url.lstrip("/")),
            os.path.join(os.path.dirname(base_proj_dir), doc.url.lstrip("/"))
        ]
        
        for p in search_paths:
            if os.path.exists(p) and os.path.isfile(p):
                local_src = p
                break
        
        if local_src:
            url_filename = os.path.basename(doc.url)
            # Filename sanitization
            safe_filename = "".join([c if c.isalnum() or c in "._-" else "_" for c in url_filename])
            
            # Truncate if too long (path limit safety)
            max_name_len = 120
            name_part, ext_part = os.path.splitext(safe_filename)
            if len(safe_filename) > max_name_len:
                safe_filename = name_part[:max_name_len - len(ext_part)] + ext_part
                
            local_dest = os.path.abspath(os.path.join(fetched_dir, safe_filename))
            try:
                shutil.copy(local_src, local_dest)
                doc.local_path = local_dest
                print(f"      [FETCHED] {safe_filename}")
            except Exception as e:
                print(f"      [FETCH ERROR] {e}")

def build_context_string(relevant_points: Dict[str, List[str]]) -> str:
    """Constructs a context string from relevant points for LLM consumption."""
    ctx = "--- MEDICAL ---\n" + "\n".join(relevant_points.get("medical", []))
    ctx += "\n\n--- PHARMACY ---\n" + "\n".join(relevant_points.get("pharmacy", []))
    return ctx
