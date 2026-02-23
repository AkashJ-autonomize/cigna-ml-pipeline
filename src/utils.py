import os
import shutil
from src.schemas import OLAMExtraction


def cache_download_documents(extraction: OLAMExtraction, output_dir: str, base_dir: str = None):
    """
    Copies referenced local documents into output_dir/fetched_files/.
    Resolves paths relative to base_dir (project root) or cwd as fallback.
    Sanitises filenames and truncates long names to avoid OS path limits.
    """
    fetched_dir   = os.path.join(output_dir, "fetched_files")
    os.makedirs(fetched_dir, exist_ok=True)

    all_docs      = extraction.medical.hyperlinks + extraction.pharmacy.hyperlinks
    base_proj_dir = base_dir or os.getcwd()

    for doc in all_docs:
        if doc.local_path and os.path.exists(doc.local_path):
            continue

        candidates = [
            os.path.join(base_proj_dir, doc.url),
            os.path.join(base_proj_dir, doc.url.lstrip("/")),
            os.path.join(os.getcwd(), doc.url),
            os.path.join(os.getcwd(), doc.url.lstrip("/")),
            os.path.join(os.path.dirname(base_proj_dir), doc.url),
            os.path.join(os.path.dirname(base_proj_dir), doc.url.lstrip("/")),
        ]
        local_src = next((p for p in candidates if os.path.exists(p) and os.path.isfile(p)), None)
        if not local_src:
            continue

        url_filename = os.path.basename(doc.url)
        safe_name    = "".join(c if c.isalnum() or c in "._-" else "_" for c in url_filename)
        name, ext    = os.path.splitext(safe_name)
        if len(safe_name) > 120:
            safe_name = name[:120 - len(ext)] + ext

        dest = os.path.abspath(os.path.join(fetched_dir, safe_name))
        try:
            shutil.copy(local_src, dest)
            doc.local_path = dest
            print(f"   Fetch   {safe_name}")
        except Exception as e:
            print(f"   [FETCH ERROR] {safe_name}: {e}")


def build_context_string(relevant_points: dict) -> str:
    """Constructs a combined Medical + Pharmacy context string for LLM consumption."""
    medical  = "\n".join(relevant_points.get("medical", []))
    pharmacy = "\n".join(relevant_points.get("pharmacy", []))
    return f"--- MEDICAL ---\n{medical}\n\n--- PHARMACY ---\n{pharmacy}"