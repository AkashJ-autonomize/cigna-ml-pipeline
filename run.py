import os
import json
import sys
from dotenv import load_dotenv

# Ensure src is in the system path for local imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.html_parser import process_olam_html
from src.processor import DrugProcessor

# Initialize environment variables
load_dotenv()

def main():
    """
    Main orchestration engine for the Cigna OLAM Pipeline.
    Iterates through provider folders, parses HTML, and performs multi-stage AI extraction.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_base = os.path.join(base_dir, "files")
    output_base = os.path.join(base_dir, "output")
    extraction_base = os.path.join(base_dir, "extraction")
    
    # --- CONFIGURATION ---
    # LOCAL_TEST_PROVIDERS: List specific folders to process (e.g., ["BJCHealthcare"])
    # USE_LOCAL_TESTING: If True, only processes providers in the LOCAL_TEST_PROVIDERS list.
    LOCAL_TEST_PROVIDERS = ["MaritzHolding", "BayCareHealth", "BJCHealthcare", "TUFTs"] 
    USE_LOCAL_TESTING = False   #True to test on the above specific providers
    # ---------------------

    if USE_LOCAL_TESTING and LOCAL_TEST_PROVIDERS:
        print(f"[LOCAL TEST] Limiting scan to: {LOCAL_TEST_PROVIDERS}")

    # Ensure output directories exist
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(extraction_base, exist_ok=True)
    
    processor = DrugProcessor()
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    # Determine target providers
    target_providers = LOCAL_TEST_PROVIDERS if (USE_LOCAL_TESTING and LOCAL_TEST_PROVIDERS) else os.listdir(input_base)

    for provider_name in target_providers:
        provider_path = os.path.join(input_base, provider_name)
        if not os.path.isdir(provider_path):
            continue

        for root, dirs, files in os.walk(provider_path):
            for filename in files:
                if filename.lower().endswith((".html", ".htm")):
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(root, input_base)
                    
                    out_dir = os.path.join(output_base, rel_path)
                    ext_dir = os.path.join(extraction_base, rel_path)
                    
                    os.makedirs(out_dir, exist_ok=True)
                    os.makedirs(ext_dir, exist_ok=True)
                    
                    print(f"\n[EXTRACT] {os.path.join(rel_path, filename)}")
                    
                    try:
                        # STAGE 1 & 2: Structural Parsing & Initial Extraction
                        extraction = process_olam_html(file_path)
                        
                        # Save structural results
                        out_file = os.path.join(out_dir, os.path.splitext(filename)[0] + ".json")
                        with open(out_file, 'w', encoding='utf-8') as f:
                            json.dump(extraction.model_dump(), f, indent=4, ensure_ascii=False)
                        print(f"  -> Saved parser results to output/")

                        # STAGE 3: Multi-Stage AI Refinement
                        if api_key:
                            relevant = processor.filter_relevant_points(extraction)
                            refined = processor.process_with_llm(relevant, extraction, out_dir)
                            
                            # Save refined AI results
                            # Check if we have drug rules or meaningful metadata
                            has_metadata = any([
                                refined.metadata.total_carve_out_drugs,
                                refined.metadata.source_document,
                                refined.metadata.policy_name,
                                refined.metadata.client_name,
                                refined.metadata.additional_info
                            ])
                            
                            if refined.drug_rules or has_metadata:
                                ext_file = os.path.join(ext_dir, os.path.splitext(filename)[0] + ".json")
                                with open(ext_file, 'w', encoding='utf-8') as f:
                                    json.dump(refined.model_dump(), f, indent=4, ensure_ascii=False)
                                print(f"  -> Saved AI rules and flags to extraction/")
                        
                    except Exception as e:
                        print(f"  [ERROR] {filename}: {e}")

    print("\n" + "="*50)
    print("  [DONE] Cigna OLAM Pipeline Scan Completed")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
