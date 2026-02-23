import os
import json
import sys
import time
import traceback
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.html_parser import process_olam_html
from src.processor import DrugProcessor

load_dotenv()


def main():
    """Cigna OLAM Pipeline — parses HTML and performs multi-stage AI drug extraction."""
    base_dir     = os.path.dirname(os.path.abspath(__file__))
    input_base   = os.path.join(base_dir, "files")
    output_base  = os.path.join(base_dir, "output")
    extract_base = os.path.join(base_dir, "extraction")

    # Set USE_LOCAL_TESTING = True and list folders in LOCAL_TEST_PROVIDERS to limit the scan.
    LOCAL_TEST_PROVIDERS = ["UPS"]
    USE_LOCAL_TESTING    = False

    os.makedirs(output_base, exist_ok=True)
    os.makedirs(extract_base, exist_ok=True)

    processor = DrugProcessor()
    api_key   = os.getenv("AZURE_OPENAI_API_KEY")

    providers = LOCAL_TEST_PROVIDERS if (USE_LOCAL_TESTING and LOCAL_TEST_PROVIDERS) else os.listdir(input_base)
    if USE_LOCAL_TESTING and LOCAL_TEST_PROVIDERS:
        print(f"[LOCAL TEST] Providers: {LOCAL_TEST_PROVIDERS}\n")

    # Pipeline-level totals and per-client stats
    pipeline_start  = time.time()
    files_scanned   = 0
    pipeline_totals = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "duration_seconds": 0.0}
    client_stats    = {}   # { client_name: llm_stats }

    for provider_name in providers:
        provider_path = os.path.join(input_base, provider_name)
        if not os.path.isdir(provider_path):
            continue

        for root, _, files in os.walk(provider_path):
            for filename in files:
                if not filename.lower().endswith((".html", ".htm")):
                    continue

                file_path = os.path.join(root, filename)
                rel_path  = os.path.relpath(root, input_base)
                out_dir   = os.path.join(output_base, rel_path)
                ext_dir   = os.path.join(extract_base, rel_path)
                os.makedirs(out_dir, exist_ok=True)
                os.makedirs(ext_dir, exist_ok=True)

                rel_file = os.path.join(rel_path, filename)
                print(f"── {rel_file}")
                files_scanned += 1

                try:
                    extraction = process_olam_html(file_path)

                    out_file = os.path.join(out_dir, os.path.splitext(filename)[0] + ".json")
                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(extraction.model_dump(), f, indent=4, ensure_ascii=False)

                    if api_key:
                        relevant = processor.filter_relevant_points(extraction)
                        refined  = processor.process_with_llm(
                            relevant, extraction, provider_name, out_dir, base_dir=base_dir
                        )

                        ext_file = os.path.join(ext_dir, os.path.splitext(filename)[0] + ".json")
                        with open(ext_file, "w", encoding="utf-8") as f:
                            json.dump(refined.model_dump(), f, indent=4, ensure_ascii=False)

                        # Accumulate pipeline totals and record per-client stats
                        file_stats = refined.metadata.llm_stats
                        for key in pipeline_totals:
                            pipeline_totals[key] = round(pipeline_totals[key] + file_stats.get(key, 0), 3)
                        client_stats[provider_name] = file_stats

                        drugs     = refined.metadata.total_drugs_found
                        scenarios = refined.metadata.total_scenarios_found
                        calls     = file_stats.get("llm_calls", 0)
                        tokens    = file_stats.get("total_tokens", 0)
                        duration  = file_stats.get("duration_seconds", 0)
                        print(f"   Saved  {os.path.relpath(ext_file, base_dir)}")
                        print(f"          {drugs} drug(s)  ·  {scenarios} scenario(s)  ·  {calls} LLM call(s)  ·  {tokens} tokens  ·  {duration}s\n")

                except Exception as e:
                    traceback.print_exc()
                    print(f"   [ERROR] {filename}: {e}\n")

    pipeline_wall = round(time.time() - pipeline_start, 2)
    pipeline_totals["wall_clock_seconds"] = pipeline_wall

    # Save pipeline_summary.json in extraction/
    summary = {
        "total_analysis": {
            "files_scanned":              files_scanned,
            "pipeline_wall_clock_seconds": pipeline_wall,
            **pipeline_totals,
        },
        "clients": client_stats,
    }
    summary_path = os.path.join(extract_base, "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"{'═' * 52}")
    print(f"  Pipeline Complete  |  {files_scanned} file(s) scanned  |  {pipeline_wall}s total")
    print(f"  LLM: {pipeline_totals['llm_calls']} call(s)  ·  {pipeline_totals['total_tokens']} tokens  ·  {pipeline_totals['duration_seconds']}s LLM time")
    print(f"  Summary saved → extraction/pipeline_summary.json")
    print(f"{'═' * 52}\n")


if __name__ == "__main__":
    main()