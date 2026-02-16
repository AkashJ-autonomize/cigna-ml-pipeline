import os
import json
import time
from src.html_parser import process_html_document

def main():
    input_dir = "files"
    output_dir = "output"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    html_files = [f for f in os.listdir(input_dir) if f.endswith(".html")]
    
    for html_file in html_files:
        print(f"Processing {html_file}...")
        start_time = time.time()
        file_path = os.path.join(input_dir, html_file)
        
        try:
            result = process_html_document(file_path)
            
            # Update duration
            result.metrics.duration_seconds = time.time() - start_time
            
            # Save output
            output_file = os.path.join(output_dir, html_file.replace(".html", ".json"))
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result.model_dump(), f, indent=4)
                
            print(f"Finished processing {html_file}. Results saved to {output_file}")
            
        except Exception as e:
            print(f"Error processing {html_file}: {e}")

if __name__ == "__main__":
    main()
