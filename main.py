import os
import json
import sys
from datetime import datetime
from document_analyzer import DocumentAnalyzer

def load_input_config(input_dir: str) -> dict:
    """Load the input configuration JSON file"""
    config_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not config_files:
        raise FileNotFoundError("No JSON configuration file found in input directory")
    
    config_file = config_files[0]  # Take the first JSON file
    config_path = os.path.join(input_dir, config_file)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def validate_input(input_data: dict) -> bool:
    """Validate the input data structure"""
    required_fields = ['documents', 'persona', 'job_to_be_done']
    
    for field in required_fields:
        if field not in input_data:
            print(f"Error: Missing required field '{field}' in input data")
            return False
    
    if not isinstance(input_data['documents'], list) or not input_data['documents']:
        print("Error: 'documents' must be a non-empty list")
        return False
    
    for doc in input_data['documents']:
        if 'filename' not in doc:
            print("Error: Each document must have a 'filename' field")
            return False
    
    if 'role' not in input_data['persona']:
        print("Error: Persona must have a 'role' field")
        return False
    
    if 'task' not in input_data['job_to_be_done']:
        print("Error: Job-to-be-done must have a 'task' field")
        return False
    
    return True

def check_pdf_files(input_data: dict, input_dir: str) -> bool:
    """Check if all required PDF files exist"""
    missing_files = []
    
    for doc in input_data['documents']:
        pdf_path = os.path.join(input_dir, doc['filename'])
        if not os.path.exists(pdf_path):
            missing_files.append(doc['filename'])
    
    if missing_files:
        print(f"Error: Missing PDF files: {missing_files}")
        return False
    
    return True

def main():
    """Main execution function"""
    print("=" * 60)
    print("Adobe India Hackathon 2025 - Round 1B")
    print("Persona-Driven Document Intelligence")
    print("=" * 60)
    
    # Define input and output directories
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load input configuration
        print(f"Loading input configuration from {input_dir}...")
        input_data = load_input_config(input_dir)
        
        # Validate input data
        if not validate_input(input_data):
            sys.exit(1)
        
        # Check if all PDF files exist
        if not check_pdf_files(input_data, input_dir):
            sys.exit(1)
        
        print(f"Found {len(input_data['documents'])} documents to process")
        print(f"Persona: {input_data['persona']['role']}")
        print(f"Job: {input_data['job_to_be_done']['task']}")
        
        # Initialize document analyzer
        analyzer = DocumentAnalyzer()
        
        # Process the document collection
        print("\nStarting document analysis...")
        start_time = datetime.now()
        
        result = analyzer.process_document_collection(input_data, input_dir)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Update timestamp in result
        result['metadata']['processing_timestamp'] = end_time.isoformat() + 'Z'
        
        # Save results
        output_filename = "challenge1b_output.json"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Documents processed: {len(input_data['documents'])}")
        print(f"Sections extracted: {len(result['extracted_sections'])}")
        print(f"Subsections analyzed: {len(result['sub_section_analysis'])}")
        print(f"Output saved to: {output_path}")
        
        # Print top sections for verification
        print("\nTop 5 Most Relevant Sections:")
        for i, section in enumerate(result['extracted_sections'][:5]):
            print(f"{i+1}. [{section['document']}] {section['section_title']} (Page {section['page_number']})")
        
        print("\nTop 3 Subsection Extracts:")
        for i, subsection in enumerate(result['sub_section_analysis'][:3]):
            text_preview = subsection['refined_text'][:100] + "..." if len(subsection['refined_text']) > 100 else subsection['refined_text']
            print(f"{i+1}. [{subsection['document']}] {text_preview} (Page {subsection['page_number']})")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in input file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()