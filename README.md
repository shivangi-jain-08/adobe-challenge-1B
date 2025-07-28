# Round 1B: Persona-Driven Document Intelligence

## Overview

This solution builds upon the Round 1A outline extractor to create an intelligent document analyst that extracts and prioritizes the most relevant sections from a collection of documents based on a specific persona and their job-to-be-done.

## Approach

### 1. Architecture

The solution consists of three main components:

- **extract_outline.py**: The foundation from Round 1A that handles PDF structure extraction
- **document_analyzer.py**: Core intelligence engine that analyzes persona relevance
- **main.py**: Orchestrator that handles I/O and coordinates the analysis pipeline

### 2. Methodology

#### Persona & Job Analysis
- Extracts keywords from persona role using domain-specific mappings
- Identifies job-specific requirements and priorities
- Creates weighted keyword sets for relevance scoring

#### Section Extraction
- Leverages Round 1A outline extraction for document structure
- Falls back to page-based sections for documents without clear structure
- Extracts full content for each identified section

#### Relevance Scoring
- **Title Scoring (5x weight)**: Keywords in section titles are heavily weighted
- **Content Scoring (2x weight for job keywords, 1.5x for persona keywords)**: Frequency-based scoring normalized by content length
- **Structural Bonuses**: Extra points for introduction, methodology, results, and recommendation sections
- **Length Factor**: Prefers substantial sections with meaningful content

#### Subsection Analysis
- Splits sections into paragraphs for granular analysis
- Scores each paragraph using the same keyword-based approach
- Extracts and refines the most relevant subsections
- Removes excessive whitespace and formats text for readability

### 3. Key Features

#### Generalized Persona Handling
- Built-in mappings for common personas (Travel Planner, Researcher, Student, Investment Analyst, etc.)
- Flexible keyword extraction from role descriptions
- Adaptive scoring based on persona type

#### Smart Job-to-be-Done Processing
- Pattern recognition for common job types (trip planning, literature review, financial analysis, exam preparation)
- Dynamic keyword weighting based on task requirements
- Context-aware section prioritization

#### Robust Document Processing
- Handles documents with and without clear structure
- Graceful fallbacks for problematic PDFs
- Memory-efficient processing for large document collections

#### Quality Filtering
- Minimum relevance thresholds to avoid noise
- Length-based filtering for substantial content
- Duplicate detection and removal

## Libraries Used

- **PyMuPDF (fitz)**: PDF text extraction and processing
- **re**: Regular expression processing for pattern matching
- **json**: Input/output data handling
- **dataclasses**: Clean data structure definitions
- **collections**: Efficient data structures (defaultdict, Counter)
- **statistics**: Statistical calculations for scoring

## Build and Run Instructions

### Building the Docker Image
```bash
docker build --platform linux/amd64 -t persona-doc-analyzer:latest .
```

### Running the Solution
```bash
 docker run --rm -v ${PWD}/input:/app/input -v ${PWD}/output:/app/output --network none persona-doc-analyzer:latest
```

### Input Format
Place a JSON configuration file and PDF documents in the input directory:

```json
{
    "challenge_info": {
        "challenge_id": "round_1b_xxx",
        "test_case_name": "test_case",
        "description": "Description"
    },
    "documents": [
        {
            "filename": "document1.pdf",
            "title": "Document 1 Title"
        }
    ],
    "persona": {
        "role": "Travel Planner"
    },
    "job_to_be_done": {
        "task": "Plan a trip of 4 days for a group of 10 college friends."
    }
}
```

### Output Format
The solution generates a `challenge1b_output.json` file with:

- **Metadata**: Input documents, persona, job description, and processing timestamp
- **Extracted Sections**: Ranked sections with importance scores
- **Sub-section Analysis**: Refined text extracts from the most relevant content

## Performance Characteristics

- **Processing Time**: Optimized for <60 seconds on document collections (3-10 PDFs)
- **Memory Usage**: Efficient text processing with minimal memory footprint
- **Model Size**: No external models required, uses rule-based intelligence
- **CPU Only**: Pure Python implementation, no GPU dependencies

## Constraints Compliance

✅ **CPU Only**: No GPU dependencies  
✅ **Model Size**: No external models (<1GB total)  
✅ **Processing Time**: <60 seconds for typical document collections  
✅ **Offline**: No internet access required  
✅ **Architecture**: Compatible with AMD64 (x86_64)

## Testing

The solution has been designed to handle diverse scenarios:

- **Academic Research**: Research papers with methodology focus
- **Business Analysis**: Financial reports with investment insights
- **Educational Content**: Textbooks with student learning objectives
- **Travel Planning**: Travel guides with practical trip information

Each scenario uses different keyword sets and scoring strategies to ensure relevant content extraction.