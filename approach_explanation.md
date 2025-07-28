# Approach Explanation: Persona-Driven Document Intelligence

## Core Methodology

Our solution implements a multi-layered relevance scoring system that adapts to different personas and job requirements through intelligent keyword extraction and weighted content analysis.

### Persona & Job Intelligence

The system first analyzes the input persona and job-to-be-done to extract domain-specific keywords. We maintain curated keyword mappings for common personas (Travel Planner, Researcher, Investment Analyst, etc.) while also dynamically extracting meaningful terms from role descriptions. Job tasks are parsed for action words and objectives, creating two distinct keyword sets with different weights in our scoring algorithm.

### Document Structure Extraction

Building upon our Round 1A outline extractor, we extract structured sections from each PDF. For documents with clear headings, we use the hierarchical structure; for less structured documents, we fall back to page-based sectioning. This ensures robust processing across diverse document types without losing critical content.

### Relevance Scoring Algorithm

Our scoring system employs multiple weighted factors:

**Title Analysis (5x multiplier)**: Section titles receive the highest weight since they typically indicate content focus. Keywords matching in titles significantly boost relevance scores.

**Content Keyword Density**: We calculate normalized keyword frequency, giving job-specific keywords 2x weight over persona keywords. This prioritizes content directly related to the task at hand.

**Structural Intelligence**: Sections identified as introductions, methodologies, results, or recommendations receive bonus points, as these typically contain high-value information across domains.

**Content Quality Metrics**: We factor in section length and linguistic patterns to prefer substantial, well-formed content over fragments or boilerplate text.

### Subsection Extraction & Refinement

For top-scoring sections, we perform granular paragraph-level analysis using the same keyword-weighted approach. The most relevant paragraphs are extracted and refined through text cleaning and formatting normalization. This provides users with precisely targeted content excerpts rather than entire sections.

### Adaptive Filtering

The system employs dynamic thresholds and quality filters to eliminate noise while preserving relevant content. We remove duplicates, filter out sections below relevance thresholds, and apply content-type detection to avoid including non-substantive text like headers, footers, or navigation elements.

This approach ensures that regardless of document domain or persona type, the system consistently identifies and ranks the most pertinent information for the user's specific needs.