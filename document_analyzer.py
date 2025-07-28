import fitz  
import json
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math
from collections import defaultdict, Counter
import statistics

@dataclass
class DocumentSection:
    """Represents a section extracted from a document"""
    document: str
    page_number: int
    section_title: str
    content: str
    importance_rank: int
    confidence_score: float

@dataclass
class SubSection:
    """Represents a subsection with refined text"""
    document: str
    refined_text: str
    page_number: int
    relevance_score: float

class DocumentAnalyzer:
    """Analyzes documents to extract persona-relevant sections"""
    
    def __init__(self):
        self.persona_keywords = {}
        self.job_keywords = {}
        self.section_importance_weights = {}
        
    def analyze_persona_and_job(self, persona: Dict, job_to_be_done: Dict) -> Tuple[List[str], List[str]]:
        """Extract keywords and focus areas from persona and job description"""
        persona_role = persona.get('role', '').lower()
        job_task = job_to_be_done.get('task', '').lower()
        
        # Domain-specific keyword mappings
        persona_keyword_map = {
            'travel planner': ['travel', 'trip', 'vacation', 'destination', 'itinerary', 'accommodation', 
                              'restaurant', 'hotel', 'tourist', 'sightseeing', 'culture', 'activity'],
            'researcher': ['research', 'study', 'analysis', 'methodology', 'data', 'experiment', 
                          'findings', 'results', 'literature', 'academic', 'theory'],
            'student': ['study', 'learn', 'exam', 'concept', 'theory', 'practice', 'homework', 
                       'assignment', 'textbook', 'lecture', 'notes'],
            'investment analyst': ['financial', 'revenue', 'profit', 'investment', 'market', 'growth', 
                                  'performance', 'analysis', 'roi', 'trends', 'forecast'],
            'salesperson': ['sales', 'customer', 'product', 'market', 'lead', 'revenue', 'target', 
                           'negotiation', 'proposal', 'client'],
            'journalist': ['news', 'report', 'story', 'interview', 'source', 'fact', 'investigation', 
                          'article', 'headline', 'coverage']
        }
        
        # Extract persona keywords
        persona_keywords = []
        for role, keywords in persona_keyword_map.items():
            if role in persona_role:
                persona_keywords.extend(keywords)
        
        # Add generic persona keywords from role description
        persona_words = re.findall(r'\b\w+\b', persona_role)
        persona_keywords.extend([w for w in persona_words if len(w) > 3])
        
        # Extract job-specific keywords
        job_keywords = []
        
        # Common job patterns
        if 'plan' in job_task and 'trip' in job_task:
            job_keywords.extend(['plan', 'trip', 'travel', 'itinerary', 'schedule', 'day', 'visit'])
        if 'literature review' in job_task:
            job_keywords.extend(['literature', 'review', 'methodology', 'study', 'research'])
        if 'financial' in job_task or 'revenue' in job_task:
            job_keywords.extend(['financial', 'revenue', 'profit', 'income', 'expense', 'budget'])
        if 'exam' in job_task or 'study' in job_task:
            job_keywords.extend(['study', 'exam', 'concept', 'theory', 'important', 'key'])
        
        # Extract all meaningful words from job description
        job_words = re.findall(r'\b\w+\b', job_task)
        job_keywords.extend([w for w in job_words if len(w) > 3])
        
        # Remove duplicates while preserving order
        persona_keywords = list(dict.fromkeys(persona_keywords))
        job_keywords = list(dict.fromkeys(job_keywords))
        
        return persona_keywords, job_keywords
    
    def extract_document_sections(self, pdf_path: str, filename: str) -> List[Dict]:
        """Extract sections from a PDF document using the existing outline extractor"""
        from extract_outline import AdaptivePDFExtractor
        
        extractor = AdaptivePDFExtractor()
        result = extractor.process_pdf(pdf_path)
        
        # Get full text content for each section
        doc = fitz.open(pdf_path)
        sections = []
        
        outline = result.get('outline', [])
        
        # If no outline found, create sections from pages
        if not outline:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    sections.append({
                        'document': filename,
                        'page_number': page_num + 1,
                        'section_title': f"Page {page_num + 1}",
                        'content': text.strip()
                    })
        else:
            # Create sections based on outline
            for i, heading in enumerate(outline):
                start_page = heading['page']
                end_page = outline[i + 1]['page'] if i + 1 < len(outline) else len(doc)
                
                content_parts = []
                for page_num in range(start_page - 1, end_page):
                    if page_num < len(doc):
                        page = doc[page_num]
                        text = page.get_text()
                        content_parts.append(text)
                
                content = '\n'.join(content_parts).strip()
                if content:
                    sections.append({
                        'document': filename,
                        'page_number': heading['page'],
                        'section_title': heading['text'],
                        'content': content
                    })
        
        doc.close()
        return sections
    
    def calculate_section_relevance(self, section: Dict, persona_keywords: List[str], 
                                  job_keywords: List[str]) -> float:
        """Calculate relevance score for a section based on persona and job keywords"""
        content = section['content'].lower()
        title = section['section_title'].lower()
        
        # Weight title higher than content
        title_score = 0
        content_score = 0
        
        # Calculate keyword matches in title (higher weight)
        for keyword in persona_keywords + job_keywords:
            if keyword in title:
                title_score += 3 if keyword in job_keywords else 2
        
        # Calculate keyword matches in content
        content_words = re.findall(r'\b\w+\b', content)
        content_word_count = len(content_words)
        
        if content_word_count > 0:
            for keyword in persona_keywords:
                matches = content.count(keyword)
                content_score += matches * 1.5
            
            for keyword in job_keywords:
                matches = content.count(keyword)
                content_score += matches * 2.0
            
            # Normalize by content length
            content_score = content_score / max(content_word_count, 1) * 1000
        
        # Additional scoring factors
        section_length_score = min(len(content) / 1000, 2)  # Prefer substantial sections
        
        # Bonus for certain section types
        title_bonus = 0
        important_section_patterns = [
            r'\b(introduction|overview|summary|conclusion)\b',
            r'\b(methodology|method|approach)\b',
            r'\b(results|findings|analysis)\b',
            r'\b(recommendation|suggestion|tip)\b'
        ]
        
        for pattern in important_section_patterns:
            if re.search(pattern, title):
                title_bonus += 1
        
        total_score = title_score * 5 + content_score + section_length_score + title_bonus
        return total_score
    
    def extract_subsections(self, section: Dict, persona_keywords: List[str], 
                          job_keywords: List[str], max_subsections: int = 3) -> List[SubSection]:
        """Extract and refine relevant subsections from a section"""
        content = section['content']
        
        # Split content into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # Score each paragraph
        paragraph_scores = []
        for i, para in enumerate(paragraphs):
            if len(para) < 50:  # Skip very short paragraphs
                continue
                
            para_lower = para.lower()
            score = 0
            
            # Keyword matching
            for keyword in persona_keywords:
                score += para_lower.count(keyword) * 1.5
            for keyword in job_keywords:
                score += para_lower.count(keyword) * 2.0
            
            # Length factor (prefer substantial paragraphs)
            length_factor = min(len(para) / 200, 2)
            score += length_factor
            
            # Position factor (earlier paragraphs slightly preferred)
            position_factor = max(0, 1 - i * 0.1)
            score += position_factor
            
            paragraph_scores.append((score, para, i))
        
        # Sort by relevance and take top subsections
        paragraph_scores.sort(key=lambda x: x[0], reverse=True)
        
        subsections = []
        for i, (score, para, orig_index) in enumerate(paragraph_scores[:max_subsections]):
            if score > 0.5:  # Minimum relevance threshold
                # Refine the text (remove excessive whitespace, fix formatting)
                refined_text = re.sub(r'\s+', ' ', para).strip()
                
                subsections.append(SubSection(
                    document=section['document'],
                    refined_text=refined_text,
                    page_number=section['page_number'],
                    relevance_score=score
                ))
        
        return subsections
    
    def rank_sections(self, sections: List[Dict], persona_keywords: List[str], 
                     job_keywords: List[str]) -> List[DocumentSection]:
        """Rank sections by relevance and create DocumentSection objects"""
        scored_sections = []
        
        for section in sections:
            relevance_score = self.calculate_section_relevance(section, persona_keywords, job_keywords)
            
            doc_section = DocumentSection(
                document=section['document'],
                page_number=section['page_number'],
                section_title=section['section_title'],
                content=section['content'],
                importance_rank=0,  # Will be set after sorting
                confidence_score=relevance_score
            )
            scored_sections.append(doc_section)
        
        # Sort by relevance score
        scored_sections.sort(key=lambda x: x.confidence_score, reverse=True)
        
        # Assign importance ranks
        for i, section in enumerate(scored_sections):
            section.importance_rank = i + 1
        
        return scored_sections
    
    def process_document_collection(self, input_data: Dict, input_dir: str) -> Dict:
        """Main processing function for Round 1B"""
        print("Processing document collection...")
        
        # Extract persona and job information
        persona = input_data['persona']
        job_to_be_done = input_data['job_to_be_done']
        documents = input_data['documents']
        
        print(f"Persona: {persona}")
        print(f"Job: {job_to_be_done}")
        
        # Analyze persona and job to extract keywords
        persona_keywords, job_keywords = self.analyze_persona_and_job(persona, job_to_be_done)
        
        print(f"Persona keywords: {persona_keywords[:10]}...")  # Show first 10
        print(f"Job keywords: {job_keywords[:10]}...")
        
        # Extract sections from all documents
        all_sections = []
        for doc_info in documents:
            filename = doc_info['filename']
            pdf_path = f"{input_dir}/{filename}"
            
            try:
                sections = self.extract_document_sections(pdf_path, filename)
                all_sections.extend(sections)
                print(f"Extracted {len(sections)} sections from {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        # Rank sections by relevance
        ranked_sections = self.rank_sections(all_sections, persona_keywords, job_keywords)
        
        # Take top sections (limit to reasonable number)
        top_sections = ranked_sections[:15]  # Adjust based on requirements
        
        # Extract subsections from top sections
        all_subsections = []
        for section in top_sections[:10]:  # Limit subsection extraction to top 10 sections
            subsections = self.extract_subsections(
                {
                    'document': section.document,
                    'page_number': section.page_number,
                    'section_title': section.section_title,
                    'content': section.content
                },
                persona_keywords, 
                job_keywords
            )
            all_subsections.extend(subsections)
        
        # Sort subsections by relevance
        all_subsections.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Create output format
        output = {
            "metadata": {
                "input_documents": [doc['filename'] for doc in documents],
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": "2025-07-28T00:00:00Z"
            },
            "extracted_sections": [
                {
                    "document": section.document,
                    "page_number": section.page_number,
                    "section_title": section.section_title,
                    "importance_rank": section.importance_rank
                }
                for section in top_sections
            ],
            "sub_section_analysis": [
                {
                    "document": subsection.document,
                    "refined_text": subsection.refined_text,
                    "page_number": subsection.page_number
                }
                for subsection in all_subsections[:20]  # Limit to top 20 subsections
            ]
        }
        
        return output