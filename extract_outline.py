import fitz  # PyMuPDF
import json
import re
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import statistics
import math

@dataclass
class TextBlock:
    """Simple container for text with all its formatting info"""
    text: str
    font_size: float
    font_name: str
    bbox: tuple
    page_num: int
    is_bold: bool
    flags: int
    line_height: float
    char_count: int

@dataclass
class HeadingCandidate:
    """Represents a potential heading with confidence scoring"""
    text: str
    page: int
    confidence: float
    level: str
    font_signature: str
    position_score: float
    uniqueness_score: float

class DocumentContext:
    """Learns document-specific patterns and conventions"""
    
    def __init__(self, text_blocks: List[TextBlock]):
        self.text_blocks = text_blocks
        # Build a map of all font combinations used in the doc
        self.font_signatures = self._build_font_signatures()
        # Figure out what the main body text looks like
        self.body_text_signature = self._identify_body_text()
        # Find fonts that are used rarely (potential headings)
        self.outlier_signatures = self._find_outlier_signatures()
        # Learn about typical positioning in this document
        self.positional_patterns = self._analyze_positions()
        # Understand text length patterns
        self.length_distribution = self._analyze_text_lengths()
        
    def _build_font_signatures(self) -> Dict[str, List[TextBlock]]:
        """Group text blocks by unique font combinations"""
        signatures = defaultdict(list)
        for block in self.text_blocks:
            # Create a unique signature for each font+size+bold combo
            sig = f"{block.font_name}_{block.font_size}_{block.is_bold}"
            signatures[sig].append(block)
        return dict(signatures)
    
    def _identify_body_text(self) -> str:
        """Figure out which font signature is most likely the main body text"""
        signature_scores = {}
        
        for sig, blocks in self.font_signatures.items():
            frequency = len(blocks)
            pages = len(set(block.page_num for block in blocks))
            avg_length = statistics.mean(len(block.text) for block in blocks)
            
            # Body text should be frequent, appear on many pages, and have decent length
            score = frequency * 0.4 + pages * 0.3 + min(avg_length / 50, 3) * 0.3
            signature_scores[sig] = score
        
        return max(signature_scores.items(), key=lambda x: x[1])[0] if signature_scores else ""
    
    def _find_outlier_signatures(self) -> Set[str]:
        """Find font signatures that are significantly different from body text"""
        if not self.body_text_signature:
            return set()
        
        body_blocks = self.font_signatures.get(self.body_text_signature, [])
        if not body_blocks:
            return set()
        
        body_font_size = body_blocks[0].font_size
        outliers = set()
        
        for sig, blocks in self.font_signatures.items():
            if sig == self.body_text_signature:
                continue
                
            block_font_size = blocks[0].font_size
            size_diff = abs(block_font_size - body_font_size)
            is_size_outlier = size_diff > 1.5
            
            usage_frequency = len(blocks)
            # If used much less than body text, might be headings
            is_sparse = usage_frequency < len(body_blocks) * 0.1
            
            if is_size_outlier or is_sparse:
                outliers.add(sig)
        
        return outliers
    
    def _analyze_positions(self) -> Dict:
        """Learn about typical positioning patterns in the document"""
        top_positions = [block.bbox[1] for block in self.text_blocks if block.bbox[1] < 200]
        left_positions = [block.bbox[0] for block in self.text_blocks]
        
        return {
            'typical_top_margin': statistics.median(top_positions) if top_positions else 100,
            'typical_left_margin': statistics.mode(left_positions) if left_positions else 72,
            'page_tops': sorted(set(top_positions))[:5] if top_positions else []
        }
    
    def _analyze_text_lengths(self) -> Dict:
        """Understand text length patterns to distinguish headings from paragraphs"""
        lengths = [len(block.text) for block in self.text_blocks]
        sorted_lengths = sorted(lengths)
        n = len(sorted_lengths)
        
        if n == 0:
            return {
                'median_length': 50,
                'short_text_threshold': 20,
                'long_text_threshold': 100
            }
        
        q1_index = n // 4
        q3_index = 3 * n // 4
        
        return {
            'median_length': statistics.median(lengths),
            'short_text_threshold': sorted_lengths[q1_index] if q1_index < n else sorted_lengths[0],
            'long_text_threshold': sorted_lengths[q3_index] if q3_index < n else sorted_lengths[-1]
        }

class AdaptivePDFExtractor:
    def __init__(self):
        self.context = None
        
    def extract_text_blocks(self, pdf_path: str) -> List[TextBlock]:
        """Extract all text blocks with comprehensive formatting information"""
        doc = fitz.open(pdf_path)
        text_blocks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")
            
            # Go through each text block on the page
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text and len(text) > 1:  # Skip tiny text fragments
                                line_height = span["bbox"][3] - span["bbox"][1]
                                
                                text_blocks.append(TextBlock(
                                    text=text,
                                    font_size=round(span["size"], 1),
                                    font_name=span["font"],
                                    bbox=span["bbox"],
                                    page_num=page_num + 1,
                                    is_bold=self._detect_bold(span),
                                    flags=span["flags"],
                                    line_height=line_height,
                                    char_count=len(text)
                                ))
        
        doc.close()
        return text_blocks
    
    def _detect_bold(self, span: dict) -> bool:
        """More robust bold detection - sometimes bold is in the name, sometimes in flags"""
        font_name = span["font"].lower()
        bold_in_name = any(word in font_name for word in ['bold', 'black', 'heavy', 'medium'])
        flag_bold = span["flags"] & 2**4  # Bold flag in PyMuPDF
        return bold_in_name or flag_bold
    
    def _combine_consecutive_blocks(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """
        Sometimes titles get split across multiple text blocks - this tries to fix that.
        Pretty tricky to get right without over-combining!
        """
        if not blocks:
            return blocks
        
        # Sort by page, then by vertical position, then by horizontal position
        sorted_blocks = sorted(blocks, key=lambda b: (b.page_num, b.bbox[1], b.bbox[0]))
        
        combined_blocks = []
        current_group = [sorted_blocks[0]]
        
        for i in range(1, len(sorted_blocks)):
            current_block = sorted_blocks[i]
            prev_block = sorted_blocks[i-1]
            
            # Check if blocks should be combined
            same_page = current_block.page_num == prev_block.page_num
            same_font_signature = (current_block.font_name == prev_block.font_name and 
                                 current_block.font_size == prev_block.font_size and 
                                 current_block.is_bold == prev_block.is_bold)
            
            # Are they vertically close (same line or very close lines)?
            vertical_distance = abs(current_block.bbox[1] - prev_block.bbox[1])
            close_vertically = vertical_distance < 10
            
            # Are they horizontally adjacent or close?
            horizontal_gap = current_block.bbox[0] - prev_block.bbox[2]
            close_horizontally = horizontal_gap < 100
            
            # Maybe they're on consecutive lines but part of same title?
            # Be conservative here to avoid over-combining
            consecutive_lines = (vertical_distance > 10 and vertical_distance < 25 and 
                               same_font_signature and same_page and
                               len(current_group) < 3)
            
            if (same_page and same_font_signature and 
                ((close_vertically and close_horizontally and horizontal_gap >= -20) or 
                 (consecutive_lines and len(prev_block.text.strip()) < 30))):
                current_group.append(current_block)
            else:
                # Finalize current group
                if len(current_group) > 1:
                    combined_text = ' '.join(block.text.strip() for block in current_group)
                    # Create a new combined block using the first block as template
                    first_block = current_group[0]
                    last_block = current_group[-1]
                    combined_block = TextBlock(
                        text=combined_text,
                        font_size=first_block.font_size,
                        font_name=first_block.font_name,
                        bbox=(first_block.bbox[0], first_block.bbox[1], 
                              last_block.bbox[2], max(first_block.bbox[3], last_block.bbox[3])),
                        page_num=first_block.page_num,
                        is_bold=first_block.is_bold,
                        flags=first_block.flags,
                        line_height=first_block.line_height,
                        char_count=len(combined_text)
                    )
                    combined_blocks.append(combined_block)
                else:
                    combined_blocks.append(current_group[0])
                
                current_group = [current_block]
        
        # Don't forget the last group!
        if len(current_group) > 1:
            combined_text = ' '.join(block.text.strip() for block in current_group)
            first_block = current_group[0]
            last_block = current_group[-1]
            combined_block = TextBlock(
                text=combined_text,
                font_size=first_block.font_size,
                font_name=first_block.font_name,
                bbox=(first_block.bbox[0], first_block.bbox[1], 
                      last_block.bbox[2], max(first_block.bbox[3], last_block.bbox[3])),
                page_num=first_block.page_num,
                is_bold=first_block.is_bold,
                flags=first_block.flags,
                line_height=first_block.line_height,
                char_count=len(combined_text)
            )
            combined_blocks.append(combined_block)
        else:
            combined_blocks.append(current_group[0])
        
        return combined_blocks

    def extract_title_candidates(self) -> List[HeadingCandidate]:
        """
        Extract potential title candidates using comprehensive scoring.
        Focuses on first few pages since that's where titles usually live.
        """
        candidates = []
        
        # Focus on first 3 pages but prioritize first page
        first_pages = [b for b in self.context.text_blocks if b.page_num <= 3]
        
        # Try to combine split titles first
        first_pages = self._combine_consecutive_blocks(first_pages)
        
        # Understanding the document's typography range helps with scoring
        all_font_sizes = [b.font_size for b in self.context.text_blocks]
        if not all_font_sizes:
            return candidates
            
        max_font_size = max(all_font_sizes)
        min_font_size = min(all_font_sizes)
        median_font_size = statistics.median(all_font_sizes)
        
        for block in first_pages:
            if self._is_obvious_non_title(block):
                continue
            
            score = 0.0
            
            # 1. Font size analysis - bigger usually means more important
            if max_font_size > min_font_size:
                size_percentile = (block.font_size - min_font_size) / (max_font_size - min_font_size)
            else:
                size_percentile = 0.5
            
            if block.font_size == max_font_size:
                score += 0.5  # Biggest font gets a big bonus
            elif size_percentile > 0.8:
                score += 0.4
            elif size_percentile > 0.6:
                score += 0.3
            
            # 2. Position scoring - titles are usually near the top
            y_position = block.bbox[1]
            if block.page_num == 1:
                if y_position < 100:
                    score += 0.6  # Very top of first page
                elif y_position < 250:
                    score += 0.4
                elif y_position < 400:
                    score += 0.2
            elif block.page_num == 2 and y_position < 150:
                score += 0.3  # Sometimes title continues on page 2
            
            # 3. How unique is this formatting? (more unique = more likely title)
            block_signature = f"{block.font_name}_{block.font_size}_{block.is_bold}"
            signature_count = len(self.context.font_signatures.get(block_signature, []))
            total_blocks = len(self.context.text_blocks)
            
            if signature_count == 1:
                score += 0.4  # Completely unique formatting
            elif signature_count <= 3:
                score += 0.3
            elif total_blocks > 0 and signature_count / total_blocks < 0.05:
                score += 0.2
            
            # 4. Length analysis - titles should be reasonable length
            length = len(block.text.strip())
            word_count = len(block.text.split())
            
            if 5 <= word_count <= 20:  # Sweet spot for titles
                score += 0.4
            elif 3 <= word_count <= 25:
                score += 0.3
            elif 2 <= word_count <= 30:
                score += 0.2
            elif word_count == 1 and length > 3:
                score += 0.1  # Single meaningful word
            
            # 5. Text characteristics
            text = block.text.strip()
            
            # Title case or proper capitalization
            if text.istitle() or (text[0].isupper() and not text.isupper()):
                score += 0.2
            
            # All caps (common for titles, but not too long)
            if text.isupper() and 3 <= len(text) <= 100:
                score += 0.15
            
            # Bold formatting
            if block.is_bold:
                score += 0.25
            
            # Titles often don't end with punctuation
            if not text.endswith(('.', ',', ';', ':')):
                score += 0.1
            
            # 6. Horizontal positioning - centered or left-aligned
            page_width = 612  # Standard PDF width
            text_center = (block.bbox[0] + block.bbox[2]) / 2
            
            if abs(text_center - page_width/2) < 50:  # Centered
                score += 0.2
            elif abs(text_center - page_width/2) < 100:  # Near center
                score += 0.1
            elif 50 <= block.bbox[0] <= 100:  # Standard left margin
                score += 0.1
            
            # 7. Penalties for obvious non-titles
            if self._looks_like_author_info(text):
                score -= 0.3
            elif self._looks_like_date_info(text):
                score -= 0.2
            elif self._looks_like_header_footer(text, y_position):
                score -= 0.4
            
            # 8. Bonus for title-like patterns
            if self._has_title_patterns(text):
                score += 0.2
            
            # 9. Bonus for complete-looking titles
            if self._looks_like_complete_title(text):
                score += 0.3
            
            # Only keep candidates with decent scores
            if score > 0.15:
                candidates.append(HeadingCandidate(
                    text=text,
                    page=block.page_num,
                    confidence=score,
                    level="TITLE",
                    font_signature=block_signature,
                    position_score=y_position,
                    uniqueness_score=signature_count / total_blocks if total_blocks > 0 else 1
                ))
        
        return sorted(candidates, key=lambda x: x.confidence, reverse=True)
    
    def _is_obvious_non_title(self, block: TextBlock) -> bool:
        """Quick filtering for stuff that's definitely not a title"""
        text = block.text.lower().strip()
        
        if len(text) <= 1:
            return True
        
        # URLs and emails
        if re.match(r'^(www\.|http|.*@.*\.)', text):
            return True
        
        # Pure numbers or dates
        if re.match(r'^\d+$', text) or re.match(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$', text):
            return True
        
        # Page numbers
        if re.match(r'^page \d+', text):
            return True
        
        # Way too long to be a title
        if len(text) > 200:
            return True
        
        return False
    
    def _looks_like_author_info(self, text: str) -> bool:
        """Check if text looks like author information"""
        text_lower = text.lower()
        author_indicators = ['by ', 'author:', 'written by', 'prepared by', 'university', 'department', 'institute']
        return any(indicator in text_lower for indicator in author_indicators)
    
    def _looks_like_date_info(self, text: str) -> bool:
        """Check if text looks like date information"""
        date_patterns = [
            r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}',
            r'\d{4}[/\-]\d{1,2}[/\-]\d{1,2}',
            r'(january|february|march|april|may|june|july|august|september|october|november|december)',
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)',
        ]
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in date_patterns)
    
    def _looks_like_header_footer(self, text: str, y_position: float) -> bool:
        """Check if text looks like header/footer content"""
        # Very top or bottom of page
        if y_position < 30 or y_position > 750:
            return True
        
        text_lower = text.lower()
        header_footer_words = ['page', 'chapter', 'confidential', 'draft', 'copyright', '©']
        return any(word in text_lower for word in header_footer_words) and len(text) < 30
    
    def _has_title_patterns(self, text: str) -> bool:
        """Look for patterns that commonly appear in titles"""
        title_indicators = [
            r'\b(introduction|overview|abstract|summary|conclusion)\b',
            r'\b(analysis|study|research|investigation|report)\b',
            r'\b(guide|manual|handbook|tutorial)\b',
            r'^(the|a|an)\s+\w+',  # "The something", "A guide to..."
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in title_indicators)
    
    def _looks_like_complete_title(self, text: str) -> bool:
        """Check if text has the structure of a well-formed title"""
        text = text.strip()
        
        # Common title patterns
        title_patterns = [
            r'^[A-Z][^.]*[^.]$',  # Starts with capital, doesn't end with period
            r'^[A-Z][^:]*:\s*[A-Z]',  # Title with subtitle pattern
            r'^\w+\s+\w+.*\w+$',  # Multi-word title
            r'^(RFP|Request|Application|Report|Study|Analysis|Guide|Manual)',  # Common document types
        ]
        
        for pattern in title_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        # Check word composition - good titles have mostly meaningful words
        words = text.split()
        if len(words) >= 3:
            meaningful_words = [w for w in words if len(w) > 2 and w.lower() not in ['the', 'and', 'for', 'of', 'to', 'in', 'on', 'at']]
            if len(meaningful_words) >= len(words) * 0.6:
                return True
        
        return False
    
    def _try_combine_title_candidates(self, candidates: List[HeadingCandidate]) -> str:
        """
        Sometimes titles are split across multiple candidates.
        This tries to smartly combine them when it makes sense.
        """
        if len(candidates) < 2:
            return ""
        
        # If the best candidate already looks complete, don't mess with it
        best_candidate = candidates[0]
        if (len(best_candidate.text.split()) >= 5 or 
            self._looks_like_complete_title(best_candidate.text)):
            return ""
        
        # Look for other high-confidence candidates on the same page
        first_page_candidates = [c for c in candidates[:4] if c.page == 1 and c.confidence > 0.3]
        
        if len(first_page_candidates) < 2:
            return ""
        
        # Sort by position (top to bottom, left to right)
        first_page_candidates.sort(key=lambda c: (c.position_score, c.text))
        
        # Try combining pairs
        for i in range(len(first_page_candidates) - 1):
            current = first_page_candidates[i]
            next_candidate = first_page_candidates[i + 1]
            
            combined_text = f"{current.text} {next_candidate.text}".strip()
            
            # Only combine if it creates something that looks more like a complete title
            if (len(combined_text.split()) >= 4 and 
                len(combined_text.split()) <= 12 and
                len(combined_text) <= 150 and
                self._looks_like_complete_title(combined_text) and
                not self._looks_like_author_info(combined_text) and
                not self._looks_like_date_info(combined_text)):
                
                return combined_text
        
        return ""
    
    def calculate_heading_confidence(self, block: TextBlock) -> Tuple[float, str]:
        """
        Calculate confidence that a block is a heading and determine its level.
        This is the core logic for finding H1, H2, H3 headings.
        """
        if self._is_obvious_non_heading(block):
            return 0.0, None
        
        confidence = 0.0
        level_indicators = []
        
        # 1. How unique is this font signature?
        block_signature = f"{block.font_name}_{block.font_size}_{block.is_bold}"
        signature_count = len(self.context.font_signatures.get(block_signature, []))
        total_blocks = len(self.context.text_blocks)
        usage_ratio = signature_count / total_blocks if total_blocks > 0 else 0
        
        # Headings should be used sparingly throughout the document
        if usage_ratio < 0.1 and signature_count >= 2:
            confidence += 0.4
        elif usage_ratio < 0.05:
            confidence += 0.3
        elif usage_ratio < 0.02:
            confidence += 0.2
        
        # 2. Font size relative to body text
        if self.context.body_text_signature:
            body_blocks = self.context.font_signatures.get(self.context.body_text_signature, [])
            if body_blocks:
                body_size = body_blocks[0].font_size
                size_ratio = block.font_size / body_size if body_size > 0 else 1
                
                # Bigger = higher level heading
                if size_ratio > 1.4:
                    confidence += 0.35
                    level_indicators.append(("H1", size_ratio - 1.4))
                elif size_ratio > 1.2:
                    confidence += 0.25
                    level_indicators.append(("H2", size_ratio - 1.2))
                elif size_ratio > 1.05:
                    confidence += 0.15
                    level_indicators.append(("H3", size_ratio - 1.05))
                
                # Bold text at similar size to body might still be a heading
                if block.is_bold and 0.95 <= size_ratio <= 1.1:
                    confidence += 0.2
                    level_indicators.append(("H3", 0.1))
        
        # 3. Look for structural patterns (numbered sections, etc.)
        structure_score = self._analyze_structure_patterns(block)
        confidence += structure_score * 0.4
        
        if structure_score > 0.5:
            text = block.text.strip()
            # Determine level based on numbering depth
            if re.match(r'^\d+\.\s+[A-Z]', text):
                level_indicators.append(("H1", 0.5))
            elif re.match(r'^\d+\.\d+\.\s+[A-Z]', text):
                level_indicators.append(("H2", 0.4))
            elif re.match(r'^\d+\.\d+\.\d+\.\s+[A-Z]', text):
                level_indicators.append(("H3", 0.3))
        
        # 4. Position analysis - where is this text positioned?
        position_score = self._analyze_position(block)
        confidence += position_score * 0.2
        
        # 5. Text characteristics - does it look like a heading?
        text_score = self._analyze_text_characteristics_strict(block)
        confidence += text_score * 0.2
        
        # 6. Consistency check - do other blocks with same formatting also look like headings?
        consistency_bonus = self._check_signature_consistency(block_signature)
        confidence += consistency_bonus * 0.15
        
        # 7. Penalty for things that look like body text
        penalty = self._calculate_false_positive_penalty(block)
        confidence -= penalty
        
        # Figure out the heading level based on all our indicators
        if level_indicators:
            best_level = max(level_indicators, key=lambda x: x[1])[0]
        else:
            # Fall back to confidence-based levels
            if confidence > 0.7:
                best_level = "H1"
            elif confidence > 0.5:
                best_level = "H2"
            else:
                best_level = "H3"
        
        return confidence, best_level
    
    def _analyze_structure_patterns(self, block: TextBlock) -> float:
        """Look for structural patterns that scream 'I am a heading!'"""
        text = block.text.strip()
        score = 0.0
        
        # Numbered sections (1. Introduction, 2.1 Methods, etc.)
        if re.match(r'^\d+\.?\s+[A-Z]', text):
            score += 0.6
        elif re.match(r'^\d+\.\d+\.?\s+[A-Z]', text):
            score += 0.5
        elif re.match(r'^\d+\.\d+\.\d+\.?\s+[A-Z]', text):
            score += 0.4
        
        # Common section headers
        if re.match(r'^(Chapter|Section|Part|Appendix)\s+\d+', text, re.IGNORECASE):
            score += 0.7
        
        # Roman numerals
        if re.match(r'^[IVX]+\.\s+[A-Z]', text):
            score += 0.5
        
        # ALL CAPS (but not too long)
        if text.isupper() and 3 <= len(text) <= 50:
            score += 0.3
        
        # Title Case
        if text.istitle() and len(text.split()) >= 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_position(self, block: TextBlock) -> float:
        """Analyze position relative to document patterns"""
        score = 0.0
        
        left_margin = block.bbox[0]
        typical_left = self.context.positional_patterns['typical_left_margin']
        
        # Headings often align with typical margins or are indented differently
        if abs(left_margin - typical_left) < 10:
            score += 0.3
        elif left_margin < typical_left - 20:  # Outdented
            score += 0.5
        
        top_position = block.bbox[1]
        
        # Higher up on page = more likely to be important
        if top_position < 150:
            score += 0.4
        
        # Consistent positioning across pages suggests headings
        similar_positions = [
            b for b in self.context.text_blocks 
            if abs(b.bbox[1] - top_position) < 20 and b.page_num != block.page_num
        ]
        
        if len(similar_positions) >= 2:
            score += 0.3
        
        return min(score, 1.0)
    
    def _analyze_text_characteristics_strict(self, block: TextBlock) -> float:
        """Stricter text analysis to avoid false positives"""
        text = block.text.strip()
        score = 0.0
        
        length = len(text)
        word_count = len(text.split())
        
        # Sweet spot for heading lengths
        if 10 <= length <= 80 and 2 <= word_count <= 12:
            score += 0.4
        elif 5 <= length <= 120 and 1 <= word_count <= 15:
            score += 0.2
        elif length > 150 or word_count > 20:
            score -= 0.3  # Too long to be a heading
        
        # Headings are usually single "sentences"
        sentences = text.split('.')
        if len(sentences) == 1 or (len(sentences) == 2 and sentences[1].strip() == ''):
            score += 0.2
        elif len(sentences) > 3:
            score -= 0.2
        
        # Headings usually don't end with punctuation
        if not text.endswith('.') and not text.endswith(','):
            score += 0.15
        
        # Paragraph-like starters are red flags
        paragraph_starters = ['the', 'this', 'that', 'these', 'those', 'in', 'on', 'at', 'for', 'with', 'by']
        first_word = text.lower().split()[0] if text.split() else ''
        if first_word in paragraph_starters and word_count > 10:
            score -= 0.2
        
        # Capital first letter is good
        words = text.split()
        if words:
            if words[0][0].isupper():
                score += 0.1
            
            # Transition words suggest this is continuing text, not a heading
            if first_word in ['however', 'moreover', 'furthermore', 'additionally', 'therefore']:
                score -= 0.3
        
        return max(score, 0.0)
    
    def _check_signature_consistency(self, signature: str) -> float:
        """If other blocks with the same formatting also look like headings, that's good"""
        signature_blocks = self.context.font_signatures.get(signature, [])
        if len(signature_blocks) < 2:
            return 0.0
        
        # If this formatting appears on multiple pages, probably headings
        pages = set(block.page_num for block in signature_blocks)
        if len(pages) >= min(3, len(signature_blocks)):
            return 0.3
        elif len(pages) >= 2:
            return 0.2
        
        return 0.0
    
    def _calculate_false_positive_penalty(self, block: TextBlock) -> float:
        """Penalize things that look like body text masquerading as headings"""
        text = block.text.strip()
        penalty = 0.0
        
        # Length penalties
        if len(text) > 200:
            penalty += 0.4
        elif len(text) > 120:
            penalty += 0.2
        
        # Body text patterns
        body_patterns = [
            r'\b(the|this|that|these|those)\b.*\b(is|are|was|were|will|would|can|could|should|may|might)\b',
            r'\b(however|moreover|furthermore|therefore|additionally|consequently)\b',
            r'\b(according to|in order to|as well as|such as|for example|for instance)\b',
        ]
        
        text_lower = text.lower()
        for pattern in body_patterns:
            if re.search(pattern, text_lower):
                penalty += 0.3
                break
        
        # Too much punctuation suggests sentences/paragraphs
        if text.count('.') > 1 or text.count(',') > 2:
            penalty += 0.2
        
        # Numbers and percentages suggest data/results text
        if re.search(r'\b\d+[.,]\d+\b', text) or re.search(r'\b\d+%\b', text):
            penalty += 0.2
        
        # Quotes suggest it's referencing something
        if '"' in text or "'" in text:
            penalty += 0.1
        
        # Words that suggest continuing content
        continuation_words = ['also', 'another', 'other', 'some', 'many', 'most', 'all', 'each', 'every']
        first_word = text.lower().split()[0] if text.split() else ''
        if first_word in continuation_words:
            penalty += 0.2
        
        return penalty
    
    def _is_obvious_non_heading(self, block: TextBlock) -> bool:
        """Rule out things that are obviously not headings"""
        text = block.text.lower().strip()
        
        # Common false positive patterns
        false_positive_patterns = [
            r'^\d+',  # Just a number
            r'^page \d+',  # Page numbers
            r'^\d+\.\d+',  # Decimal numbers
            r'^©.*',  # Copyright
            r'^figure \d+',  # Figure captions
            r'^table \d+',  # Table captions
            r'^[^\w\s]+',  # Just punctuation
            r'^www\.',  # URLs
            r'^http',  # URLs
            r'@.*\.com',  # Emails
        ]
        
        for pattern in false_positive_patterns:
            if re.match(pattern, text):
                return True
        
        # List items and numbered items
        if re.match(r'^[•·\-\*]\s*', text) or re.match(r'^\d+\)\s*', text):
            return True
        
        # Way too long
        if len(block.text) > 300:
            return True
        
        # Too short
        if len(text) <= 2:
            return True
        
        # Common footer patterns
        footer_patterns = ['page', 'chapter', 'section', 'figure', 'table']
        if any(pattern in text for pattern in footer_patterns) and len(text) < 20:
            return True
        
        return False
    
    def _fallback_title_extraction(self) -> str:
        """When all else fails, try these backup strategies for finding a title"""
        first_page_blocks = [b for b in self.context.text_blocks if b.page_num == 1]
        if not first_page_blocks:
            return "Untitled Document"
        
        # Strategy 1: Just grab the largest font on first page
        largest_size = max(block.font_size for block in first_page_blocks)
        largest_blocks = [b for b in first_page_blocks if b.font_size == largest_size]
        
        for block in largest_blocks:
            if (not self._is_obvious_non_title(block) and 
                5 <= len(block.text.strip()) <= 150 and
                block.bbox[1] < 500):  # Not too far down the page
                return block.text.strip()
        
        # Strategy 2: Look for bold text near the top
        bold_blocks = [b for b in first_page_blocks if b.is_bold and b.bbox[1] < 300]
        for block in sorted(bold_blocks, key=lambda x: x.bbox[1]):
            if (not self._is_obvious_non_title(block) and 
                5 <= len(block.text.strip()) <= 100):
                return block.text.strip()
        
        # Strategy 3: Just take the first reasonable text block
        for block in sorted(first_page_blocks, key=lambda x: x.bbox[1]):
            text = block.text.strip()
            if (len(text) >= 5 and len(text) <= 100 and 
                not self._is_obvious_non_title(block) and
                not text.lower().startswith(('page', 'chapter', 'section'))):
                return text
        
        return "Untitled Document"
    
    def _calculate_dynamic_threshold(self) -> float:
        """Adjust threshold based on how complex/varied the document formatting is"""
        total_blocks = len(self.context.text_blocks)
        unique_signatures = len(self.context.font_signatures)
        
        complexity_ratio = unique_signatures / max(total_blocks, 1)
        
        # More varied formatting = lower threshold (easier to find headings)
        # Less varied = higher threshold (be more selective)
        if complexity_ratio > 0.1:
            return 0.35
        elif complexity_ratio > 0.05:
            return 0.45
        else:
            return 0.55
    
    def _advanced_heading_filter(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Advanced filtering to clean up the heading list"""
        if not candidates:
            return []
        
        filtered = []
        
        for i, candidate in enumerate(candidates):
            should_include = True
            
            # Don't include very similar headings on same page
            for existing in filtered:
                if (existing.page == candidate.page and 
                    self._texts_are_similar(existing.text, candidate.text)):
                    should_include = False
                    break
            
            # Remove headings that are just substrings of longer headings
            for j, other in enumerate(candidates):
                if (i != j and candidate.page == other.page and 
                    candidate.text in other.text and 
                    len(candidate.text) < len(other.text) * 0.8):
                    should_include = False
                    break
            
            # Skip obvious continuation text
            if i > 0:
                prev_candidate = candidates[i-1]
                if (prev_candidate.page == candidate.page and 
                    candidate.text.lower().startswith(('and', 'or', 'but', 'also'))):
                    should_include = False
            
            # Content-based filtering
            if self._is_content_false_positive(candidate.text):
                should_include = False
            
            # Confidence threshold (dynamic based on the candidate quality)
            min_confidence = max(0.4, candidate.confidence * 0.7)
            if candidate.confidence < min_confidence:
                should_include = False
            
            if should_include:
                filtered.append(candidate)
        
        return filtered
    
    def _texts_are_similar(self, text1: str, text2: str) -> bool:
        """Check if two texts are suspiciously similar (probably duplicates)"""
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        
        if t1 == t2:
            return True
        
        # One contains the other and they're similar length
        if t1 in t2 or t2 in t1:
            shorter = min(len(t1), len(t2))
            longer = max(len(t1), len(t2))
            if shorter / longer > 0.8:
                return True
        
        # High word overlap
        words1 = set(t1.split())
        words2 = set(t2.split())
        if words1 and words2:
            overlap = len(words1.intersection(words2))
            total_unique = len(words1.union(words2))
            if overlap / total_unique > 0.7:
                return True
        
        return False
    
    def _is_content_false_positive(self, text: str) -> bool:
        """Additional content-based filtering for false positives"""
        text_lower = text.lower().strip()
        
        # Patterns that suggest this isn't a heading
        false_patterns = [
            r'^(fig|figure|table|chart|diagram)\s*\d*[:\.]',
            r'^(see|refer to|as shown|as described)',
            r'^(note|warning|caution|important):',
            r'^\([a-z]\)',  # (a), (b), (c) style lists
            r'^\d+\)',  # 1), 2), 3) style lists
            r'^[•·▪▫]\s',  # Bullet points
        ]
        
        for pattern in false_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Mathematical content
        if re.search(r'\b(equation|formula|algorithm)\s+\d+', text_lower):
            return True
        
        # Citations and references
        if re.search(r'\[\d+\]|\(\d{4}\)', text):
            return True
        
        # Repetitive text (might be artifacts)
        words = text_lower.split()
        if len(words) > 3:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.5:  # Too much repetition
                return True
        
        return False
    
    def _remove_title_duplicates(self, headings: List[HeadingCandidate], title: str) -> List[HeadingCandidate]:
        """Make sure we don't include the title again as a heading"""
        if not title or not headings:
            return headings
        
        title_lower = title.lower().strip()
        filtered_headings = []
        
        for heading in headings:
            heading_lower = heading.text.lower().strip()
            
            # Exact match - definitely skip
            if heading_lower == title_lower:
                continue
            
            # Substantial overlap - probably the same thing
            if (heading_lower in title_lower and len(heading_lower) > len(title_lower) * 0.7) or \
               (title_lower in heading_lower and len(title_lower) > len(heading_lower) * 0.7):
                continue
            
            # Check word overlap for similar titles
            title_words = set(title_lower.split())
            heading_words = set(heading_lower.split())
            
            if title_words and heading_words:
                overlap = len(title_words.intersection(heading_words))
                total_unique = len(title_words.union(heading_words))
                
                # High overlap + similar length = probably duplicate
                if (overlap / total_unique > 0.8 and 
                    abs(len(title_words) - len(heading_words)) <= 2):
                    continue
            
            # Prefix matching for split titles
            if (len(heading_lower) >= 10 and len(title_lower) >= 10):
                if (heading_lower.startswith(title_lower[:10]) or 
                    title_lower.startswith(heading_lower[:10])):
                    # Count common prefix
                    min_len = min(len(heading_lower), len(title_lower))
                    common_prefix_len = 0
                    for i in range(min_len):
                        if heading_lower[i] == title_lower[i]:
                            common_prefix_len += 1
                        else:
                            break
                    
                    if common_prefix_len > min_len * 0.6:
                        continue
            
            filtered_headings.append(heading)
        
        return filtered_headings
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """
        Main processing function - this orchestrates the whole pipeline.
        Extract text -> build context -> find title -> find headings -> clean up results
        """
        print(f"Processing: {pdf_path}")
        
        # Step 1: Extract all text blocks with formatting info
        text_blocks = self.extract_text_blocks(pdf_path)
        if not text_blocks:
            return {"title": "Empty Document", "outline": []}
        
        # Step 2: Build document context (learn the document's patterns)
        self.context = DocumentContext(text_blocks)
        
        # Step 3: Extract title with multiple strategies
        title_candidates = self.extract_title_candidates()
        
        if title_candidates:
            best_title = title_candidates[0].text
            
            # Try to find a better title by combining fragments
            if len(title_candidates) > 1:
                combined_title = self._try_combine_title_candidates(title_candidates)
                if combined_title and len(combined_title) > len(best_title):
                    best_title = combined_title
            
            # If the best candidate seems weak, try alternatives
            if (title_candidates[0].confidence < 0.3 and len(title_candidates) > 1):
                for candidate in title_candidates[1:4]:
                    if candidate.confidence > 0.2:
                        best_title = candidate.text
                        break
            
            title = best_title
        else:
            # Fallback strategies when main title extraction fails
            title = self._fallback_title_extraction()
        
        # Step 4: Find headings throughout the document
        heading_candidates = []
        for block in text_blocks:
            confidence, level = self.calculate_heading_confidence(block)
            
            # Use dynamic threshold based on document complexity
            threshold = self._calculate_dynamic_threshold()
            
            if confidence > threshold and level:
                heading_candidates.append(HeadingCandidate(
                    text=block.text.strip(),
                    page=block.page_num,
                    confidence=confidence,
                    level=level,
                    font_signature=f"{block.font_name}_{block.font_size}_{block.is_bold}",
                    position_score=0.0,
                    uniqueness_score=0.0
                ))
        
        # Step 5: Sort and clean up the heading list
        heading_candidates.sort(key=lambda h: (h.page, -h.confidence))
        
        # Remove duplicates and false positives
        filtered_headings = self._advanced_heading_filter(heading_candidates)
        
        # Make sure we don't duplicate the title as a heading
        filtered_headings = self._remove_title_duplicates(filtered_headings, title)
        
        # Step 6: Build the final outline structure
        outline = []
        for heading in filtered_headings:
            outline.append({
                "level": heading.level,
                "text": heading.text,
                "page": heading.page
            })
        
        return {
            "title": title,
            "outline": outline
        }

def process_directory(input_dir: str, output_dir: str):
    """Process all PDFs in the input directory and save results to output directory"""
    extractor = AdaptivePDFExtractor()
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            output_filename = filename.replace('.pdf', '.json')
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                result = extractor.process_pdf(pdf_path)
                
                # Save the result as JSON
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Processed {filename} -> {output_filename}")
                print(f"   Title: {result['title']}")
                print(f"   Headings found: {len(result['outline'])}")
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")
                # Create empty result for failed processing
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({"title": "Processing Failed", "outline": []}, f)

if __name__ == "__main__":
    input_directory = "/app/input"
    output_directory = "/app/output"
    
    process_directory(input_directory, output_directory)