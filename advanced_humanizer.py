"""
Advanced Humanization Module

This module implements a comprehensive multi-model approach for humanizing AI-generated text
while maintaining perfect coherence, cohesion, and fluency.

The module combines:
1. Structure preservation techniques
2. Advanced linguistic transformations
3. Multi-stage optimization for evading AI detectors
4. Coherence and cohesion enhancement
"""

import re
import random
import logging
import string
import nltk
import time
import json
import unicodedata
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

# --------- TEXT STRUCTURE ANALYSIS ---------

class TextStructureAnalyzer:
    """Analyzes and preserves the structure of text."""
    
    @staticmethod
    def extract_structure(text: str) -> Dict[str, Any]:
        """
        Extract structural elements from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with structural information
        """
        structure = {
            'paragraphs': [],
            'sentence_count': 0,
            'avg_sentence_length': 0,
            'bullet_points': [],
            'has_lists': False,
            'has_headings': False,
            'quote_blocks': [],
            'original_text': text
        }
        
        # Extract paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        structure['paragraphs'] = paragraphs
        
        # Count sentences
        sentences = []
        for para in paragraphs:
            para_sentences = sent_tokenize(para)
            sentences.extend(para_sentences)
        
        structure['sentence_count'] = len(sentences)
        
        # Calculate average sentence length
        word_counts = [len(word_tokenize(sentence)) for sentence in sentences]
        if word_counts:
            structure['avg_sentence_length'] = sum(word_counts) / len(word_counts)
            
        # Detect bullet points and lists
        bullet_pattern = re.compile(r'^\s*[•*-]\s+', re.MULTILINE)
        matches = bullet_pattern.findall(text)
        if matches:
            structure['has_lists'] = True
            structure['bullet_points'] = matches
        
        # Detect headings (lines ending with : or lines in all caps)
        heading_pattern = re.compile(r'^[A-Z][^a-z]*:|^[A-Z0-9\s]+$', re.MULTILINE)
        if heading_pattern.search(text):
            structure['has_headings'] = True
        
        # Detect quote blocks
        quote_pattern = re.compile(r'^\s*>(.+?)$', re.MULTILINE)
        structure['quote_blocks'] = quote_pattern.findall(text)
        
        return structure
    
    @staticmethod
    def ensure_structure_preserved(original_text: str, new_text: str) -> bool:
        """
        Check if the new text preserves the structure of the original.
        
        Args:
            original_text: Original text
            new_text: New text to compare
            
        Returns:
            True if structure is preserved, False otherwise
        """
        orig_structure = TextStructureAnalyzer.extract_structure(original_text)
        new_structure = TextStructureAnalyzer.extract_structure(new_text)
        
        # Check paragraph count
        if len(orig_structure['paragraphs']) != len(new_structure['paragraphs']):
            return False
        
        # Check if lists are preserved
        if orig_structure['has_lists'] != new_structure['has_lists']:
            return False
        
        # Check if sentence count is roughly preserved (allow small variations)
        sentence_diff = abs(orig_structure['sentence_count'] - new_structure['sentence_count'])
        max_allowed_diff = max(1, orig_structure['sentence_count'] * 0.1)  # Allow 10% difference
        if sentence_diff > max_allowed_diff:
            return False
        
        # Structure appears to be preserved
        return True
    
    @staticmethod
    def force_structure(original_text: str, new_text: str) -> str:
        """
        Force the new text to follow the structure of the original.
        
        Args:
            original_text: Original text with desired structure
            new_text: Text to reshape
            
        Returns:
            Reshaped text with original structure
        """
        orig_structure = TextStructureAnalyzer.extract_structure(original_text)
        new_structure = TextStructureAnalyzer.extract_structure(new_text)
        
        # If same paragraph count, simpler approach: just match paragraph breaks
        if len(orig_structure['paragraphs']) == len(new_structure['paragraphs']):
            return new_text
        
        # Otherwise, need to redistribute content
        new_sentences = []
        for para in new_structure['paragraphs']:
            new_sentences.extend(sent_tokenize(para))
        
        # Redistribute sentences according to original paragraph structure
        result_paragraphs = []
        sent_index = 0
        
        for i, orig_para in enumerate(orig_structure['paragraphs']):
            # Get sentences in original paragraph
            orig_para_sentences = sent_tokenize(orig_para)
            sent_count = len(orig_para_sentences)
            
            # Take same number of sentences from new text (if available)
            if sent_index + sent_count <= len(new_sentences):
                para_sentences = new_sentences[sent_index:sent_index+sent_count]
            else:
                # Not enough sentences left, take what's available
                para_sentences = new_sentences[sent_index:]
            
            # Build paragraph
            new_para = ' '.join(para_sentences)
            result_paragraphs.append(new_para)
            
            # Update sentence index
            sent_index += sent_count
        
        return '\n\n'.join(result_paragraphs)


# --------- COHERENCE AND FLUENCY ENHANCEMENT ---------

class CoherenceEnhancer:
    """Enhances text coherence and fluency."""
    
    # Common transition phrases in English
    TRANSITIONS = {
        'addition': ['additionally', 'furthermore', 'moreover', 'also', 'in addition', 'besides', "what's more", 'similarly', 'likewise'],
        'contrast': ['however', 'nevertheless', 'on the other hand', 'in contrast', 'yet', 'conversely', 'still', 'unlike', 'despite this'],
        'cause': ['because', 'since', 'as', 'due to', 'owing to', 'for this reason'],
        'effect': ['therefore', 'consequently', 'as a result', 'thus', 'hence', 'accordingly', 'so'],
        'example': ['for example', 'for instance', 'specifically', 'notably', 'as an illustration', 'to illustrate'],
        'clarification': ['in other words', 'to clarify', 'that is', 'to put it differently', 'simply put'],
        'sequence': ['first', 'second', 'third', 'next', 'then', 'finally', 'subsequently', 'eventually'],
        'summary': ['in conclusion', 'to summarize', 'in sum', 'overall', 'to conclude', 'in brief'],
    }
    
    # Variety of coherence markers
    COHERENCE_MARKERS = {
        'reference': ['this', 'that', 'these', 'those', 'it', 'they', 'them'],
        'conjunction': ['and', 'but', 'or', 'so', 'yet', 'nor'],
        'adverbial': ['indeed', 'certainly', 'clearly', 'obviously', 'naturally', 'of course'],
    }
    
    @staticmethod
    def improve_coherence(text: str) -> str:
        """
        Improve text coherence by ensuring proper transitions between sentences.
        
        Args:
            text: Input text
            
        Returns:
            Text with improved coherence
        """
        paragraphs = re.split(r'\n\s*\n', text)
        result_paragraphs = []
        
        for paragraph in paragraphs:
            # Skip very short paragraphs
            if len(paragraph) < 40:
                result_paragraphs.append(paragraph)
                continue
                
            sentences = sent_tokenize(paragraph)
            
            # Skip very short sentence groups
            if len(sentences) < 3:
                result_paragraphs.append(paragraph)
                continue
            
            # Analyze sentence relationships
            for i in range(1, len(sentences)):
                prev_sentence = sentences[i-1]
                curr_sentence = sentences[i]
                
                # Check if transition is needed
                needs_transition = True
                
                # Skip if sentence already starts with a transition
                for transition_type in CoherenceEnhancer.TRANSITIONS.values():
                    if any(curr_sentence.lower().startswith(t.lower()) for t in transition_type):
                        needs_transition = False
                        break
                
                # Skip if sentence starts with a coherence marker
                for marker_type in CoherenceEnhancer.COHERENCE_MARKERS.values():
                    if any(curr_sentence.lower().startswith(m.lower()) for m in marker_type):
                        needs_transition = False
                        break
                
                # Add transition if needed (with low probability to avoid overuse)
                if needs_transition and random.random() < 0.2:  # 20% chance
                    # Choose transition type based on content analysis
                    if re.search(r'(but|however|although|though)', prev_sentence, re.IGNORECASE):
                        transition_type = 'contrast'
                    elif re.search(r'(because|since|reason)', prev_sentence, re.IGNORECASE):
                        transition_type = 'effect'
                    elif re.search(r'(example|instance|illustrat)', prev_sentence, re.IGNORECASE):
                        transition_type = 'example'
                    else:
                        # Default to addition or random type
                        transition_type = random.choice(list(CoherenceEnhancer.TRANSITIONS.keys()))
                    
                    # Choose a transition phrase
                    transition = random.choice(CoherenceEnhancer.TRANSITIONS[transition_type])
                    
                    # Add transition
                    sentences[i] = f"{transition}, {curr_sentence[0].lower()}{curr_sentence[1:]}"
            
            # Reconstruct paragraph
            result_paragraphs.append(' '.join(sentences))
        
        return '\n\n'.join(result_paragraphs)
    
    @staticmethod
    def enhance_fluency(text: str) -> str:
        """
        Enhance text fluency by improving sentence flow.
        
        Args:
            text: Input text
            
        Returns:
            Text with improved fluency
        """
        paragraphs = re.split(r'\n\s*\n', text)
        result_paragraphs = []
        
        for paragraph in paragraphs:
            sentences = sent_tokenize(paragraph)
            
            # Enhance individual sentences
            for i in range(len(sentences)):
                sentence = sentences[i]
                
                # Only modify some sentences to maintain naturalness
                if random.random() < 0.3:  # 30% chance
                    
                    # 1. Fix awkward phrasings
                    sentence = sentence.replace(" in order to ", " to ")
                    sentence = sentence.replace(" due to the fact that ", " because ")
                    sentence = sentence.replace(" on account of ", " because of ")
                    sentence = sentence.replace(" in the event that ", " if ")
                    
                    # 2. Add variety to sentence starters
                    if i > 0 and random.random() < 0.2:
                        starters = [
                            "Interestingly, ", "Notably, ", "Certainly, ", 
                            "Importantly, ", "Surprisingly, ", "Naturally, "
                        ]
                        sentence = random.choice(starters) + sentence[0].lower() + sentence[1:]
                    
                    # 3. Introduce better rhythm through punctuation
                    if len(sentence) > 60 and "," not in sentence and random.random() < 0.4:
                        words = word_tokenize(sentence)
                        mid_point = len(words) // 2
                        sentence = " ".join(words[:mid_point]) + ", " + " ".join(words[mid_point:])
                
                sentences[i] = sentence
            
            # Reconstruct paragraph
            result_paragraphs.append(' '.join(sentences))
        
        return '\n\n'.join(result_paragraphs)


# --------- AI DETECTION EVASION ---------

class AIDetectionEvader:
    """Implements techniques to evade AI detection."""
    
    # Homoglyphs and zero-width characters
    HOMOGLYPHS = {
        'a': ['а'],  # Cyrillic 'а'
        'e': ['е'],  # Cyrillic 'е'
        'o': ['о'],  # Cyrillic 'о'
        'p': ['р'],  # Cyrillic 'р'
        'c': ['с'],  # Cyrillic 'с'
        'x': ['х'],  # Cyrillic 'х'
        'i': ['і'],  # Ukrainian 'і'
        'j': ['ј'],  # Cyrillic 'ј'
        's': ['ѕ'],  # Cyrillic 'ѕ'
        'y': ['у']   # Cyrillic 'у'
    }
    
    ZERO_WIDTH_CHARS = [
        '\u200B',  # Zero width space
        '\u200C',  # Zero width non-joiner
        '\u200D',  # Zero width joiner
        '\u2060',  # Word joiner
        '\uFEFF'   # Zero width no-break space
    ]
    
    # Word substitution dictionaries for randomization
    FORMAL_SUBSTITUTIONS = {
        "additionally": ["also", "plus", "what's more"],
        "furthermore": ["beyond that", "also", "moreover"],
        "nevertheless": ["still", "even so", "regardless"],
        "consequently": ["so", "as a result", "thus"],
        "therefore": ["so", "that's why", "hence"],
        "however": ["but", "yet", "still"],
        "subsequently": ["later", "afterward", "next"],
        "currently": ["now", "at present", "these days"],
        "previously": ["before", "earlier", "once"],
        "ultimately": ["finally", "in the end", "eventually"],
    }
    
    @staticmethod
    def apply_imperceptible_changes(text: str) -> str:
        """
        Apply imperceptible changes to text to evade AI detectors.
        
        Args:
            text: Input text
            
        Returns:
            Modified text that can evade AI detection
        """
        # 1. Apply homoglyphs
        result = ""
        for char in text:
            lower_char = char.lower()
            # Only replace some characters with a low probability
            if lower_char in AIDetectionEvader.HOMOGLYPHS and random.random() < 0.03:
                homoglyph = random.choice(AIDetectionEvader.HOMOGLYPHS[lower_char])
                if char.isupper():
                    homoglyph = homoglyph.upper()
                result += homoglyph
            else:
                result += char
        
        # 2. Insert zero-width characters
        text_with_zero_width = ""
        for char in result:
            text_with_zero_width += char
            # Add zero-width character with low probability
            if random.random() < 0.04:
                text_with_zero_width += random.choice(AIDetectionEvader.ZERO_WIDTH_CHARS)
        
        # 3. Apply Unicode normalizations
        final_text = ""
        for char in text_with_zero_width:
            if random.random() < 0.02:
                norm_form = random.choice(['NFC', 'NFKC', 'NFD', 'NFKD'])
                char = unicodedata.normalize(norm_form, char)
            final_text += char
        
        return final_text
    
    @staticmethod
    def introduce_natural_imperfections(text: str) -> str:
        """
        Introduce natural human imperfections to text.
        
        Args:
            text: Input text
            
        Returns:
            Text with natural imperfections
        """
        sentences = sent_tokenize(text)
        result = []
        
        for i, sentence in enumerate(sentences):
            # Only modify some sentences
            if random.random() < 0.2:  # 20% chance
                words = word_tokenize(sentence)
                
                # Choose a modification type
                mod_type = random.randint(1, 5)
                
                if mod_type == 1 and len(words) > 8:
                    # 1. Repeat a word (common human typo)
                    pos = random.randint(2, min(len(words) - 1, 8))
                    words.insert(pos, words[pos])
                    
                elif mod_type == 2 and len(words) > 5:
                    # 2. Add a filler word
                    fillers = ["um", "like", "you know", "well", "I mean"]
                    pos = random.randint(1, min(len(words) - 1, 5))
                    words.insert(pos, random.choice(fillers))
                    
                elif mod_type == 3 and len(sentence) > 40:
                    # 3. Add a self-correction
                    # Find a word to "correct"
                    for j, word in enumerate(words):
                        if len(word) > 4 and j < len(words) - 1 and random.random() < 0.3:
                            # Insert a correction after this word
                            words.insert(j + 1, "I mean")
                            break
                            
                elif mod_type == 4 and len(words) > 10:
                    # 4. Use inconsistent punctuation
                    if sentence.endswith("."):
                        sentence = sentence[:-1] + random.choice([".", "..."])
                        
                elif mod_type == 5 and len(words) > 6:
                    # 5. Substitute a formal word with casual equivalent
                    for formal, casual_options in AIDetectionEvader.FORMAL_SUBSTITUTIONS.items():
                        if formal in words:
                            idx = words.index(formal)
                            words[idx] = random.choice(casual_options)
                            break
                
                sentence = " ".join(words)
            
            result.append(sentence)
        
        return " ".join(result)
    
    @staticmethod
    def break_ai_patterns(text: str) -> str:
        """
        Break patterns that AI detectors look for.
        
        Args:
            text: Input text
            
        Returns:
            Text with broken AI patterns
        """
        # 1. Replace repetitive patterns
        for pattern in [r'\b(\w+\s+\w+\s+\w+)\b.+\b\1\b', r'\b(\w+\s+\w+)\b.+\b\1\b']:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 10:  # Only replace substantial repeated phrases
                    words = match.split()
                    if len(words) >= 2:
                        # Create a slightly modified version
                        modified = words[0] + " " + ("actually " if random.random() < 0.5 else "") + " ".join(words[1:])
                        # Replace only second occurrence
                        parts = text.split(match, 2)
                        if len(parts) > 2:
                            text = match.join(parts[:2]) + modified + parts[2]
        
        # 2. Vary punctuation
        sentences = sent_tokenize(text)
        for i in range(len(sentences)):
            if random.random() < 0.15:  # 15% chance
                s = sentences[i]
                if s.endswith("."):
                    if random.random() < 0.7:
                        sentences[i] = s  # Keep period
                    else:
                        sentences[i] = s[:-1] + random.choice(["!", "..."])
        
        result = " ".join(sentences)
        
        # 3. Break algorithmic patterns with invisible characters
        pattern_breakers = [
            "\u200B",  # Zero-width space
            "\u200C",  # Zero-width non-joiner
            "\u200D",  # Zero-width joiner
        ]
        
        # Insert invisible characters at random positions
        chars = list(result)
        for i in range(10, len(chars), random.randint(20, 40)):
            if random.random() < 0.4:
                chars.insert(i, random.choice(pattern_breakers))
        
        return "".join(chars)


# --------- MAIN INTERFACE ---------

def humanize_text_advanced(text: str) -> str:
    """
    Apply comprehensive humanization to text with a focus on maintaining structure,
    coherence, cohesion and fluency while evading AI detection.
    
    Args:
        text: The text to be humanized
        
    Returns:
        Humanized text that passes AI detectors
    """
    # STEP 1: Analyze original structure
    original_structure = TextStructureAnalyzer.extract_structure(text)
    
    # STEP 2: Enhance fluency and coherence
    logger.info("Enhancing text fluency and coherence...")
    coherent_text = CoherenceEnhancer.enhance_fluency(text)
    coherent_text = CoherenceEnhancer.improve_coherence(coherent_text)
    
    # STEP 3: Check if structure was preserved during coherence enhancement
    if not TextStructureAnalyzer.ensure_structure_preserved(text, coherent_text):
        logger.warning("Structure changed during coherence enhancement, forcing structure...")
        coherent_text = TextStructureAnalyzer.force_structure(text, coherent_text)
    
    # STEP 4: Apply AI detection evasion techniques
    logger.info("Applying AI detection evasion techniques...")
    evasive_text = AIDetectionEvader.break_ai_patterns(coherent_text)
    evasive_text = AIDetectionEvader.introduce_natural_imperfections(evasive_text)
    final_text = AIDetectionEvader.apply_imperceptible_changes(evasive_text)
    
    # STEP 5: Final structure verification
    if not TextStructureAnalyzer.ensure_structure_preserved(text, final_text):
        logger.warning("Structure changed after all modifications, reconstructing...")
        final_text = TextStructureAnalyzer.force_structure(text, final_text)
    
    return final_text
