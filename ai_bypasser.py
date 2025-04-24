"""
AI Bypasser Module

This module implements scientific techniques to make AI-generated text pass as human-written while
preserving the exact structure of the original text.

Based on research papers:
1. "Detecting AI-Generated Text via Watermarking and How to Bypass Detection" (2023)
2. "Evading AI-Generated Text Detection" (2023)
3. "Character-level Perturbations for Fooling AI Classifiers" (2022)
"""

import re
import random
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import unicodedata

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Mapping for homoglyphs (characters that look similar but have different Unicode)
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

# Zero-width characters that are invisible but can disrupt patterns
ZERO_WIDTH_CHARS = [
    '\u200B',  # Zero width space
    '\u200C',  # Zero width non-joiner
    '\u200D',  # Zero width joiner
    '\u2060',  # Word joiner
    '\uFEFF'   # Zero width no-break space
]

# List of subtle markdown-safe style modifiers
STYLE_MODIFIERS = [
    '​',  # Zero-width space
    ' ',  # Thin space
    ' ',  # Hair space
    ' ',  # Narrow no-break space
]

def apply_imperceptible_homoglyphs(text, probability=0.02):
    """
    Replace some characters with their homoglyphs (look-alikes) to defeat AI detectors
    while keeping the text visually identical.
    
    Args:
        text: The input text
        probability: Probability of replacing a character (low values to keep it subtle)
        
    Returns:
        Modified text with some homoglyphs
    """
    result = ""
    for char in text:
        lower_char = char.lower()
        # Only replace some characters with a certain probability
        if lower_char in HOMOGLYPHS and random.random() < probability:
            # Get homoglyph and maintain original case
            homoglyph = random.choice(HOMOGLYPHS[lower_char])
            if char.isupper():
                homoglyph = homoglyph.upper()
            result += homoglyph
        else:
            result += char
    return result

def insert_zero_width_characters(text, probability=0.03):
    """
    Insert zero-width characters at random positions to disrupt pattern detection.
    These characters are invisible and don't affect display.
    
    Args:
        text: The input text
        probability: Probability of inserting a zero-width char after each character
        
    Returns:
        Modified text with zero-width characters
    """
    result = ""
    for char in text:
        result += char
        # Randomly insert zero-width character
        if random.random() < probability:
            result += random.choice(ZERO_WIDTH_CHARS)
    return result

def apply_typography_variations(text, probability=0.02):
    """
    Apply subtle typographic variations that are imperceptible
    but disrupt AI detection patterns.
    
    Args:
        text: The input text
        probability: Probability of applying a variation to each word
        
    Returns:
        Modified text with typography variations
    """
    words = text.split()
    result = []
    
    for word in words:
        if len(word) > 4 and random.random() < probability:
            # Insert a zero-width character between two characters
            pos = random.randint(1, len(word) - 1)
            word = word[:pos] + random.choice(STYLE_MODIFIERS) + word[pos:]
        result.append(word)
    
    return ' '.join(result)

def apply_unicode_normalizations(text, probability=0.015):
    """
    Apply Unicode normalizations to certain characters.
    This doesn't visually change the text but alters its digital representation.
    
    Args:
        text: The input text
        probability: Probability of normalizing a character
        
    Returns:
        Modified text with some characters normalized differently
    """
    result = ""
    for char in text:
        if random.random() < probability:
            # Apply a random normalization form
            norm_form = random.choice(['NFC', 'NFKC', 'NFD', 'NFKD'])
            char = unicodedata.normalize(norm_form, char)
        result += char
    return result

def reroute_sentence_flow(text, probability=0.15):
    """
    Apply sentence flow changes that preserve structure but
    make it harder for AI detectors to identify patterns.
    
    This function rewrites sentences in ways that obscure AI detection
    while maintaining the original meaning and structure.
    
    Args:
        text: The input text
        probability: Probability of modifying a sentence
        
    Returns:
        Modified text with subtle rewrites
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    
    for sentence in sentences:
        if random.random() > probability or len(sentence) < 10:
            result.append(sentence)
            continue
            
        tokens = word_tokenize(sentence)
        
        # Find adjectives or descriptive words that can be slightly modified
        for i in range(len(tokens)):
            if len(tokens) > 5 and i > 0 and i < len(tokens) - 1:
                word = tokens[i]
                
                # Words longer than 5 chars could be candidates for minor modifications
                if len(word) > 5 and word.isalpha() and random.random() < 0.3:
                    alternatives = {
                        'very': ['really', 'quite', 'rather'],
                        'extremely': ['incredibly', 'tremendously', 'exceedingly'],
                        'good': ['great', 'excellent', 'fine'],
                        'bad': ['poor', 'terrible', 'awful'],
                        'big': ['large', 'huge', 'enormous'],
                        'small': ['tiny', 'little', 'minute'],
                        'interesting': ['intriguing', 'fascinating', 'compelling'],
                        'important': ['crucial', 'essential', 'critical'],
                        'necessary': ['essential', 'required', 'needed'],
                        'difficult': ['challenging', 'hard', 'tough'],
                        'easy': ['simple', 'straightforward', 'effortless']
                    }
                    
                    word_lower = word.lower()
                    if word_lower in alternatives:
                        tokens[i] = random.choice(alternatives[word_lower])
        
        # Reconstruct sentence with modified words
        modified_sentence = TreebankWordDetokenizer().detokenize(tokens)
        result.append(modified_sentence)
    
    return ' '.join(result)

def make_human_like(text):
    """
    Apply multiple transformations to make AI-generated text appear human-written
    while preserving original structure.
    
    Args:
        text: The input text to be humanized
        
    Returns:
        Text that preserves structure but appears human-written to AI detectors
    """
    # Stage 1: Apply homoglyphs substitution (extremely subtle, invisible changes)
    text = apply_imperceptible_homoglyphs(text)
    
    # Stage 2: Insert zero-width characters to disrupt patterns
    text = insert_zero_width_characters(text)
    
    # Stage 3: Apply typography variations
    text = apply_typography_variations(text)
    
    # Stage 4: Apply unicode normalizations
    text = apply_unicode_normalizations(text)
    
    # Stage 5: Reroute sentence flow with minimal changes
    text = reroute_sentence_flow(text)
    
    return text

def is_ai_text(text):
    """
    Placeholder function for AI detection testing.
    In a real system, this would connect to an AI detector API.
    
    Args:
        text: Text to check
        
    Returns:
        Probability of AI generation (0-1)
    """
    # This is just a placeholder - in a real system, you would
    # connect to an actual AI detector API
    return 0.5  # Simulated 50% probability

def optimize_evasion(text, iterations=3):
    """
    Iteratively apply transformations to ensure the text passes AI detectors.
    
    Args:
        text: Input text
        iterations: Number of optimization iterations
        
    Returns:
        Optimized text that better passes AI detection
    """
    current_text = text
    best_text = text
    best_score = is_ai_text(text)
    
    for i in range(iterations):
        # Apply transformations with varying intensity
        modified_text = make_human_like(current_text)
        
        # Check the modified text with simulated AI detector
        score = is_ai_text(modified_text)
        
        # If better, keep it
        if score < best_score:
            best_score = score
            best_text = modified_text
        
        current_text = modified_text
    
    return best_text
