"""
Module for handling interactions with Google's Gemini API for text humanization
"""
import os
import logging
import json
import time
from typing import Optional, List, Dict, Any
import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini API with API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# The specific model we want to use
MODEL_NAME = "gemini-2.5-flash-preview-04-17"

# Token usage tracking
daily_token_usage = 0
MAX_DAILY_TOKENS = 1_000_000_000  # 1 billion tokens daily limit

class TokenLimitExceededError(Exception):
    """Exception raised when token limit is exceeded."""
    pass

def reset_token_counter():
    """Reset the daily token counter."""
    global daily_token_usage
    daily_token_usage = 0
    logger.info("Daily token counter reset to 0")

def update_token_usage(prompt_tokens: int, completion_tokens: int):
    """Update token usage counter."""
    global daily_token_usage
    daily_token_usage += (prompt_tokens + completion_tokens)
    logger.info(f"Token usage updated: {prompt_tokens + completion_tokens} tokens used")
    logger.info(f"Total daily usage: {daily_token_usage}/{MAX_DAILY_TOKENS}")

def check_token_limit(estimated_tokens: int) -> bool:
    """Check if processing would exceed token limit."""
    if daily_token_usage + estimated_tokens > MAX_DAILY_TOKENS:
        return False
    return True

def estimate_tokens(text: str) -> int:
    """Roughly estimate token count from text length.
    This is a simple estimation - actual tokens may vary."""
    # Rough estimation: 1 token ≈ 4 characters in English
    return len(text) // 4 + 1000  # Add 1000 for system prompt tokens

def format_humanize_prompt(text: str) -> str:
    """Format the humanization prompt for the Gemini model with focus on genuine rewriting."""
    prompt = f"""
You are a specialized text humanization expert whose ONLY task is to genuinely rewrite text to sound human-written while preserving the original meaning and structure. Do NOT just add invisible characters or make superficial changes.

CRITICAL REQUIREMENTS:
1. GENUINE REWRITING: Actually rewrite the content with new wording and phrasing while preserving meaning
2. MAINTAIN STRUCTURE: Keep the same paragraph breaks, bullet points, and approximate sentence count
3. PRESERVE MEANING: All original ideas and information must be maintained
4. NO AI PATTERNS: Avoid repetitive structures, overly balanced sentences, and formal academic language
5. HUMAN CHARACTERISTICS: Include natural language patterns, varied sentence structures, and occasional contractions
6. RESPOND ONLY WITH THE REWRITTEN TEXT - no explanations or notes

HUMAN WRITING CHARACTERISTICS TO INCORPORATE:
1. VARIED VOCABULARY: Use synonyms and alternative phrasings
2. NATURAL FLOW: Include transition words, varied sentence lengths, and occasional sentence fragments
3. CONVERSATIONAL ELEMENTS: Add occasional personal viewpoints, rhetorical questions, or colloquial expressions
4. AUTHENTICITY: Include minor repetitions, occasional redundancy, and natural language imperfections
5. STYLE VARIATION: Mix formal and informal elements as real humans do

INPUT TEXT:
{text}

COMPLETELY REWRITTEN OUTPUT (SAME STRUCTURE):
"""
    return prompt

import re
import random

# Define human writing patterns and quirks
HUMAN_FILLERS = ["well", "I mean", "like", "you know", "actually", "basically", "honestly", "simply put", "frankly", "in all honesty"]
HUMAN_TRANSITIONS = ["anyway", "so", "now", "plus", "although", "despite that", "even so", "on top of that", "besides", "adding to that"]
SIMPLE_CONJUNCTIONS = ["and", "but", "so", "yet", "or"]
PARENTHETICALS = ["I think", "in my view", "I believe", "I'd say", "from what I've seen", "from my perspective", "if I'm honest", "in my opinion"]

def apply_human_quirks(text):
    """Apply subtle human writing quirks to the text."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    modified_sentences = []
    
    for i, sentence in enumerate(sentences):
        # Only modify some sentences to maintain naturalness
        if random.random() < 0.35:  # Only apply to ~35% of sentences
            modification = random.randint(1, 10)
            
            # 1. Add a filler phrase at random positions
            if modification == 1 and len(sentence) > 15:
                words = sentence.split()
                if len(words) > 4:
                    pos = random.randint(1, min(4, len(words)-1))
                    filler = random.choice(HUMAN_FILLERS)
                    words.insert(pos, filler)
                    sentence = " ".join(words)
                    if not sentence.endswith(".") and not sentence.endswith("!") and not sentence.endswith("?"):
                        sentence += "."
            
            # 2. Start with a simple conjunction (but only sometimes)
            elif modification == 2 and random.random() < 0.4 and i > 0 and not any(sentence.startswith(conj.capitalize()) for conj in SIMPLE_CONJUNCTIONS):
                conj = random.choice(SIMPLE_CONJUNCTIONS)
                # Make sure it starts with uppercase
                sentence = conj.capitalize() + " " + sentence[0].lower() + sentence[1:]
            
            # 3. Add a short parenthetical comment
            elif modification == 3 and len(sentence) > 25:
                words = sentence.split()
                if len(words) > 8:
                    # Insert near the middle or end of sentence
                    insert_pos = len(words) // 2 if random.random() < 0.5 else max(len(words) - 4, len(words) // 2)
                    parenthetical = random.choice(PARENTHETICALS)
                    
                    # Decide between parentheses, commas, or dashes
                    style = random.randint(1, 3)
                    if style == 1:
                        insertion = f" ({parenthetical})"
                    elif style == 2:
                        insertion = f", {parenthetical},"
                    else:
                        insertion = f" — {parenthetical} —"
                    
                    # Insert the parenthetical
                    if insert_pos < len(words):
                        before = " ".join(words[:insert_pos])
                        after = " ".join(words[insert_pos:])
                        sentence = before + insertion + " " + after
            
            # 4. Create a typo but then correct it
            elif modification == 4 and random.random() < 0.15:  # Very rare, only 15% of already modified sentences
                words = sentence.split()
                if len(words) > 5:
                    word_pos = random.randint(1, len(words)-1)
                    # Only consider words longer than 5 chars for "typo"
                    if len(words[word_pos]) > 5:
                        words[word_pos] = words[word_pos][:-1] + "*" + words[word_pos][-1]
                        sentence = " ".join(words)
            
            # 5. Create a single comma splice (joining independent clauses with a comma)
            elif modification == 5 and i < len(sentences) - 1 and random.random() < 0.3:
                next_sentence = sentences[i+1] if i+1 < len(sentences) else ""
                if len(next_sentence) < 40 and len(sentence) < 60:  # Only join shorter sentences
                    if sentence.endswith("."):
                        sentence = sentence[:-1] + "," + " " + next_sentence[0].lower() + next_sentence[1:]
                        # Mark next sentence for removal
                        sentences[i+1] = ""
            
            # 6. Insert an em-dash instead of a period or comma
            elif modification == 6 and len(sentence) > 30:
                # Find a suitable position for the em-dash
                pos = len(sentence) // 2
                if pos < len(sentence) and sentence[pos] == " ":
                    sentence = sentence[:pos] + " —" + sentence[pos:]
                    
            # 7. Use inconsistent spacing around punctuation
            elif modification == 7:
                if ", " in sentence and random.random() < 0.3:
                    sentence = sentence.replace(", ", ",", 1)  # Remove space after a comma once
            
            # 8. Uncommon capitalization pattern
            elif modification == 8 and "i" in sentence.lower() and random.random() < 0.2:
                sentence = sentence.replace(" i ", " I ")  # Ensure 'i' is always capitalized
            
            # 9. Short emphasis
            elif modification == 9 and random.random() < 0.3:
                words = sentence.split()
                if len(words) > 5:
                    word_pos = random.randint(2, min(5, len(words)-1))
                    if len(words[word_pos]) > 3:  # Only emphasize words of reasonable length
                        if random.random() < 0.5:
                            # ALL CAPS for emphasis
                            words[word_pos] = words[word_pos].upper()
                        else:
                            # Italics using asterisks
                            words[word_pos] = "*" + words[word_pos] + "*"
                        sentence = " ".join(words)
            
            # 10. Add a subtle intensifier
            elif modification == 10:
                intensifiers = ["really", "very", "quite", "pretty", "fairly", "somewhat", "rather"]
                words = sentence.split()
                if len(words) > 5:
                    adj_pos = -1
                    # Look for adjectives or verbs to intensify
                    for j, word in enumerate(words):
                        if j > 0 and len(word) > 4:  # Simple heuristic to find adjectives
                            adj_pos = j
                            break
                    
                    if adj_pos > 0:
                        words.insert(adj_pos, random.choice(intensifiers))
                        sentence = " ".join(words)
        
        if sentence:  # Only add non-empty sentences
            modified_sentences.append(sentence)
    
    return " ".join(modified_sentences)


def apply_structural_changes(text):
    """Apply structural changes to make text more human-like."""
    # 1. Make some paragraphs of different lengths
    paragraphs = text.split("\n\n")
    modified_paragraphs = []
    
    for paragraph in paragraphs:
        if len(paragraph) > 200 and random.random() < 0.4:  # 40% chance for long paragraphs
            # Split the paragraph at a sentence boundary
            sentences = re.split(r'(?<=[.!?]) +', paragraph)
            if len(sentences) > 3:
                split_point = random.randint(len(sentences)//3, 2*len(sentences)//3)
                first_part = " ".join(sentences[:split_point])
                second_part = " ".join(sentences[split_point:])
                modified_paragraphs.append(first_part)
                modified_paragraphs.append(second_part)
            else:
                modified_paragraphs.append(paragraph)
        else:
            modified_paragraphs.append(paragraph)
    
    # 2. Adjust spacing (sometimes add extra newlines)
    for i in range(len(modified_paragraphs)-1):
        if random.random() < 0.2:  # 20% chance
            modified_paragraphs[i] += "\n"  # Extra newline
    
    return "\n\n".join(modified_paragraphs)


def post_process_humanized_text(text):
    """Apply post-processing to the humanized text to enhance human-like qualities."""
    # 1. Apply human writing quirks
    text = apply_human_quirks(text)
    
    # 2. Apply structural changes
    text = apply_structural_changes(text)
    
    # 3. Fix common pattern issues (clean up double spaces, etc.)
    text = re.sub(r' +', ' ', text)  # Remove excess spaces
    text = re.sub(r'\n\n\n+', '\n\n', text)  # Remove excess newlines
    
    # 4. Break perfect patterns - replace exactly repeated phrases
    for phrase in re.findall(r'\b(\w+\s+\w+\s+\w+)\b.+\b\1\b', text):
        if len(phrase) > 10:  # Only replace substantial repeated phrases
            words = phrase.split()
            if len(words) >= 3:
                # Create a slightly modified version of the phrase
                modified = words[0] + " " + (words[1] if random.random() < 0.5 else "actually " + words[1]) + " " + words[2]
                # Replace the second occurrence only
                parts = text.split(phrase, 2)
                if len(parts) > 2:
                    text = phrase.join(parts[:2]) + modified + parts[2]
    
    return text


# Import the AI bypasser module
from ai_bypasser import make_human_like, optimize_evasion
from advanced_humanizer import humanize_text_advanced

import re
import random
import logging
import os
import json
import time
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv
import google.generativeai as genai

# Import the genuine human rewriter module
from human_rewriter import rewrite_text_humanly, TextStructureAnalyzer

def is_structure_preserved(original: str, modified: str) -> bool:
    """
    Check if the modified text preserves the structure of the original.
    Uses the TextStructureAnalyzer from the human_rewriter module.
    
    Args:
        original: Original text
        modified: Modified text
        
    Returns:
        True if structure is preserved, False otherwise
    """
    return TextStructureAnalyzer.ensure_structure_preserved(original, modified)

async def humanize_text(text: str) -> str:
    """
    Humanize text using genuine rewriting techniques rather than character manipulation.
    
    The function uses a two-stage approach:
    1. Gemini API for creative rewriting while preserving meaning and structure
    2. Human-style transformer for enhancing natural language patterns
    
    Args:
        text: The text to humanize
        
    Returns:
        Genuinely rewritten human-like text
    """
    logger.info("Starting genuine human rewriting process...")
    
    # If text is too short, return it as is
    if len(text) < 10:
        return text
        
    # Estimate token usage
    estimated_tokens = estimate_tokens(text)
    
    # Check token limits
    if not check_token_limit(estimated_tokens):
        raise TokenLimitExceededError("Daily token limit would be exceeded")
    
    try:
        # APPROACH 1: Use Gemini for creative rewriting with proper guidance
        prompt = format_humanize_prompt(text)
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Generate the genuinely rewritten content
        response = model.generate_content(prompt)
        rewritten_text = response.text.strip()
        
        # Update token usage
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            update_token_usage(response.usage_metadata)
        else:
            # Estimate if metadata not available
            prompt_tokens = estimated_tokens
            completion_tokens = len(rewritten_text) // 4
            update_token_usage(prompt_tokens, completion_tokens)
        
        # Check if structure has been preserved
        if not is_structure_preserved(text, rewritten_text):
            logger.warning("Structure not properly preserved by Gemini, trying human rewriter approach")
            # APPROACH 2: Use our local human rewriter as an alternative
            return rewrite_text_humanly(text)
        
        # If Gemini preserved structure, further enhance the text with our human rewriter
        logger.info("Applying additional human writing characteristics...")
        final_text = rewrite_text_humanly(rewritten_text)
        
        # Final structure check
        if not is_structure_preserved(text, final_text):
            logger.warning("Structure changed in final rewrite, using Gemini output only")
            return rewritten_text
        
        return final_text
        
    except Exception as e:
        logger.error(f"Error in humanize_text: {str(e)}")
        # Fallback to direct human rewriter
        try:
            logger.info("Falling back to direct human rewriter...")
            return rewrite_text_humanly(text)
        except Exception as e2:
            logger.error(f"Human rewriter fallback failed: {str(e2)}")
            # Ultimate fallback to rule-based approach
            logger.info("Using rule-based humanization as last resort")
            return rule_based_humanize(text)

async def humanize_long_text(text: str, max_chunk_size: int = 4000) -> str:
    """
    Humanize longer text by splitting it into manageable chunks.
    
    Args:
        text: The long text to humanize
        max_chunk_size: Maximum size of each chunk in characters
        
    Returns:
        Fully humanized text
    """
    # If text is shorter than chunk size, process it directly
    if len(text) <= max_chunk_size:
        return await humanize_text(text)
    
    # Split text into paragraphs
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Group paragraphs into chunks
    for paragraph in paragraphs:
        if current_length + len(paragraph) > max_chunk_size and current_chunk:
            # Process current chunk
            chunk_text = '\n\n'.join(current_chunk)
            humanized_chunk = await humanize_text(chunk_text)
            chunks.append(humanized_chunk)
            
            # Reset for next chunk
            current_chunk = [paragraph]
            current_length = len(paragraph)
        else:
            current_chunk.append(paragraph)
            current_length += len(paragraph)
    
    # Process the last chunk if it exists
    if current_chunk:
        chunk_text = '\n\n'.join(current_chunk)
        humanized_chunk = await humanize_text(chunk_text)
        chunks.append(humanized_chunk)
    
    # Combine all humanized chunks
    return '\n\n'.join(chunks)

# Rule-based fallback humanization function
def rule_based_humanize(text: str) -> str:
    """
    Advanced rule-based text humanization when API is unavailable.
    
    This fallback applies sophisticated text transformations designed
    specifically to make AI-generated text pass AI detectors while
    maintaining the original meaning.
    """
    import random
    import re
    
    # Clean the input text and split into sentences
    result = text.strip()
    sentences = re.split(r'(?<=[.!?]) +', result)
    
    # Formal phrase replacements (AI patterns → human patterns)
    formal_replacements = [
        ("Additionally,", ["Also,", "Plus,", "What's more,"]),
        ("Furthermore,", ["Beyond that,", "Also,"]),
        ("Moreover,", ["And,", "On top of that,"]),
        ("In conclusion,", ["So,", "To wrap up,", "In the end,"]),
        ("Consequently,", ["So,", "As a result,"]),
        ("Subsequently,", ["Later,", "After that,", "Then,"]),
        ("In summary,", ["To sum up,", "Basically,"]),
        ("due to the fact that", ["because", "since"]),
        ("a significant number of", ["many", "lots of", "tons of"]),
        ("a substantial amount of", ["lots of", "a bunch of"]),
        ("it is important to note that", ["", "note that", "remember"]),
        ("it should be noted that", ["", "just know that"]),
        ("it is worth mentioning that", ["", "by the way"]),
        ("in order to", ["to"]),
        ("for the purpose of", ["to", "for"]),
        ("in the event that", ["if"]),
        ("in spite of the fact that", ["although", "even though"]),
        ("with regard to", ["about", "regarding"]),
        ("with reference to", ["about"]),
        ("in relation to", ["about", "concerning"]),
        ("utilize", ["use"]),
        ("utilization", ["use"]),
        ("implementation", ["use", "setup"]),
        ("demonstrate", ["show", "prove"]),
        ("facilitate", ["help", "make easier"]),
        ("necessitate", ["need", "require"]),
        ("sufficient", ["enough"]),
        ("numerous", ["many", "lots of"]),
        ("possess", ["have", "own"]),
        ("regarding", ["about", "on"]),
        ("concerning", ["about", "regarding"]),
        ("is able to", ["can"]),
        ("prior to", ["before"]),
        ("subsequent to", ["after"]),
        ("is required to", ["must", "has to"]),
    ]
    
    # Apply randomized replacements to avoid patterns
    for i, sentence in enumerate(sentences):
        # Apply formal phrase replacements with randomization
        for old, new_options in formal_replacements:
            if old in sentence:
                # For each replacement opportunity, randomly choose from options
                new = random.choice(new_options)
                sentences[i] = sentence.replace(old, new)
    
    # Randomly apply human writing characteristics to avoid AI detection patterns
    humanized_sentences = []
    for i, sentence in enumerate(sentences):
        current = sentence
        
        # Only apply transformations to some sentences, not all (random)
        if random.random() < 0.4:  # 40% chance
            transformation = random.randint(1, 10)
            
            if transformation == 1 and len(current) > 40:
                # Split a long sentence with a dash
                words = current.split()
                split_point = len(words) // 2
                current = " ".join(words[:split_point]) + " — " + " ".join(words[split_point:])
                
            elif transformation == 2 and not current.startswith("And") and not current.startswith("But") and not current.startswith("So") and len(current) > 10:
                # Start with a conjunction (only if it makes sense)
                conjunctions = ["And", "But", "So"]
                current = random.choice(conjunctions) + " " + current[0].lower() + current[1:]
                
            elif transformation == 3 and "not" in current:
                # Convert "is not" to "isn't" etc. (contractions)
                current = current.replace("is not", "isn't")
                current = current.replace("are not", "aren't")
                current = current.replace("do not", "don't")
                current = current.replace("does not", "doesn't")
                current = current.replace("will not", "won't")
                current = current.replace("cannot", "can't")
                
            elif transformation == 4:
                # Add a mild personal reaction
                reactions = [
                    " I think this matters.", 
                    " That's pretty interesting.", 
                    " This really stands out to me."
                ]
                if current.endswith("."):
                    current = current[:-1] + random.choice(reactions)
                    
            elif transformation == 5 and random.random() < 0.3:
                # Add subtle intensifier
                intensifiers = ["actually", "really", "truly", "definitely"]
                words = current.split()
                if len(words) > 4:
                    insert_pos = random.randint(1, min(3, len(words)-1))
                    words.insert(insert_pos, random.choice(intensifiers))
                    current = " ".join(words)
                    
            elif transformation == 6 and len(current) > 30:
                # Add a parenthetical remark
                parentheticals = [
                    " (at least in my experience)",
                    " (this is pretty important)",
                    " (which makes sense)"
                ]
                pos = len(current) // 2
                if current[pos] == " ":
                    current = current[:pos] + random.choice(parentheticals) + current[pos:]
            
            elif transformation == 7:
                # Convert to more informal phrasing
                current = current.replace("therefore", "so")
                current = current.replace("however", "but")
                current = current.replace("nevertheless", "still")
                current = current.replace("thus", "so")
                
            elif transformation == 8 and len(current) < 30 and random.random() < 0.3:
                # Extremely short sentence for emphasis
                current += " Simple as that."
                
            elif transformation == 9 and i > 0 and i < len(sentences) - 1:
                # Add a thought interruption
                interruptions = ["Wait.", "Hold on.", "Actually, no.", "Hmm."]
                if random.random() < 0.15:  # Keep this rare
                    humanized_sentences.append(random.choice(interruptions))
                    
            elif transformation == 10 and random.random() < 0.2:
                # Intentional comma splice (common human error)
                if i < len(sentences) - 1 and len(sentences[i+1]) < 30:
                    current = current + ", " + sentences[i+1]
                    # Skip the next sentence since we used it
                    sentences[i+1] = ""
        
        if current and not current.isspace():
            humanized_sentences.append(current)
    
    # Filter out any empty sentences and join
    humanized_sentences = [s for s in humanized_sentences if s]
    result = " ".join(humanized_sentences)
    
    # Final pass - add a few more human touches
    result = result.replace(" AI ", " A.I. ")
    
    # Final cleanup - fix any double spaces or punctuation issues
    result = re.sub(r' +', ' ', result)
    result = re.sub(r'\?\?', '?', result)
    result = re.sub(r'!!', '!', result)
    result = re.sub(r'\.\. ', '. ', result)
    
    return result
