# EnHumanizer - AI Text Humanization Tool

EnHumanizer is a sophisticated text processing system that transforms AI-generated English text to be indistinguishable from human writing. This project includes a Telegram bot interface and a robust text humanization engine that works with multiple API providers (Gemini, DeepSeek, Hugging Face).

## Key Features

- Transforms AI-generated text to be indistinguishable from human writing
- Advanced text type detection (academic, business, resume, educational, narrative)
- Intelligent style analysis and formality level detection
- Preserves original structure, meaning and context with high fidelity
- Sophisticated regex-based post-processing for natural language patterns
- Handles spacing, punctuation, and grammatical nuances automatically
- No fabricated information added during transformation
- Multiple API support (Gemini, DeepSeek, Hugging Face)
- Handles long texts by splitting into manageable chunks
- Robust fallback rule-based humanization if API issues occur
- Optimized token usage to stay within free tier limits
- Telegram bot interface for easy access
- User statistics tracking

## Requirements

- Python 3.8+
- Telegram Bot Token (for bot interface)
- One of the following API keys:
  - Google AI Gemini API Key (preferred)
  - DeepSeek API Key
  - Hugging Face API Token

## Installation

1. Clone this repository
   ```
   git clone https://github.com/YOUR-USERNAME/enhumanizer.git
   cd enhumanizer
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up NLTK data (required for text analysis):
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
   ```

4. Create a `.env` file with your API keys:
   ```
   # Required for Telegram bot
   TELEGRAM_TOKEN=your_telegram_token
   
   # Choose one of the following API options
   GEMINI_API_KEY=your_gemini_api_key
   # -- OR --
   DEEPSEEK_API_KEY=your_deepseek_api_key
   # -- OR --
   HUGGINGFACE_API_KEY=your_huggingface_token
   ```

## Usage

### Telegram Bot

1. Start the bot:
   ```
   python bot.py
   ```
2. Open Telegram and search for your bot
3. Send any text you want to humanize
4. Receive the humanized version

### Direct Python Usage

```python
from human_rewriter import rewrite_text_humanly

original_text = "Your AI-generated text here"
humanized_text = rewrite_text_humanly(original_text)
print(humanized_text)
```

## Bot Commands

- `/start` - Initialize the bot and get welcome message
- `/help` - Show help information
- `/stats` - View your usage statistics

## API Limits

- **Gemini API**: Free tier includes 60 queries per minute, 1M tokens per month (Gemini 2.5 Flash Preview has up to 1 billion tokens on the free tier)
- **DeepSeek API**: Varies based on subscription level
- **Hugging Face**: Free tier includes 30,000 inference requests per month

## Technical Details

### Text Type Detection

EnHumanizer intelligently detects five major text types to apply appropriate style transformations:

- **Academic/Scientific**: Papers, research, technical documentation
- **Business/Marketing**: Promotional content, reports, professional communications
- **Resume/Professional**: CVs, cover letters, personal statements
- **Educational/Explanatory**: Tutorials, guides, how-to content
- **Narrative/Storytelling**: Fiction, personal narratives, blog posts

### Core Components

- **human_rewriter.py**: Main text humanization engine with advanced regex patterns
- **gemini_api.py**: API integration for Gemini model
- **ai_bypasser.py**: AI detection evasion techniques
- **bot.py**: Telegram bot interface

## Disclaimer

This tool is designed for legitimate use cases such as improving the quality and readability of AI-generated content. Users are responsible for ensuring they comply with terms of service for all APIs and platforms used with this tool. The developers of EnHumanizer do not endorse using this tool for deceptive or unethical purposes.

## License

MIT License - See LICENSE file for details
## Notes

This bot focuses on creating human-sounding text that passes AI detection tools while maintaining complete fidelity to the original meaning.
