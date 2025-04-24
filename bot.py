"""
Telegram bot for humanizing text using Gemini API.
"""
import os
import asyncio
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from telegram import Update, constants
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
    CallbackContext
)

from gemini_api import humanize_long_text, TokenLimitExceededError, rule_based_humanize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Telegram token
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("Missing TELEGRAM_TOKEN in environment variables")

# User session data
user_sessions: Dict[int, Dict[str, Any]] = {}

# Constants
MAX_MESSAGE_LENGTH = 4096  # Telegram message length limit

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    welcome_message = (
        f"👋 *Welcome, {user.first_name}!* 👋\n\n"
        "✨ I'm your *Text Humanizer Bot* (BETA) ✨\n\n"
        "I transform AI-generated text to sound 100% human-written while preserving the original meaning.\n\n"
        "🔹 *How to use:*\n"
        "Simply send me any text, and I'll make it sound naturally human.\n\n"
        "🔹 *Commands:*\n"
        "/start - Show this welcome message\n"
        "/help - Tips for best results\n"
        "/stats - View your usage statistics\n\n"
        "⚠️ *Beta Notice:* This is a beta version using Gemini API. Your feedback helps improve the service!"
    )
    await update.message.reply_text(welcome_message, parse_mode=constants.ParseMode.MARKDOWN)
    
    # Initialize user session if needed
    if user.id not in user_sessions:
        user_sessions[user.id] = {
            "processed_texts": 0,
            "total_characters": 0,
        }

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    help_text = (
        "📚 *Text Humanizer Bot - Guide* 📚\n\n"
        "🔸 *Best Practices:*\n"
        "• Send complete paragraphs for more coherent results\n"
        "• Include context when possible\n"
        "• For formal documents, mention this in your message\n"
        "• For longer texts (>4000 chars), the bot will process them in parts\n\n"
        "🔸 *What This Bot Does:*\n"
        "• Makes AI text sound authentically human-written\n"
        "• Preserves the original meaning completely\n"
        "• Adjusts tone while maintaining the text's structure\n"
        "• Removes telltale AI patterns and phrasing\n"
        "• Optimizes readability and engagement\n\n"
        "🔸 *Commands:*\n"
        "/start - Display welcome screen\n"
        "/help - Show this guide\n"
        "/stats - View your usage statistics\n\n"
        "✨ *Pro Tip:* The bot works best with English text and adapts to both formal and informal writing styles.\n\n"
        "⚠️ This is a BETA version. Occasional hiccups might happen!"
    )
    await update.message.reply_text(help_text, parse_mode=constants.ParseMode.MARKDOWN)

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show user statistics."""
    user_id = update.effective_user.id
    
    if user_id not in user_sessions:
        await update.message.reply_text("📊 *No Usage Data*\n\nYou haven't used the bot yet. Send some text to get started!", parse_mode=constants.ParseMode.MARKDOWN)
        return
    
    stats = user_sessions[user_id]
    estimated_tokens = stats['total_characters'] // 4
    
    # Calculate a simple efficiency rating
    if stats['processed_texts'] > 0:
        avg_chars = stats['total_characters'] / stats['processed_texts']
        if avg_chars > 1000:
            efficiency = "Excellent! You're efficiently using the API with longer texts."
        else:
            efficiency = "Good. Consider sending longer texts for better efficiency."
    else:
        efficiency = "No data yet."
    
    stats_text = (
        f"📊 *Your Usage Statistics* 📊\n\n"
        f"🔹 *Activity*\n"
        f"• Texts processed: {stats['processed_texts']}\n"
        f"• Total characters: {stats['total_characters']:,}\n"
        f"• Estimated tokens: ~{estimated_tokens:,}\n\n"
        f"🔹 *Efficiency Rating*\n"
        f"• {efficiency}\n\n"
        f"🔹 *API Usage*\n"
        f"• Daily limit: 1 billion tokens\n"
        f"• Your usage: {estimated_tokens:,} tokens ({(estimated_tokens/1_000_000_000)*100:.6f}% of daily limit)\n\n"
        f"\u2728 *BETA TEST* \u2728\n"
    )
    
    await update.message.reply_text(stats_text, parse_mode=constants.ParseMode.MARKDOWN)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process text messages from users."""
    user_id = update.effective_user.id
    text = update.message.text
    
    # Initialize user session if needed
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "processed_texts": 0,
            "total_characters": 0,
        }
    
    # Update user session with incoming text
    user_sessions[user_id]["total_characters"] += len(text)
    
    # Send 'processing' message
    processing_message = await update.message.reply_text(
        "🧠 *Processing your text...*\n\nThis may take a moment depending on length.",
        parse_mode=constants.ParseMode.MARKDOWN
    )
    
    try:
        # Call the Gemini API to humanize the text
        humanized_text = await humanize_long_text(text)
        
        # Update processed texts count
        user_sessions[user_id]["processed_texts"] += 1
        
        # Send confirmation message that humanized text follows
        await update.message.reply_text(
            "✅ *Humanization complete!*\n\nHere is your human-written text:",
            parse_mode=constants.ParseMode.MARKDOWN
        )
        
        # Split message if it exceeds Telegram's length limit
        if len(humanized_text) <= MAX_MESSAGE_LENGTH:
            await update.message.reply_text(humanized_text)
        else:
            # Split into chunks of MAX_MESSAGE_LENGTH
            chunks = [humanized_text[i:i + MAX_MESSAGE_LENGTH] 
                     for i in range(0, len(humanized_text), MAX_MESSAGE_LENGTH)]
            
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await update.message.reply_text(chunk)
                else:
                    await update.message.reply_text(f"(Continued {i+1}/{len(chunks)})\n\n{chunk}")
        
    except TokenLimitExceededError:
        await update.message.reply_text(
            "⚠️ *Daily token limit reached*\n\nPlease try again tomorrow or send shorter texts.",
            parse_mode=constants.ParseMode.MARKDOWN
        )
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        
        # Try rule-based humanization as fallback
        try:
            fallback_text = rule_based_humanize(text)
            
            await update.message.reply_text(
                "⚠️ *API issue detected*\n\nUsing basic humanization instead (simplified version):",
                parse_mode=constants.ParseMode.MARKDOWN
            )
            
            await update.message.reply_text(fallback_text)
            
        except Exception as fallback_error:
            logger.error(f"Fallback error: {str(fallback_error)}")
            await update.message.reply_text(
                "\u274c Sorry, I encountered an error processing your text. Please try again later."
            )
    
    finally:
        # Delete the processing message
        await processing_message.delete()

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors."""
    logger.error(f"Update {update} caused error {context.error}")
    
    # Notify user if possible
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "Sorry, something went wrong. Please try again later."
        )

def main() -> None:
    """Start the bot."""
    # Create the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    # Register error handler
    application.add_error_handler(error_handler)

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
