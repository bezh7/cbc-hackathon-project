"""Configuration settings for the application."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# DeepSeek API Settings
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-vl"
DEEPSEEK_TEMPERATURE = 0.1
DEEPSEEK_MAX_TOKENS = 2000

# OpenAI API Settings
OPENAI_MODEL = "gpt-4"
OPENAI_TEMPERATURE = 0.3
OPENAI_MAX_TOKENS = 1000

# PDF Processing Settings
MAX_PAGES_TO_PROCESS = 3
IMAGE_ZOOM_FACTOR = 2.0  # Higher = better quality but larger files
TEMP_IMAGE_DIR = "temp_images"

# Chat Settings
MAX_CHAT_HISTORY_CONTEXT = 5  # Number of previous messages to include in context
