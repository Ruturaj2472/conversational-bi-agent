import os

# app.config builds Settings at import time and requires an API key
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
