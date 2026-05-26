import os

LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
AGENT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "25"))
AGENT_DEFAULT_MODE = os.environ.get("AGENT_DEFAULT_MODE", "dynamic")
AGENT_HISTORY_FILE = os.environ.get("AGENT_HISTORY_FILE", ".agent_history.jsonl")
