import os

os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("API_KEY_HEADER", "x-api-key")

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()
