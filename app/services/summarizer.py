import time
import hashlib
from typing import Dict, Tuple
from contextlib import nullcontext
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.core.config import get_settings

_settings = get_settings()

_cache: Dict[str, Tuple[str, float]] = {}
_CACHE_MAX = 500


class _SummarizerHolder:
    model = None
    tokenizer = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load():
    if _SummarizerHolder.model is None:
        _SummarizerHolder.tokenizer = AutoTokenizer.from_pretrained(
            _settings.summarizer_model
        )
        _SummarizerHolder.model = AutoModelForSeq2SeqLM.from_pretrained(
            _settings.summarizer_model
        ).to(_SummarizerHolder.device)
    return _SummarizerHolder.tokenizer, _SummarizerHolder.model


def _truncate_tokens(tokenizer, text: str, max_len: int) -> str:
    tokens = tokenizer.encode(text, truncation=True, max_length=max_len)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def summarize_text(text: str, max_len: int = None, min_len: int = None):
    t0 = time.time()
    tokenizer, model = _load()

    max_new = max_len or _settings.summarizer_max_new_tokens
    min_new = min_len or _settings.summarizer_min_new_tokens

    truncated = _truncate_tokens(tokenizer, text, _settings.summarizer_truncate_tokens)
    key_raw = f"{truncated}|{max_new}|{min_new}|{_settings.summarizer_model}"
    key = hashlib.sha256(key_raw.encode()).hexdigest()
    if key in _cache:
        summary, cached_latency = _cache[key]
        latency_ms = (time.time() - t0) * 1000.0
        return {
            "summary": summary,
            "model_version": _settings.summarizer_version,
            "latency_ms": latency_ms,
            "cached": True,
        }

    inputs = tokenizer([truncated], return_tensors="pt").to(_SummarizerHolder.device)
    use_cuda = _SummarizerHolder.device.type == "cuda"
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.float16) if use_cuda else nullcontext()
    )
    with torch.inference_mode():
        # Autocast speeds up generation on CUDA while saving memory
        with autocast_ctx:
            output_ids = model.generate(
                **inputs,
                num_beams=_settings.summarizer_num_beams,
                max_new_tokens=max_new,
                min_new_tokens=min_new,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
            )
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    latency_ms = (time.time() - t0) * 1000.0
    if len(_cache) >= _CACHE_MAX:
        # simple FIFO eviction
        _cache.pop(next(iter(_cache)))
    _cache[key] = (summary, latency_ms)
    return {
        "summary": summary,
        "model_version": _settings.summarizer_version,
        "latency_ms": latency_ms,
        "cached": False,
    }
