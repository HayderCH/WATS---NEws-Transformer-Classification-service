import time
import joblib
from pathlib import Path
from typing import Dict, Any
from .preprocessing import basic_clean
from app.core.config import get_settings
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.cuda.amp import autocast

_settings = get_settings()


class _ClassifierHolder:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.classes = None
        self.backend = _settings.classifier_backend
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.tr_tokenizer = None
        self.tr_model = None
        # Default label names (AG News fallback);
        # will be replaced for transformer
        self.label_names: Dict[int, str] = {
            0: "WORLD",
            1: "SPORTS",
            2: "BUSINESS",
            3: "SCIENCE_TECH",
        }
        # precision flags
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        self.use_bf16 = bool(
            torch.cuda.is_available()
            and callable(is_bf16_supported)
            and is_bf16_supported()
        )
        self.use_fp16 = bool(torch.cuda.is_available() and not self.use_bf16)
        # faster matmul on modern GPUs
        if torch.cuda.is_available():
            try:
                torch.set_float32_matmul_precision("high")
            except (AttributeError, RuntimeError):
                pass

    def load(self):
        if self.backend == "sklearn":
            if self.model is not None:
                return
            model_dir = Path(_settings.model_dir) / "classifier"
            self.vectorizer = joblib.load(model_dir / "tfidf_vectorizer.pkl")
            self.model = joblib.load(model_dir / "logreg.pkl")
            meta = joblib.load(model_dir / "label_meta.pkl")
            self.classes = meta["classes_"]
        elif self.backend == "transformer":
            if self.tr_model is not None:
                return
            tdir = Path(_settings.transformer_model_dir) / "best"
            self.tr_tokenizer = AutoTokenizer.from_pretrained(
                str(tdir),
                local_files_only=True,
            )  # nosec B615
            model = AutoModelForSequenceClassification.from_pretrained(
                str(tdir),
                local_files_only=True,
            )  # nosec B615
            self.tr_model = model.to(self.device)
            self.tr_model.eval()
            # Attempt to load label mapping if present
            # Fallback to id labels 0..N-1 if not found
            config = self.tr_model.config
            if hasattr(config, "id2label") and config.id2label:
                self.label_names = {
                    int(k): str(v)
                    for k, v in config.id2label.items()
                }
            else:
                self.label_names = {
                    i: str(i) for i in range(config.num_labels)
                }

    def predict(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        self.load()
        t0 = time.time()
        clean = basic_clean(text)

        if self.backend == "sklearn":
            vec = self.vectorizer.transform([clean])
            probs = self.model.predict_proba(vec)[0]
            top_idx = probs.argmax()
            categories = [
                {
                    "name": self.label_names.get(
                        self.classes[i], f"UNKNOWN_{self.classes[i]}"
                    ),
                    "prob": float(probs[i]),
                }
                for i in range(len(self.classes))
            ]
        else:
            # transformer backend
            enc = self.tr_tokenizer(
                clean,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                if self.device.type == "cuda":
                    # Use autocast; fall back if dtype arg unsupported
                    try:
                        with autocast(
                            dtype=(
                                torch.bfloat16
                                if self.use_bf16
                                else torch.float16
                            )
                        ):
                            logits = self.tr_model(**enc).logits
                    except TypeError:
                        # Older torch autocast signature
                        with autocast():
                            logits = self.tr_model(**enc).logits
                else:
                    logits = self.tr_model(**enc).logits
                # Softmax in float32 for stability and NumPy compatibility
                probs_t = torch.softmax(logits.float(), dim=-1)[0]
                probs = probs_t.detach().to(torch.float32).cpu().numpy()
            top_idx = int(probs.argmax())
            categories = [
                {
                    "name": self.label_names.get(i, str(i)),
                    "prob": float(probs[i]),
                }
                for i in range(len(probs))
            ]

        categories_sorted = sorted(
            categories,
            key=lambda x: x["prob"],
            reverse=True,
        )[:top_k]
        latency_ms = (time.time() - t0) * 1000.0

        # Calculate confidence metrics
        top_prob = categories_sorted[0]["prob"]
        second_prob = (
            categories_sorted[1]["prob"] if len(categories_sorted) > 1 else 0.0
        )
        confidence_margin = top_prob - second_prob

        # Determine confidence level
        if top_prob >= 0.8:
            confidence_level = "HIGH"
        elif top_prob >= 0.6:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

        # Add suggestion for low confidence
        suggestion = None
        if confidence_level == "LOW":
            suggestion = (
                "Article may span multiple topics or need more context"
            )
        elif confidence_margin < 0.1:
            suggestion = "Close classification - consider human review"

        top_name = (
            self.label_names.get(
                self.classes[top_idx], f"UNKNOWN_{self.classes[top_idx]}"
            )
            if self.backend == "sklearn"
            else self.label_names.get(top_idx, str(top_idx))
        )

        result = {
            "top_category": top_name,
            "categories": categories_sorted,
            "confidence_level": confidence_level,
            "confidence_score": float(top_prob),
            "confidence_margin": float(confidence_margin),
            "model_version": _settings.classifier_version,
            "latency_ms": latency_ms,
        }

        if suggestion:
            result["suggestion"] = suggestion

        return result


_classifier_holder = _ClassifierHolder()


def classify_text(text: str) -> Dict[str, Any]:
    return _classifier_holder.predict(text)


def classify_batch(items: list[dict], top_k: int = 5) -> list[Dict[str, Any]]:
    """
    Batch classification.
    Each item is a mapping with optional 'title' and required 'text' fields.
    """
    _classifier_holder.load()
    merged = []
    for it in items:
        txt = it.get("text", "")
        title = it.get("title")
        merged.append(basic_clean((f"{title}. {txt}" if title else txt)))

    results: list[Dict[str, Any]] = []
    t_start = time.time()
    if _classifier_holder.backend == "sklearn":
        vec = _classifier_holder.vectorizer.transform(merged)
        probs_mat = _classifier_holder.model.predict_proba(vec)
        for row in probs_mat:
            top_idx = int(row.argmax())
            categories = [
                {
                    "name": _classifier_holder.label_names.get(
                        _classifier_holder.classes[i],
                        f"UNKNOWN_{_classifier_holder.classes[i]}",
                    ),
                    "prob": float(row[i]),
                }
                for i in range(len(_classifier_holder.classes))
            ]
            categories_sorted = sorted(
                categories, key=lambda x: x["prob"], reverse=True
            )[:top_k]
            top_prob = categories_sorted[0]["prob"]
            second_prob = (
                categories_sorted[1]["prob"]
                if len(categories_sorted) > 1
                else 0.0
            )
            margin = top_prob - second_prob
            if top_prob >= 0.8:
                level = "HIGH"
            elif top_prob >= 0.6:
                level = "MEDIUM"
            else:
                level = "LOW"
            suggestion = None
            if level == "LOW":
                suggestion = (
                    "Article may span multiple topics or need more context"
                )
            elif margin < 0.1:
                suggestion = "Close classification - consider human review"
            top_name = _classifier_holder.label_names.get(
                _classifier_holder.classes[top_idx],
                f"UNKNOWN_{_classifier_holder.classes[top_idx]}",
            )
            results.append(
                {
                    "top_category": top_name,
                    "categories": categories_sorted,
                    "confidence_level": level,
                    "confidence_score": float(top_prob),
                    "confidence_margin": float(margin),
                    "model_version": _settings.classifier_version,
                    "latency_ms": (time.time() - t_start) * 1000.0,
                    "suggestion": suggestion,
                }
            )
    else:
        # transformer batch
        enc = _classifier_holder.tr_tokenizer(
            merged,
            truncation=True,
            max_length=256,
            return_tensors="pt",
            padding=True,
        ).to(_classifier_holder.device)
        with torch.inference_mode():
            if _classifier_holder.device.type == "cuda":
                try:
                    with autocast(
                        dtype=(
                            torch.bfloat16
                            if _classifier_holder.use_bf16
                            else torch.float16
                        )
                    ):
                        logits = _classifier_holder.tr_model(**enc).logits
                except TypeError:
                    with autocast():
                        logits = _classifier_holder.tr_model(**enc).logits
            else:
                logits = _classifier_holder.tr_model(**enc).logits
            probs_mat = (
                torch.softmax(logits.float(), dim=-1)
                .to(torch.float32)
                .cpu()
                .numpy()
            )
        for row in probs_mat:
            top_idx = int(row.argmax())
            categories = [
                {
                    "name": _classifier_holder.label_names.get(i, str(i)),
                    "prob": float(row[i]),
                }
                for i in range(len(row))
            ]
            categories_sorted = sorted(
                categories, key=lambda x: x["prob"], reverse=True
            )[:top_k]
            top_prob = categories_sorted[0]["prob"]
            second_prob = (
                categories_sorted[1]["prob"]
                if len(categories_sorted) > 1
                else 0.0
            )
            margin = top_prob - second_prob
            if top_prob >= 0.8:
                level = "HIGH"
            elif top_prob >= 0.6:
                level = "MEDIUM"
            else:
                level = "LOW"
            suggestion = None
            if level == "LOW":
                suggestion = (
                    "Article may span multiple topics or need more context"
                )
            elif margin < 0.1:
                suggestion = "Close classification - consider human review"
            top_name = _classifier_holder.label_names.get(
                top_idx,
                str(top_idx),
            )
            results.append(
                {
                    "top_category": top_name,
                    "categories": categories_sorted,
                    "confidence_level": level,
                    "confidence_score": float(top_prob),
                    "confidence_margin": float(margin),
                    "model_version": _settings.classifier_version,
                    "latency_ms": (time.time() - t_start) * 1000.0,
                    "suggestion": suggestion,
                }
            )
    return results
