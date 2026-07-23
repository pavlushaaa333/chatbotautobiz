from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from threading import Lock
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
FALLBACK_EMBEDDING_BACKEND = "hashing_fallback"
_MODEL_CACHE: dict[str, object] = {}
_MODEL_ERRORS: dict[str, str] = {}
_WARNED_MODEL_ERRORS: set[str] = set()
_MODEL_LOCK = Lock()


@dataclass(slots=True)
class EmbeddingStatus:
    backend: str
    model_name: str
    dimension: int
    error: str | None = None


class AutoBizEmbeddingModel:
    """Local embedding wrapper.

    The primary path uses sentence-transformers. The hashing fallback keeps the
    app/test suite usable when the heavy ML dependencies or model cache are not
    available yet.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        prefer_sentence_transformers: bool = True,
    ):
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self.prefer_sentence_transformers = prefer_sentence_transformers
        self._model = _MODEL_CACHE.get(self.model_name)
        self._model_load_error: str | None = None
        self._backend = (
            "sentence_transformers"
            if self._model is not None
            else FALLBACK_EMBEDDING_BACKEND
        )
        self._fallback = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_features=384,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )

    @property
    def backend(self) -> str:
        return self._backend

    def status(self) -> EmbeddingStatus:
        return EmbeddingStatus(
            backend=self._backend,
            model_name=self.model_name,
            dimension=self.dimension,
            error=self._model_load_error or _MODEL_ERRORS.get(self.model_name),
        )

    @property
    def dimension(self) -> int:
        if self._backend == "sentence_transformers" and self._model is not None:
            try:
                if hasattr(self._model, "get_embedding_dimension"):
                    return int(self._model.get_embedding_dimension())
                return int(self._model.get_sentence_embedding_dimension())
            except Exception:
                return 384
        return 384

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        items = [str(text or "") for text in texts]
        if not items:
            return []

        model = self._load_sentence_transformer()
        if model is not None:
            try:
                embeddings = model.encode(
                    items,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                self._backend = "sentence_transformers"
                return np.asarray(embeddings, dtype=np.float32).tolist()
            except Exception as exc:  # pragma: no cover - depends on local ML runtime.
                self._record_model_error(exc)

        self._backend = FALLBACK_EMBEDDING_BACKEND
        matrix = self._fallback.transform(items)
        matrix = normalize(matrix, norm="l2", copy=False)
        return matrix.astype(np.float32).toarray().tolist()

    def _load_sentence_transformer(self):
        if not self.prefer_sentence_transformers:
            self._backend = FALLBACK_EMBEDDING_BACKEND
            return None

        cached_model = _MODEL_CACHE.get(self.model_name)
        if cached_model is not None:
            self._model = cached_model
            self._backend = "sentence_transformers"
            return cached_model

        if self._model is not None:
            return self._model

        cached_error = _MODEL_ERRORS.get(self.model_name)
        if cached_error:
            self._model_load_error = cached_error
            self._backend = FALLBACK_EMBEDDING_BACKEND
            return None

        if self._model_load_error:
            return None

        with _MODEL_LOCK:
            cached_model = _MODEL_CACHE.get(self.model_name)
            if cached_model is not None:
                self._model = cached_model
                self._backend = "sentence_transformers"
                return cached_model

            cached_error = _MODEL_ERRORS.get(self.model_name)
            if cached_error:
                self._model_load_error = cached_error
                self._backend = FALLBACK_EMBEDDING_BACKEND
                return None

            try:
                from sentence_transformers import SentenceTransformer

                self._model = self._load_sentence_transformer_instance(
                    SentenceTransformer
                )
                _MODEL_CACHE[self.model_name] = self._model
                self._backend = "sentence_transformers"
                return self._model
            except (
                Exception
            ) as exc:  # pragma: no cover - depends on local ML cache/network.
                self._record_model_error(exc)
                return None

    def _load_sentence_transformer_instance(self, sentence_transformer_cls):
        if (
            _env_flag("AUTOBIZ_EMBEDDING_LOCAL_FILES_ONLY")
            or _env_flag("TRANSFORMERS_OFFLINE")
            or _env_flag("HF_HUB_OFFLINE")
        ):
            return _new_sentence_transformer(
                sentence_transformer_cls,
                self.model_name,
                local_files_only=True,
            )

        try:
            return _new_sentence_transformer(
                sentence_transformer_cls,
                self.model_name,
                local_files_only=True,
            )
        except Exception:
            return _new_sentence_transformer(
                sentence_transformer_cls,
                self.model_name,
                local_files_only=False,
            )

    def _record_model_error(self, exc: Exception) -> None:
        error = str(exc)
        self._model_load_error = error
        _MODEL_ERRORS[self.model_name] = error
        self._backend = FALLBACK_EMBEDDING_BACKEND
        if self.model_name not in _WARNED_MODEL_ERRORS:
            warnings.warn(
                f"SentenceTransformer embedding model '{self.model_name}' failed to load/use; "
                f"falling back to {FALLBACK_EMBEDDING_BACKEND}. Error: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            _WARNED_MODEL_ERRORS.add(self.model_name)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _new_sentence_transformer(
    sentence_transformer_cls, model_name: str, *, local_files_only: bool
):
    try:
        return sentence_transformer_cls(model_name, local_files_only=local_files_only)
    except TypeError as exc:
        if "local_files_only" not in str(exc):
            raise
        return sentence_transformer_cls(model_name)
