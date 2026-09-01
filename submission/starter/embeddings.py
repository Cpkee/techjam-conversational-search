"""BROWSE: dense retrieval via a swappable embedding model (CLAUDE.md 2.5), plain
cosine similarity via brute-force numpy matmul against pre-computed catalog
embeddings -- no vector DB. At 50,000 products x a few hundred dims this is well
within reason (CLAUDE.md 2.6: reach for a framework only when a specific,
demonstrated need arises).

Model choice: CLAUDE.md 2.5 named blair-roberta-base (pretrained on Amazon Reviews
2023, the same source family as this catalog) as the primary choice, with
bge-small-en-v1.5 as a documented fallback -- "test both against real dev sessions
before locking it in." That test happened during Step 5 remediation: raw
blair-roberta-base's mean-pooled embeddings were severely anisotropic (cosine-to-
centroid mean=0.895, min=0.767 across the full catalog), causing catalog items with
degenerate text (too-short SKU-code titles, or extremely long keyword-stuffed
listings) to surface as spurious top matches for unrelated queries -- e.g. near-
identical top-5 BROWSE results for "basketball for men" and "women's dresses". ZCA
whitening (compute_whitening_transform/apply_whitening below) fixed the AGGREGATE
geometry (cosine-to-centroid mean dropped to 0.006) but did NOT fully eliminate the
practical hub-item pollution -- the same degenerate-text items still appeared in both
queries' top-5 post-whitening. Raw bge-small-en-v1.5 (CLS pooling, no whitening
needed) showed zero overlap on the same test, so it is the default here. blair-
roberta-base (with or without whitening) remains available via model_key for anyone
who wants to re-verify or build on the whitening path.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_CONFIGS = {
    "bge": {"name": "BAAI/bge-small-en-v1.5", "pooling": "cls", "dim": 384},
    "blair": {"name": "hyp1231/blair-roberta-base", "pooling": "mean", "dim": 768},
}
DEFAULT_MODEL_KEY = "bge"  # see module docstring: raw BGE resolved the hubness
# problem that blair-roberta-base (even whitened) did not, on this catalog.

EMBED_BATCH_SIZE = 64
EMBED_MAX_LENGTH = 64

_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
_tokenizer_cache: dict[str, object] = {}
_model_cache: dict[str, object] = {}


def _load_model(model_key: str):
    if model_key not in _model_cache:
        config = MODEL_CONFIGS[model_key]
        tokenizer = AutoTokenizer.from_pretrained(config["name"])
        model = AutoModel.from_pretrained(config["name"])
        model.to(_DEVICE)
        model.eval()
        _tokenizer_cache[model_key] = tokenizer
        _model_cache[model_key] = model
    return _tokenizer_cache[model_key], _model_cache[model_key]


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        return last_hidden_state[:, 0]
    return _mean_pool(last_hidden_state, attention_mask)


def embed_texts(texts: list[str], model_key: str = DEFAULT_MODEL_KEY, batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    """Returns an (N, dim) float32 matrix of L2-normalized embeddings (so cosine
    similarity reduces to a plain dot product / matmul). dim depends on model_key."""
    tokenizer, model = _load_model(model_key)
    pooling = MODEL_CONFIGS[model_key]["pooling"]
    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=EMBED_MAX_LENGTH
            ).to(_DEVICE)
            output = model(**inputs)
            pooled = _pool(output.last_hidden_state, inputs["attention_mask"], pooling)
            normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(normed.cpu().numpy())
    return np.concatenate(vectors, axis=0).astype(np.float32)


def embed_query(text: str, model_key: str = DEFAULT_MODEL_KEY) -> np.ndarray:
    return embed_texts([text], model_key=model_key)[0]


def _embedding_text(product: dict) -> str:
    title = str(product.get("title") or "")
    features = product.get("features")
    if isinstance(features, list):
        feature_text = " ".join(str(item) for item in features)
    else:
        feature_text = str(features or "")
    return (title + " " + feature_text).strip()[:512]


def build_or_load_catalog_embeddings(
    catalog_path: str | Path, model_key: str = DEFAULT_MODEL_KEY, cache_dir: str | Path | None = None
) -> tuple[np.ndarray, list[str]]:
    """Pre-computed catalog embeddings, cached to disk next to the catalog so the
    (expensive, one-time) embedding pass over all 50,000 products only runs once."""
    catalog_path = Path(catalog_path)
    cache_dir = Path(cache_dir) if cache_dir else catalog_path.parent
    vectors_path = cache_dir / f"{catalog_path.stem}.{model_key}_embeddings.npy"
    ids_path = cache_dir / f"{catalog_path.stem}.{model_key}_embeddings_ids.json"

    if vectors_path.exists() and ids_path.exists():
        vectors = np.load(vectors_path)
        parent_asins = json.loads(ids_path.read_text(encoding="utf-8"))
        return vectors, parent_asins

    parent_asins: list[str] = []
    texts: list[str] = []
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            parent_asins.append(str(product["parent_asin"]))
            texts.append(_embedding_text(product))

    vectors = embed_texts(texts, model_key=model_key)
    np.save(vectors_path, vectors)
    ids_path.write_text(json.dumps(parent_asins), encoding="utf-8")
    return vectors, parent_asins


def compute_whitening_transform(vectors: np.ndarray, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """ZCA whitening: decorrelate + rescale each dimension to unit variance while
    keeping the whitened space aligned with the original axes (unlike PCA-whitening,
    which also reorders/rotates by variance). Returns (mu, W) such that
    whiten(x) = (x - mu) @ W. Fixes AGGREGATE anisotropy but (per this catalog's
    testing, see module docstring) does not fully eliminate practical hub-item
    pollution for blair-roberta-base -- kept available for anyone re-verifying or
    extending that path, not used by the default (bge) model."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        mu = vectors.mean(axis=0)
        centered = vectors - mu
        cov = (centered.T @ centered) / len(vectors)
        eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
        eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
        W = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals + eps)) @ eigvecs.T).astype(np.float32)
    return mu.astype(np.float32), W


def apply_whitening(vectors: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Applies a cached whitening transform and re-normalizes to unit norm (whitening
    changes vector magnitudes, so cosine similarity needs unit-norm vectors again)."""
    single = vectors.ndim == 1
    if single:
        vectors = vectors[np.newaxis, :]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        whitened = (vectors - mu) @ W
        norms = np.linalg.norm(whitened, axis=1, keepdims=True)
        whitened = whitened / np.clip(norms, a_min=1e-9, a_max=None)
    whitened = whitened.astype(np.float32)
    return whitened[0] if single else whitened


def build_or_load_whitening(
    catalog_path: str | Path,
    raw_vectors: np.ndarray,
    model_key: str = DEFAULT_MODEL_KEY,
    cache_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Whitening transform fit on the catalog's own raw embedding distribution,
    cached alongside the embeddings file so the O(N*D^2) covariance/eigendecomposition
    only runs once."""
    catalog_path = Path(catalog_path)
    cache_dir = Path(cache_dir) if cache_dir else catalog_path.parent
    transform_path = cache_dir / f"{catalog_path.stem}.{model_key}_whitening.npz"

    if transform_path.exists():
        data = np.load(transform_path)
        return data["mu"], data["W"]

    mu, W = compute_whitening_transform(raw_vectors)
    np.savez(transform_path, mu=mu, W=W)
    return mu, W


def top_k_by_cosine(
    query_vector: np.ndarray, catalog_vectors: np.ndarray, catalog_ids: list[str], top_k: int
) -> list[tuple[str, float]]:
    """Brute-force cosine similarity (both sides pre-normalized -> plain dot product).

    macOS numpy's Accelerate BLAS backend emits spurious divide/overflow/invalid
    RuntimeWarnings on this matmul shape even when the result is fully finite
    (verified: no NaN/Inf in the output, cross-checked against np.einsum) --
    suppressed here rather than left to alarm on every call."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = catalog_vectors @ query_vector
    if top_k >= len(scores):
        top_indices = np.argsort(-scores)
    else:
        partial = np.argpartition(-scores, top_k)[:top_k]
        top_indices = partial[np.argsort(-scores[partial])]
    return [(catalog_ids[i], float(scores[i])) for i in top_indices[:top_k]]
