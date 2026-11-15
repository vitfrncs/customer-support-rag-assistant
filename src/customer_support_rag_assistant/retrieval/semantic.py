"""Funções de similaridade e ordenação utilizadas na recuperação semântica."""

import numpy as np


def compute_similarities(query_embedding: np.ndarray,embeddings: np.ndarray,) -> np.ndarray:
    return embeddings @ query_embedding


def get_top_k_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    return np.argsort(-scores, kind="stable")[:top_k]