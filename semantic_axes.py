# cortex12/semantic_axes.py
"""
Semantic Axis Certification for CORTEX-12
Enables interpretable, verifiable perception by mapping fixed embedding subspaces
to human-understandable attributes (e.g., color, shape).
"""

import numpy as np
import json
import os
from typing import Dict, Any, Union, Optional

# Fixed embedding layout — must be respected during training
EMBEDDING_LAYOUT = {
    "hue": slice(0, 16),
    "saturation": slice(16, 32),
    "brightness": slice(32, 48),
    "shape_circularity": slice(48, 64),
    "size": slice(64, 80),
    "texture_roughness": slice(80, 96),
    # dims 96–127: generic/contextual (not certified by default)
}

# Global registry of decoders (loaded from JSON)
_SEMANTIC_DECODERS: Dict[str, Dict[str, Any]] = {}


def load_certificates(cert_dir: str = "certificates") -> None:
    """Load precomputed semantic decoders from disk."""
    global _SEMANTIC_DECODERS
    _SEMANTIC_DECODERS = {}
    if not os.path.exists(cert_dir):
        return
    for axis in EMBEDDING_LAYOUT:
        path = os.path.join(cert_dir, f"{axis}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                _SEMANTIC_DECODERS[axis] = json.load(f)


def certify_axis(
    embeddings: np.ndarray,
    labels: np.ndarray,
    axis_name: str,
    method: str = "cluster",
    cert_dir: str = "certificates"
) -> float:
    """
    Certify a semantic axis using validation data.
    Saves a lightweight decoder to {cert_dir}/{axis_name}.json.
    """
    if axis_name not in EMBEDDING_LAYOUT:
        raise ValueError(f"Unknown axis: {axis_name}")
    
    subspace = embeddings[:, EMBEDDING_LAYOUT[axis_name]]
    os.makedirs(cert_dir, exist_ok=True)
    cert_path = os.path.join(cert_dir, f"{axis_name}.json")

    if method == "cluster":
        unique_labels = np.unique(labels)
        centroids = {}
        for label in unique_labels:
            mask = labels == label
            centroid = np.mean(subspace[mask], axis=0).tolist()
            centroids[label] = centroid
        
        # Evaluate accuracy
        pred_labels = []
        for vec in subspace:
            dists = [np.linalg.norm(vec - np.array(c)) for c in centroids.values()]
            pred_label = list(centroids.keys())[np.argmin(dists)]
            pred_labels.append(pred_label)
        score = float(np.mean(np.array(pred_labels) == labels))
        
        _SEMANTIC_DECODERS[axis_name] = {
            "type": "cluster",
            "centroids": centroids,
            "metric": "accuracy",
            "score": score
        }

    elif method == "linear":
        X, y = subspace.astype(np.float64), labels.astype(np.float64)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        try:
            theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            intercept, coef = theta[0], theta[1:]
            preds = X_b.dot(theta)
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            score = max(0.0, 1.0 - (ss_res / ss_tot))
        except np.linalg.LinAlgError:
            score, intercept, coef = 0.0, 0.0, np.zeros(X.shape[1])
        
        _SEMANTIC_DECODERS[axis_name] = {
            "type": "linear",
            "coef": coef.tolist(),
            "intercept": float(intercept),
            "metric": "r2",
            "score": float(score)
        }

    else:
        raise ValueError("method must be 'cluster' or 'linear'")

    with open(cert_path, "w") as f:
        json.dump(_SEMANTIC_DECODERS[axis_name], f, indent=2)
    return float(score)


def probe(embedding: np.ndarray, axis_name: str) -> Union[str, float, None]:
    """Interpret an embedding along a certified semantic axis."""
    if axis_name not in _SEMANTIC_DECODERS:
        return None
    decoder = _SEMANTIC_DECODERS[axis_name]
    subspace = embedding[EMBEDDING_LAYOUT[axis_name]]
    
    if decoder["type"] == "cluster":
        centroids = decoder["centroids"]
        dists = [np.linalg.norm(subspace - np.array(c)) for c in centroids.values()]
        return list(centroids.keys())[np.argmin(dists)]
    
    elif decoder["type"] == "linear":
        coef = np.array(decoder["coef"])
        intercept = decoder["intercept"]
        return float(np.dot(subspace, coef) + intercept)
    return None


def validate_embedding(embedding: np.ndarray, expected: Dict[str, Any]) -> Dict[str, bool]:
    """Validate embedding against expected semantic properties."""
    results = {}
    for axis, spec in expected.items():
        value = probe(embedding, axis)
        if value is None:
            results[axis] = False
            continue
        if isinstance(spec, str) and spec.startswith(">"):
            results[axis] = value > float(spec[1:])
        elif isinstance(spec, str) and spec.startswith("<"):
            results[axis] = value < float(spec[1:])
        else:
            results[axis] = value == spec
    return results


# Auto-load certificates on import
load_certificates()
