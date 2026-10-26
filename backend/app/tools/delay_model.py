"""Delay-risk model: classical ML where deep learning would be overkill.

Trained on simulated trips (training/train_delay_model.py). Replace the
simulator with real trip history once a transporter shares it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import BASE_DIR

FEATURES = [
    "remaining_km",
    "avg_speed_kmph",
    "stop_count",
    "stop_minutes",
    "max_deviation_km",
    "hours_to_deadline",
    "hour_of_day",
]
MODEL_PATH = BASE_DIR / "models" / "delay_model.joblib"


def simulate_trips(n: int = 4000, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    remaining = rng.uniform(5, 900, n)
    speed = np.clip(rng.normal(42, 12, n), 0, 75)
    stops = rng.poisson(2, n)
    stop_min = stops * rng.gamma(2.0, 25, n)
    deviation = rng.exponential(2.5, n)
    hour = rng.integers(0, 24, n)
    hours_left = rng.uniform(0.5, 30, n)
    # Ground truth: realised speed drops with night driving in cities, long stops and detours.
    night_penalty = np.where((hour >= 22) | (hour <= 5), 0.85, 1.0)
    realised_speed = np.maximum(speed * night_penalty * rng.normal(0.85, 0.12, n), 5)
    hours_needed = remaining / realised_speed + stop_min / 600 + deviation / 30
    y = (hours_needed > hours_left).astype(int)
    X = np.column_stack([remaining, speed, stops, stop_min, deviation, hours_left, hour])
    return X, y


def train(path: Path = MODEL_PATH, n: int = 4000, seed: int = 7) -> dict:
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    X, y = simulate_trips(n, seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=seed)
    model.fit(X_tr, y_tr)
    auc = float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
    # Baseline the agent would otherwise use: "late if remaining/speed > hours_left".
    naive = (X_te[:, 0] / np.maximum(X_te[:, 1], 1) > X_te[:, 5]).astype(int)
    naive_acc = float((naive == y_te).mean())
    model_acc = float((model.predict(X_te) == y_te).mean())
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return {
        "auc": round(auc, 4),
        "accuracy": round(model_acc, 4),
        "naive_rule_accuracy": round(naive_acc, 4),
        "n_test": len(y_te),
    }


@lru_cache(maxsize=1)
def _model():
    import joblib

    if not MODEL_PATH.exists():
        train()
    return joblib.load(MODEL_PATH)


def predict_delay_risk(features: dict) -> float:
    row = np.array([[float(features.get(f, 0.0)) for f in FEATURES]])
    return round(float(_model().predict_proba(row)[0, 1]), 3)
