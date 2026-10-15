#!/usr/bin/env python3
"""
Train the capstone's three models exactly as in the notebook, then export the
final Random Forest into a compact binary that the browser app can run.

Usage:  python3 export_model.py <path to BRFSS csv> <output folder>

Writes: model.bin   the 100 trees as flat arrays
        meta.json   feature names, tree offsets, metrics, calibration table
        parity_X.f32, parity_p.f64   test rows and scikit-learn scores, used to
                                     check the browser predictor against sklearn
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

csv_path, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

# ---- Same data steps as the notebook ---------------------------------------
df = pd.read_csv(csv_path)
raw_rows = len(df)
df = df.drop_duplicates()
X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print(f"rows raw={raw_rows} clean={len(df)} train={len(X_train)} test={len(X_test)}")
print(f"positive rate clean={y.mean():.4f} train={y_train.mean():.4f} test={y_test.mean():.4f}")

# ---- Same three models as the notebook -------------------------------------
models = {
    "Baseline Logistic Regression": LogisticRegression(max_iter=1000),
    "Balanced Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_leaf=50,
        class_weight="balanced", random_state=42, n_jobs=-1),
}
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    row = {
        "name": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    results.append(row)
    print(f"{name:30s} " + " ".join(f"{k}={v:.3f}" for k, v in row.items() if k != "name"))

rf = models["Random Forest"]
proba_rf = rf.predict_proba(X_test)[:, 1]
pred_rf = rf.predict(X_test)
y_true = y_test.to_numpy().astype(int)

tp = int(((pred_rf == 1) & (y_true == 1)).sum())
fp = int(((pred_rf == 1) & (y_true == 0)).sum())
fn = int(((pred_rf == 0) & (y_true == 1)).sum())
tn = int(((pred_rf == 0) & (y_true == 0)).sum())
print("confusion (test):", dict(tp=tp, fp=fp, fn=fn, tn=tn))

# ---- Flatten the forest -----------------------------------------------------
# Layout per node: value (float32), right child (uint16, local index),
# feature (uint8, 255 = leaf), threshold*2 (uint8). The left child of an
# internal node is always the next node, so it needs no storage.
vals, rights, feats, thr2s, starts = [], [], [], [], []
total = 0
for est in rf.estimators_:
    t = est.tree_
    left, right = t.children_left, t.children_right
    internal = left >= 0
    idx = np.arange(t.node_count)
    assert np.all(left[internal] == idx[internal] + 1), "tree is not stored depth-first"
    v = t.value[:, 0, :]
    p1 = (v[:, 1] / v.sum(axis=1)).astype(np.float32)
    th2 = np.where(internal, t.threshold * 2.0, 0.0)
    assert np.allclose(th2, np.round(th2)), "threshold is not a multiple of 0.5"
    assert th2.max() <= 254 and th2.min() >= 0
    assert t.node_count < 65536
    vals.append(p1)
    rights.append(np.where(internal, right, 0).astype(np.uint16))
    feats.append(np.where(internal, t.feature, 255).astype(np.uint8))
    thr2s.append(np.round(th2).astype(np.uint8))
    starts.append(total)
    total += t.node_count

blob = (np.concatenate(vals).astype("<f4").tobytes()
        + np.concatenate(rights).astype("<u2").tobytes()
        + np.concatenate(feats).astype(np.uint8).tobytes()
        + np.concatenate(thr2s).astype(np.uint8).tobytes())
assert len(blob) == 8 * total
with open(os.path.join(out_dir, "model.bin"), "wb") as f:
    f.write(blob)
print(f"trees={len(starts)} nodes={total} bytes={len(blob)}")

# ---- Test-set evidence: how people with similar scores really turned out ----
edges = np.linspace(0, 1, 21)
cnt = np.histogram(proba_rf, bins=edges)[0]
pos = np.histogram(proba_rf[y_true == 1], bins=edges)[0]
MIN_N = 300
bins, lo, n, k = [], 0, 0, 0
for i in range(20):
    n += int(cnt[i]); k += int(pos[i])
    if n >= MIN_N:
        bins.append({"lo": float(edges[lo]), "hi": float(edges[i + 1]), "n": n, "k": k})
        lo, n, k = i + 1, 0, 0
if n > 0:
    if bins:
        bins[-1]["hi"] = 1.0; bins[-1]["n"] += n; bins[-1]["k"] += k
    else:
        bins.append({"lo": 0.0, "hi": 1.0, "n": n, "k": k})
bins[-1]["hi"] = 1.0
fine = [{"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": int(cnt[i]), "k": int(pos[i])}
        for i in range(20)]
print("score bins (lo, hi, n, rate):")
for b in bins:
    print(f"  {b['lo']:.2f}-{b['hi']:.2f}  n={b['n']:6d}  rate={b['k'] / b['n']:.3f}")

# ---- Metadata ---------------------------------------------------------------
imp = sorted(zip(X.columns, rf.feature_importances_), key=lambda a: -a[1])
meta = {
    "features": list(X.columns),
    "nodes": total,
    "starts": starts,
    "threshold": 0.5,
    "metrics": results,
    "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    "importance": [{"feature": f, "value": float(v)} for f, v in imp],
    "calibration": bins,
    "fine": fine,
    "data": {
        "raw_rows": raw_rows, "clean_rows": int(len(df)),
        "duplicates": int(raw_rows - len(df)),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "positive_rate_clean": float(y.mean()), "positive_rate_test": float(y_test.mean()),
    },
    "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in X.columns},
    "sklearn": sklearn.__version__,
}
with open(os.path.join(out_dir, "meta.json"), "w") as f:
    json.dump(meta, f)

X_test.to_numpy(dtype=np.float32).tofile(os.path.join(out_dir, "parity_X.f32"))
proba_rf.astype(np.float64).tofile(os.path.join(out_dir, "parity_p.f64"))

print("top importances:", [(f, round(float(v), 3)) for f, v in imp[:10]])
print("ranges:", {c: meta["ranges"][c] for c in ("BMI", "Age", "Education", "Income", "GenHlth", "MentHlth", "PhysHlth")})
