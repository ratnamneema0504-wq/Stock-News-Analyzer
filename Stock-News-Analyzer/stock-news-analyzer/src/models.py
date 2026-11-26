"""AI models mapped to the course policy (Unit 5 Learning -> CO4, Unit 3/6 -> CO3).

Two model families are implemented with scikit-learn and trained live from the
bundled dataset:

- Supervised learning (CO4, lab experiment "Decision Tree Algorithm"): a decision
  tree classifier that predicts the post-news price direction from news features.
  The trained tree is also unwrapped into human-readable IF-THEN rules, which is
  the knowledge-representation / expert-system angle of CO3.
- Unsupervised learning (CO4, lab experiment "K-means clustering"): K-means over
  TF-IDF text of the news to discover clusters without labels.

Everything is descriptive modelling of past observations, never a forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COLUMN_MAP


class ModelUnavailableError(Exception):
    """Raised when the dataset lacks the columns or volume needed to fit a model."""


# The rare class below this share is dropped so the classifier is not skewed by a
# handful of "No Change" rows; documented to the user in the metrics payload.
_MIN_CLASS_SHARE = 0.02


def _logical_col(name: str) -> str | None:
    col = COLUMN_MAP.get(name)
    return col


def direction_model_available(df: pd.DataFrame) -> bool:
    label = _logical_col("price_direction")
    return bool(label) and label in df.columns and df[label].notna().any()


def clustering_available(df: pd.DataFrame) -> bool:
    text = _logical_col("text")
    return bool(text) and text in df.columns and df[text].notna().any()


def _build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Return (X, y, categorical_columns, numeric_columns) from whatever exists."""
    label = _logical_col("price_direction")
    cats: list[str] = []
    nums: list[str] = []
    data = pd.DataFrame(index=df.index)

    for logical, colname in (("sentiment", "sentiment"), ("event_type", "topic"), ("ticker", "company")):
        col = _logical_col(logical)
        if col and col in df.columns:
            data[colname] = df[col].astype(str).str.strip().replace({"nan": "Unknown"})
            cats.append(colname)

    if "__date" in df.columns and df["__date"].notna().any():
        data["year"] = pd.to_numeric(df["__date"].dt.year, errors="coerce").astype(float)
        nums.append("year")

    vcol = _logical_col("volume")
    if vcol and vcol in df.columns:
        vol = pd.to_numeric(df[vcol], errors="coerce").fillna(0).clip(lower=0)
        data["log_volume"] = np.log1p(vol.to_numpy())
        nums.append("log_volume")

    y = df[label].astype(str).str.strip()
    return data[cats + nums], y, cats, nums


def _drop_rare_classes(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    counts = y.value_counts(normalize=True)
    keep = [c for c in counts.index if counts[c] >= _MIN_CLASS_SHARE]
    dropped = [str(c) for c in counts.index if c not in keep]
    mask = y.isin(keep)
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True), dropped


def _base_feature_name(enc_name: str) -> str:
    """ColumnTransformer prefixes names with the step, e.g. 'cat__sentiment_Positive'."""
    return enc_name.split("__", 1)[1] if "__" in enc_name else enc_name


def _split_onehot_name(feature_name: str) -> tuple[str, str] | None:
    """One-hot names look like 'sentiment_Positive'; field names never contain '_'."""
    name = _base_feature_name(feature_name)
    for field in ("sentiment", "topic", "company"):
        if name.startswith(field + "_"):
            return field, name[len(field) + 1:]
    return None


def _extract_rules(clf, feature_names: list[str], total_rows: int) -> list[dict]:
    """Walk the fitted tree and emit readable IF-THEN rules for informative leaves."""
    tree = clf.tree_
    min_support = max(25, int(0.01 * total_rows))
    rules: list[dict] = []

    def recurse(node: int, conditions: list[str]):
        if tree.children_left[node] == tree.children_right[node]:
            support = int(tree.n_node_samples[node])  # raw count, not class-weighted
            if support >= min_support and conditions:
                probabilities = tree.value[node][0]
                predicted_idx = int(np.argmax(probabilities))
                confidence = float(probabilities[predicted_idx] / max(probabilities.sum(), 1e-9))
                rules.append({
                    "conditions": " AND ".join(conditions),
                    "predicted": _class_name(clf, predicted_idx),
                    "support": support,
                    "confidence": confidence,
                })
            return
        name = _base_feature_name(feature_names[tree.feature[node]])
        threshold = float(tree.threshold[node])
        onehot = _split_onehot_name(name)
        if onehot:
            field, value = onehot
            recurse(tree.children_left[node], conditions + [f"{field} != {value}"])
            recurse(tree.children_right[node], conditions + [f"{field} == {value}"])
        else:
            recurse(tree.children_left[node], conditions + [f"{name} <= {threshold:.2f}"])
            recurse(tree.children_right[node], conditions + [f"{name} > {threshold:.2f}"])

    recurse(0, [])
    rules.sort(key=lambda r: (r["confidence"], r["support"]), reverse=True)
    return rules


def _class_name(clf, idx: int) -> str:
    try:
        return str(clf.classes_[idx])
    except Exception:
        return str(idx)


def train_direction_model(df: pd.DataFrame) -> dict:
    """Fit a decision-tree classifier on post-news direction; return metrics + rules + a live predictor."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                                 confusion_matrix)
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.tree import DecisionTreeClassifier

    if not direction_model_available(df):
        raise ModelUnavailableError("The dataset has no price-direction column to learn from.")

    X, y, cats, nums = _build_feature_frame(df)
    if X.shape[1] == 0:
        raise ModelUnavailableError("No usable news features were found to train the model.")
    X, y, dropped = _drop_rare_classes(X, y)
    if len(y) < 50 or y.nunique() < 2:
        raise ModelUnavailableError("Not enough balanced observations to train the model.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    transformers = []
    if cats:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cats))
    if nums:
        transformers.append(("num", "passthrough", nums))
    pre = ColumnTransformer(transformers)

    clf = DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=25, class_weight="balanced", random_state=42
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    feature_names = list(pipe.named_steps["pre"].get_feature_names_out())
    y_pred = pipe.predict(X_test)
    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)

    importances = clf.feature_importances_
    per_field: dict[str, float] = {}
    top_rows = []
    for raw_name, imp in zip(feature_names, importances):
        base = _base_feature_name(raw_name)
        top_rows.append({"feature": base, "importance": float(imp)})
        onehot = _split_onehot_name(base)
        key = onehot[0] if onehot else base
        per_field[key] = per_field.get(key, 0.0) + float(imp)

    field_imp = (
        pd.DataFrame({"field": list(per_field), "importance": list(per_field.values())})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    top_feat = (
        pd.DataFrame(top_rows).sort_values("importance", ascending=False).head(12).reset_index(drop=True)
    )

    rules = _extract_rules(clf, feature_names, len(X))
    majority = float(y.value_counts(normalize=True).max())

    return {
        "pipeline": pipe,
        "metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "baseline_majority": majority,
            "report": report,
        },
        "labels": [str(l) for l in labels],
        "confusion": cm,
        "field_importance": field_imp,
        "top_features": top_feat,
        "rules": rules,
        "categories": {c: sorted(X[c].astype(str).unique().tolist()) for c in cats},
        "numeric_columns": nums,
        "feature_frame_columns": list(X.columns),
        "rows_used": int(len(X)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "dropped_classes": dropped,
        "n_features": int(X.shape[1]),
    }


def predict_direction(bundle: dict, values: dict) -> dict:
    """Predict one row from a what-if form. values maps feature column -> raw input."""
    row = {}
    for col in bundle["feature_frame_columns"]:
        if col in values:
            row[col] = values[col]
        elif col in bundle["categories"]:
            opts = bundle["categories"][col]
            row[col] = opts[0]
        else:  # numeric column without an explicit value
            row[col] = float(values.get(col, 0.0) or 0.0)
    frame = pd.DataFrame([row], columns=bundle["feature_frame_columns"])
    for col in bundle["categories"]:
        frame[col] = frame[col].astype(str)
    for col in bundle["numeric_columns"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).astype(float)
    pipe = bundle["pipeline"]
    proba = pipe.predict_proba(frame)[0]
    classes = [str(c) for c in pipe.classes_]
    scores = sorted(zip(classes, proba), key=lambda t: t[1], reverse=True)
    return {"predicted": scores[0][0], "probs": {c: float(p) for c, p in scores}}


def _text_matrix(df: pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    text = _logical_col("text")
    if not text or text not in df.columns:
        raise ModelUnavailableError("The dataset has no text column to cluster.")
    texts = df[text].astype(str)
    vec = TfidfVectorizer(stop_words="english", max_features=4000, min_df=2, sublinear_tf=True)
    return vec.fit_transform(texts)


def _cluster_coords(tfidf_matrix) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    n_comp = int(min(2, tfidf_matrix.shape[1], tfidf_matrix.shape[0] - 1))
    if n_comp < 2:
        raise ModelUnavailableError("Not enough distinct terms to cluster the news.")
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(tfidf_matrix)
    norm = coords / (np.linalg.norm(coords, axis=1, keepdims=True) + 1e-9)
    return norm


def train_clusters(df: pd.DataFrame, n_clusters: int = 5) -> dict:
    """K-means over TF-IDF text reduced to 2-D (for plotting) and normalized."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if not clustering_available(df):
        raise ModelUnavailableError("The dataset has no news text to cluster.")

    tfidf = _text_matrix(df)
    coords = _cluster_coords(tfidf)
    k = int(max(2, min(n_clusters, len(coords) - 1)))

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(coords)
    sil = float(silhouette_score(coords, labels)) if len(set(labels)) > 1 else float("nan")

    text = _logical_col("text")
    sent = _logical_col("sentiment")
    change = _logical_col("price_change_percent")
    company = _logical_col("ticker")

    frame = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "cluster": labels})
    frame["news"] = df[text].astype(str).to_numpy()
    if sent and sent in df.columns:
        frame["sentiment"] = df[sent].astype(str).to_numpy()
    if change and change in df.columns:
        frame["change"] = pd.to_numeric(df[change], errors="coerce").to_numpy()
    if company and company in df.columns:
        frame["company"] = df[company].astype(str).to_numpy()

    stats = _cluster_stats(frame, sent, change)
    return {"points": frame, "silhouette": sil, "k": k, "stats": stats, "inertia": float(km.inertia_)}


def cluster_quality(df: pd.DataFrame, ks=range(2, 9)) -> pd.DataFrame:
    """Inertia + silhouette across a range of k for the elbow/silhouette chart."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    tfidf = _text_matrix(df)
    coords = _cluster_coords(tfidf)
    rows = []
    for k in ks:
        k = int(k)
        if k >= len(coords):
            continue
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(coords)
        sil = float(silhouette_score(coords, labels)) if len(set(labels)) > 1 else float("nan")
        rows.append({"k": k, "silhouette": sil})
    return pd.DataFrame(rows)


def _cluster_stats(frame: pd.DataFrame, sent: str | None, change: str | None) -> pd.DataFrame:
    rows = []
    for cid, grp in frame.groupby("cluster"):
        entry = {"cluster": int(cid), "size": int(len(grp))}
        if sent and sent in grp.columns:
            dominant = grp["sentiment"].value_counts()
            entry["dominant_sentiment"] = dominant.index[0] if not dominant.empty else "n/a"
            entry["sentiment_mix"] = " / ".join(f"{k}: {v}" for k, v in dominant.head(3).items())
        if change and change in grp.columns:
            vals = pd.to_numeric(grp["change"], errors="coerce").dropna()
            entry["avg_change_pct"] = float(vals.mean()) if not vals.empty else float("nan")
        sample = str(grp["news"].mode().iat[0]) if "news" in grp.columns and not grp["news"].empty else ""
        entry["example"] = sample[:120]
        rows.append(entry)
    return pd.DataFrame(rows)
