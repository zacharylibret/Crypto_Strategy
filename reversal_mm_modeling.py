from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from paper_workflow import (
    ModelBundle,
    audit_feature_distributions,
    fit_statsmodels_logit,
    suggest_log1p_features,
)
from reversal_mm_features import Dataset


@dataclass(slots=True)
class ModelTrainingResult:
    bundle: ModelBundle
    summary: dict
    logistic_coefficients: pd.DataFrame
    logistic_eval_table: pd.DataFrame


def _safe_binary_metric(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


def evaluate_probabilistic_classifier(y_true: np.ndarray, prob_pos: np.ndarray, split: str) -> dict[str, float | int | str]:
    y_true = np.asarray(y_true, dtype=np.int8)
    prob_pos = np.clip(np.asarray(prob_pos, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    pred = (prob_pos >= 0.5).astype(np.int8)
    return {
        "split": split,
        "n_orders": int(y_true.size),
        "pos_rate": float(np.mean(y_true)) if y_true.size else np.nan,
        "roc_auc": _safe_binary_metric(roc_auc_score, y_true, prob_pos) if np.unique(y_true).size > 1 else np.nan,
        "avg_precision": _safe_binary_metric(average_precision_score, y_true, prob_pos) if np.unique(y_true).size > 1 else np.nan,
        "log_loss": _safe_binary_metric(log_loss, y_true, prob_pos, labels=[0, 1]),
        "brier": _safe_binary_metric(brier_score_loss, y_true, prob_pos),
        "accuracy@0.5": float(np.mean(pred == y_true)) if y_true.size else np.nan,
    }


def fit_reversal_models(
    train_dataset: Dataset,
    validation_dataset: Dataset | None,
    model_cfg,
    fill_surface,
) -> ModelTrainingResult:
    train_order = np.argsort(train_dataset.post_ts)

    X_train = np.nan_to_num(train_dataset.X[train_order], nan=0.0, posinf=0.0, neginf=0.0)
    y_train = train_dataset.y[train_order]
    X_val = None
    y_val = None
    if validation_dataset is not None and validation_dataset.X.shape[0] > 0:
        val_order = np.argsort(validation_dataset.post_ts)
        X_val = np.nan_to_num(validation_dataset.X[val_order], nan=0.0, posinf=0.0, neginf=0.0)
        y_val = validation_dataset.y[val_order]

    audit_df = audit_feature_distributions(X_train, train_dataset.feature_names)
    log1p_features = suggest_log1p_features(audit_df, exclude_exact=("is_weekend",))
    available = set(train_dataset.feature_names)
    interaction_pairs = [
        pair
        for pair in [
            ("opp_count_80ms_w0", "total_opp_80ms_w0"),
            ("top_near_liq", "total_opp_80ms_w0"),
            ("ob_near_half", "opp_count_1s_w0"),
            ("ret_vwap_80ms_w1", "opp_count_1s_w0"),
            ("amplitude_80ms_w0", "amplitude_1s_w0"),
            ("amplitude_80ms_w0", "ret_autocov_300s_w1"),
        ]
        if pair[0] in available and pair[1] in available
    ]

    print("  Fitting statsmodels logistic regression...")
    logistic, logistic_coef_df, logistic_summary = fit_statsmodels_logit(
        X_train,
        y_train,
        train_dataset.feature_names,
        max_iter=model_cfg.logistic_max_iter,
        log1p_features=log1p_features,
        interaction_pairs=interaction_pairs,
    )
    logistic_train_probs = logistic.predict_proba(X_train)[:, 1]
    logistic_eval_rows = [
        evaluate_probabilistic_classifier(y_train, logistic_train_probs, "train"),
    ]
    if X_val is not None and y_val is not None:
        logistic_val_probs = logistic.predict_proba(X_val)[:, 1]
        logistic_eval_rows.append(evaluate_probabilistic_classifier(y_val, logistic_val_probs, "validation"))
        print("  Validation threshold selection will be done from the HftBacktest validation sweep only.")
    else:
        print("  Final refit complete on the provided training set.")
    logistic_eval_table = pd.DataFrame(logistic_eval_rows)

    summary = {
        "logit_summary": logistic_summary,
        "logit_eval_rows": {row["split"]: row for row in logistic_eval_rows},
        "feature_audit": audit_df,
        "log1p_features": log1p_features,
        "interaction_pairs": interaction_pairs,
        "selected_threshold": None,
    }
    bundle = ModelBundle(
        fill_surface=fill_surface,
        logistic=logistic,
        feature_names=train_dataset.feature_names,
        selected_threshold=None,
        validation_summary=summary,
        logistic_coef_table=logistic_coef_df,
    )
    return ModelTrainingResult(
        bundle=bundle,
        summary=summary,
        logistic_coefficients=logistic_coef_df,
        logistic_eval_table=logistic_eval_table,
    )
