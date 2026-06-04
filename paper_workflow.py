from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay, QhullError
from scipy.special import expit


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _extract_source_date(day_source: Any) -> str:
    stem = Path(str(day_source)).stem
    match = DATE_RE.search(stem)
    if match is None:
        raise ValueError(f"Could not extract YYYY-MM-DD date from source {day_source!r}")
    return match.group(1)


def combine_signal_days(
    day_sources: Iterable[Any],
    signal_map: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    ordered = sorted(day_sources, key=_extract_source_date)
    keys = [_extract_source_date(src) for src in ordered]
    fields = ["ts", "imbalance", "fill_prob_buy", "fill_prob_sell", "p_buy", "p_sell"]
    if not keys:
        return {field: np.empty(0, dtype=np.int64 if field == "ts" else np.float64) for field in fields}

    combined: dict[str, np.ndarray] = {}
    for field in fields:
        parts: list[np.ndarray] = []
        for key in keys:
            block = signal_map[key]
            if field not in block:
                raise KeyError(f"Signal block {key} missing field {field}")
            parts.append(np.asarray(block[field]))
        arr = np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
        combined[field] = arr.astype(np.int64 if field == "ts" else np.float64, copy=False)

    if combined["ts"].size:
        order = np.argsort(combined["ts"], kind="stable")
        for field in fields:
            combined[field] = combined[field][order]
    return combined


@dataclass(slots=True)
class FillSurface:
    near_edges: np.ndarray
    opp_edges: np.ndarray
    probs: np.ndarray
    counts: np.ndarray
    fills: np.ndarray
    global_fill_rate: float
    q99_near: float
    q99_opp: float
    n_orders_used: int
    max_notional_cap: float | None = None
    smooth_near: np.ndarray | None = None
    smooth_opp: np.ndarray | None = None
    smooth_probs: np.ndarray | None = None

    def predict(self, near: np.ndarray, opp: np.ndarray) -> np.ndarray:
        near = np.asarray(near, dtype=np.float64)
        opp = np.asarray(opp, dtype=np.float64)
        if self.smooth_near is None or self.smooth_opp is None or self.smooth_probs is None:
            nc = np.clip(near, self.near_edges[0], self.q99_near)
            oc = np.clip(opp, self.opp_edges[0], self.q99_opp)
            i = np.clip(np.searchsorted(self.near_edges, nc, side="right") - 1, 0, self.probs.shape[0] - 1)
            j = np.clip(np.searchsorted(self.opp_edges, oc, side="right") - 1, 0, self.probs.shape[1] - 1)
            out = self.probs[i, j]
            return np.where(np.isfinite(out), out, self.global_fill_rate)
        nc = np.clip(near, float(np.ravel(self.smooth_near)[0]), float(np.ravel(self.smooth_near)[-1]))
        oc = np.clip(opp, float(np.ravel(self.smooth_opp)[0]), float(np.ravel(self.smooth_opp)[-1]))
        return _bilinear_interp(
            self.smooth_near,
            self.smooth_opp,
            self.smooth_probs,
            nc,
            oc,
            fill_value=self.global_fill_rate,
        ).clip(0.0, 1.0)


def _bilinear_interp(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    fill_value: float,
) -> np.ndarray:
    x_grid = np.ravel(np.asarray(x_grid, dtype=np.float64))
    y_grid = np.ravel(np.asarray(y_grid, dtype=np.float64))
    z_grid = np.asarray(z_grid, dtype=np.float64)
    if z_grid.ndim != 2:
        raise ValueError("z_grid must be 2-dimensional")

    if (
        z_grid.shape[0] == y_grid.size
        and z_grid.shape[1] == x_grid.size
        and z_grid.shape != (x_grid.size, y_grid.size)
    ):
        warnings.warn(
            "Fill-surface array axes appear transposed relative to the interpolation grids; transposing.",
            RuntimeWarning,
        )
        z_grid = z_grid.T

    nx = min(len(x_grid), z_grid.shape[0])
    ny = min(len(y_grid), z_grid.shape[1])
    if nx < 2 or ny < 2:
        return np.full(np.asarray(x).shape, fill_value, dtype=np.float64)
    if nx != len(x_grid) or ny != len(y_grid):
        warnings.warn(
            "Fill-surface grid lengths do not match the smoothed surface array; truncating to the common domain for safe interpolation.",
            RuntimeWarning,
        )
        x_grid = x_grid[:nx]
        y_grid = y_grid[:ny]
        z_grid = z_grid[:nx, :ny]

    x = np.clip(np.asarray(x, dtype=np.float64), x_grid[0], x_grid[-1])
    y = np.clip(np.asarray(y, dtype=np.float64), y_grid[0], y_grid[-1])
    ix = np.searchsorted(x_grid, x, side="right") - 1
    iy = np.searchsorted(y_grid, y, side="right") - 1
    ix = np.clip(ix, 0, max(z_grid.shape[0] - 2, 0))
    iy = np.clip(iy, 0, max(z_grid.shape[1] - 2, 0))
    ix1 = np.clip(ix + 1, 0, z_grid.shape[0] - 1)
    iy1 = np.clip(iy + 1, 0, z_grid.shape[1] - 1)
    x0 = x_grid[ix]
    x1 = x_grid[ix1]
    y0 = y_grid[iy]
    y1 = y_grid[iy1]
    z00 = z_grid[ix, iy]
    z10 = z_grid[ix1, iy]
    z01 = z_grid[ix, iy1]
    z11 = z_grid[ix1, iy1]
    tx = np.divide(x - x0, np.maximum(x1 - x0, 1e-12))
    ty = np.divide(y - y0, np.maximum(y1 - y0, 1e-12))
    out = (
        (1.0 - tx) * (1.0 - ty) * z00
        + tx * (1.0 - ty) * z10
        + (1.0 - tx) * ty * z01
        + tx * ty * z11
    )
    return np.where(np.isfinite(out), out, fill_value)


@dataclass(slots=True)
class StatsmodelsLogitModel:
    result: Any
    feature_names: list[str]
    design_feature_names: list[str]
    keep_mask: np.ndarray
    mean_: np.ndarray
    scale_: np.ndarray
    coef_table_: pd.DataFrame
    summary_: dict[str, Any]
    log1p_mask_: np.ndarray | None = None
    interaction_pairs_idx_: tuple[tuple[int, int], ...] = ()
    interaction_terms_: tuple[str, ...] = ()

    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if self.log1p_mask_ is not None and np.any(self.log1p_mask_):
            X = X.copy()
            X[:, self.log1p_mask_] = np.log1p(np.maximum(X[:, self.log1p_mask_], 0.0))
        if self.interaction_pairs_idx_:
            parts = [X]
            for left_idx, right_idx in self.interaction_pairs_idx_:
                parts.append((X[:, left_idx] * X[:, right_idx]).reshape(-1, 1))
            X = np.hstack(parts)
        return X

    def _transform(self, X: np.ndarray) -> np.ndarray:
        Xd = self._design_matrix(X)
        Xk = Xd[:, self.keep_mask]
        Z = (Xk - self.mean_[self.keep_mask]) / self.scale_[self.keep_mask]
        return sm.add_constant(Z, has_constant="add")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        exog = self._transform(X)
        logits = np.asarray(exog @ np.asarray(self.result.params, dtype=np.float64), dtype=np.float64)
        p = expit(np.clip(logits, -60.0, 60.0))
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


@dataclass(slots=True)
class ModelBundle:
    fill_surface: FillSurface
    logistic: StatsmodelsLogitModel
    feature_names: list[str]
    selected_threshold: float | None
    validation_summary: dict[str, Any]
    logistic_coef_table: pd.DataFrame


def compute_signed_return_bp(
    ref_price: np.ndarray,
    future_price: np.ndarray,
    side: np.ndarray,
    *,
    bp_scale: float = 10_000.0,
) -> np.ndarray:
    ref = np.asarray(ref_price, dtype=np.float64)
    future = np.asarray(future_price, dtype=np.float64)
    signed_side = np.asarray(side, dtype=np.float64)
    out = np.full(np.broadcast(ref, future, signed_side).shape, np.nan, dtype=np.float64)
    valid = np.isfinite(ref) & (ref > 0.0) & np.isfinite(future) & (future > 0.0) & np.isfinite(signed_side)
    if np.any(valid):
        out[valid] = signed_side[valid] * ((future[valid] / ref[valid]) - 1.0) * float(bp_scale)
    return out


def trade_intensity_count_per_second(
    trade_count: np.ndarray | float | int,
    window_seconds: float,
    elapsed_seconds: np.ndarray | float | None = None,
) -> np.ndarray:
    counts = np.asarray(trade_count, dtype=np.float64)
    out = np.zeros_like(counts, dtype=np.float64)
    if window_seconds <= 0:
        return out
    out = np.where(counts > 0.0, counts / float(window_seconds), 0.0)
    if elapsed_seconds is None:
        return out
    elapsed = np.asarray(elapsed_seconds, dtype=np.float64)
    dense = (counts >= 2.0) & np.isfinite(elapsed) & (elapsed > 0.0)
    out[dense] = (counts[dense] - 1.0) / elapsed[dense]
    return out


def horizon_indices(
    grid_times: np.ndarray,
    anchor_ts: np.ndarray,
    *,
    horizon_ns: int,
    side: str = "left",
) -> np.ndarray:
    grid = np.asarray(grid_times, dtype=np.int64)
    anchor = np.asarray(anchor_ts, dtype=np.int64)
    if grid.size == 0:
        return np.full(anchor.shape, -1, dtype=np.int64)
    target = anchor + int(horizon_ns)
    idx = np.searchsorted(grid, target, side=side)
    if side == "right":
        idx = idx - 1
    return np.clip(idx, 0, grid.size - 1).astype(np.int64, copy=False)


def compute_post_fill_markout_bp(
    fill_ts: np.ndarray,
    ref_price: np.ndarray,
    side: np.ndarray,
    grid_times: np.ndarray,
    future_price_grid: np.ndarray,
    *,
    horizon_ns: int,
    lookup_side: str = "left",
    bp_scale: float = 10_000.0,
) -> np.ndarray:
    fill_ts = np.asarray(fill_ts, dtype=np.int64)
    ref_price = np.asarray(ref_price, dtype=np.float64)
    side = np.asarray(side, dtype=np.float64)
    grid_times = np.asarray(grid_times, dtype=np.int64)
    future_price_grid = np.asarray(future_price_grid, dtype=np.float64)
    out = np.full(fill_ts.shape, np.nan, dtype=np.float64)
    if fill_ts.size == 0 or grid_times.size == 0:
        return out
    idx = horizon_indices(grid_times, fill_ts, horizon_ns=horizon_ns, side=lookup_side)
    valid = (idx >= 0) & (idx < grid_times.size) & np.isfinite(ref_price) & (ref_price > 0.0)
    if np.any(valid):
        out[valid] = compute_signed_return_bp(
            ref_price[valid],
            future_price_grid[idx[valid]],
            side[valid],
            bp_scale=bp_scale,
        )
    return out


def compute_next_mid_change_return_bp(
    fill_ts: np.ndarray,
    ref_price: np.ndarray,
    side: np.ndarray,
    mid_change_times: np.ndarray,
    mid_change_best_bid: np.ndarray,
    mid_change_best_ask: np.ndarray,
    *,
    strict_after_fill: bool = True,
    bp_scale: float = 10_000.0,
) -> np.ndarray:
    fill_ts = np.asarray(fill_ts, dtype=np.int64)
    ref_price = np.asarray(ref_price, dtype=np.float64)
    side = np.asarray(side, dtype=np.float64)
    mid_change_times = np.asarray(mid_change_times, dtype=np.int64)
    mid_change_best_bid = np.asarray(mid_change_best_bid, dtype=np.float64)
    mid_change_best_ask = np.asarray(mid_change_best_ask, dtype=np.float64)
    out = np.full(fill_ts.shape, np.nan, dtype=np.float64)
    if fill_ts.size == 0 or mid_change_times.size == 0:
        return out

    idx = np.searchsorted(mid_change_times, fill_ts, side="right" if strict_after_fill else "left")
    valid = (idx >= 0) & (idx < mid_change_times.size) & np.isfinite(ref_price) & (ref_price > 0.0)
    if not np.any(valid):
        return out
    next_bid = mid_change_best_bid[idx[valid]]
    next_ask = mid_change_best_ask[idx[valid]]
    next_mid = np.where(
        np.isfinite(next_bid) & (next_bid > 0.0) & np.isfinite(next_ask) & (next_ask > 0.0),
        0.5 * (next_bid + next_ask),
        np.where(
            np.isfinite(next_bid) & (next_bid > 0.0),
            next_bid,
            np.where(np.isfinite(next_ask) & (next_ask > 0.0), next_ask, np.nan),
        ),
    )
    out[valid] = compute_signed_return_bp(ref_price[valid], next_mid, side[valid], bp_scale=bp_scale)
    return out


def _equal_width_edges(vals: np.ndarray, bins: int, quantile: float) -> tuple[np.ndarray, float]:
    finite = vals[np.isfinite(vals)]
    lo = float(np.min(finite))
    hi = float(np.quantile(finite, quantile))
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, bins + 1, dtype=np.float64), hi


def build_empirical_fill_surface(
    near_size: np.ndarray,
    opp_size: np.ndarray,
    filled: np.ndarray,
    bins: int = 20,
    quantile: float = 0.99,
    max_notional_cap: float | None = None,
    interpolation_grid_size: int = 121,
) -> FillSurface:
    near = np.asarray(near_size, dtype=np.float64)
    opp = np.asarray(opp_size, dtype=np.float64)
    y = np.asarray(filled, dtype=np.float64)
    mask = np.isfinite(near) & np.isfinite(opp) & np.isfinite(y)
    if max_notional_cap is not None:
        cap = float(max_notional_cap)
        mask &= (near <= cap) & (opp <= cap)
    near = near[mask]
    opp = opp[mask]
    y = y[mask]

    near_edges, q99_near = _equal_width_edges(near, bins, quantile)
    opp_edges, q99_opp = _equal_width_edges(opp, bins, quantile)
    nc = np.clip(near, near_edges[0], q99_near)
    oc = np.clip(opp, opp_edges[0], q99_opp)
    ii = np.clip(np.searchsorted(near_edges, nc, side="right") - 1, 0, bins - 1)
    jj = np.clip(np.searchsorted(opp_edges, oc, side="right") - 1, 0, bins - 1)

    counts = np.zeros((bins, bins), dtype=np.int64)
    fills = np.zeros((bins, bins), dtype=np.float64)
    np.add.at(counts, (ii, jj), 1)
    np.add.at(fills, (ii, jj), y)
    probs = np.full((bins, bins), np.nan, dtype=np.float64)
    nz = counts > 0
    probs[nz] = fills[nz] / counts[nz]

    surface = FillSurface(
        near_edges=near_edges,
        opp_edges=opp_edges,
        probs=probs,
        counts=counts,
        fills=fills,
        global_fill_rate=float(np.mean(y)),
        q99_near=q99_near,
        q99_opp=q99_opp,
        n_orders_used=int(y.shape[0]),
        max_notional_cap=float(max_notional_cap) if max_notional_cap is not None else None,
    )
    _attach_delaunay_surface(surface, grid_size=interpolation_grid_size)
    return surface


def _attach_delaunay_surface(surface: FillSurface, *, grid_size: int) -> None:
    near_centers = 0.5 * (surface.near_edges[:-1] + surface.near_edges[1:])
    opp_centers = 0.5 * (surface.opp_edges[:-1] + surface.opp_edges[1:])
    qn, qo = np.meshgrid(near_centers, opp_centers, indexing="ij")
    mask = np.isfinite(surface.probs)
    points = np.column_stack([qn[mask], qo[mask]])
    values = surface.probs[mask]
    surface.smooth_near = np.linspace(surface.near_edges[0], surface.q99_near, grid_size, dtype=np.float64)
    surface.smooth_opp = np.linspace(surface.opp_edges[0], surface.q99_opp, grid_size, dtype=np.float64)
    gx, gy = np.meshgrid(surface.smooth_near, surface.smooth_opp, indexing="ij")
    if points.shape[0] == 0:
        surface.smooth_probs = np.full_like(gx, surface.global_fill_rate, dtype=np.float64)
        return
    try:
        tri = Delaunay(points)
    except QhullError:
        nearest = NearestNDInterpolator(points, values)
        smooth = nearest(gx, gy)
        surface.smooth_probs = np.clip(np.where(np.isfinite(smooth), smooth, surface.global_fill_rate), 0.0, 1.0)
        return
    linear = LinearNDInterpolator(tri, values)
    smooth = linear(gx, gy)
    if np.any(~np.isfinite(smooth)):
        nearest = NearestNDInterpolator(points, values)
        smooth = np.where(np.isfinite(smooth), smooth, nearest(gx, gy))
    surface.smooth_probs = np.clip(np.where(np.isfinite(smooth), smooth, surface.global_fill_rate), 0.0, 1.0)


def surface_regression_frame(surface: FillSurface, *, use_smoothed_surface: bool = True) -> pd.DataFrame:
    if use_smoothed_surface and surface.smooth_near is not None and surface.smooth_opp is not None and surface.smooth_probs is not None:
        qn, qo = np.meshgrid(surface.smooth_near, surface.smooth_opp, indexing="ij")
        z = surface.smooth_probs
        weight = np.full(z.shape, np.nan, dtype=np.float64)
    else:
        near_centers = 0.5 * (surface.near_edges[:-1] + surface.near_edges[1:])
        opp_centers = 0.5 * (surface.opp_edges[:-1] + surface.opp_edges[1:])
        qn, qo = np.meshgrid(near_centers, opp_centers, indexing="ij")
        z = surface.probs
        weight = surface.counts.astype(np.float64)

    qn_norm = qn / max(surface.q99_near, 1e-12)
    qo_norm = qo / max(surface.q99_opp, 1e-12)
    imb = (qn - qo) / np.maximum(qn + qo, 1e-12)
    df = pd.DataFrame(
        {
            "q_near": qn.ravel(),
            "q_opp": qo.ravel(),
            "q_near_norm": qn_norm.ravel(),
            "q_opp_norm": qo_norm.ravel(),
            "imbalance": imb.ravel(),
            "surface_fill_prob": z.ravel(),
            "count": weight.ravel(),
        }
    )
    return df[np.isfinite(df["surface_fill_prob"])].reset_index(drop=True)


def fit_fill_surface_ols(surface: FillSurface, *, use_smoothed_surface: bool = True) -> dict[str, Any]:
    df = surface_regression_frame(surface, use_smoothed_surface=use_smoothed_surface)
    X = sm.add_constant(df[["q_near_norm", "q_opp_norm", "imbalance"]], has_constant="add")
    result = sm.OLS(df["surface_fill_prob"], X).fit()
    df["fitted_fill_prob"] = result.predict(X)
    coef_table = pd.DataFrame(
        {
            "term": result.params.index,
            "coef": result.params.values,
            "std_err": result.bse.values,
            "p_value": result.pvalues.values,
        }
    )
    summary = {
        "r_squared": float(result.rsquared),
        "n_surface_points": int(df.shape[0]),
        "mean_abs_error": float(np.mean(np.abs(df["surface_fill_prob"] - df["fitted_fill_prob"]))),
        "target": "smoothed_delaunay_surface" if use_smoothed_surface else "raw_empirical_bins",
        "regressors": ["const", "q_near_norm", "q_opp_norm", "imbalance"],
    }
    return {"result": result, "frame": df, "coef_table": coef_table, "summary": summary}


def _sanitize_X(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def audit_feature_distributions(
    X: np.ndarray,
    feature_names: Iterable[str],
    *,
    min_zero_share: float = 0.05,
    min_q99_to_median: float = 20.0,
    min_skew_proxy: float = 20.0,
    exclude_prefixes: Iterable[str] = ("ret_vwap_", "ret_sum_", "ret_autocov_", "stdev_"),
    exclude_exact: Iterable[str] = (),
    eps: float = 1e-12,
) -> pd.DataFrame:
    X = _sanitize_X(X)
    feature_names = list(feature_names)
    exclude_prefixes = tuple(exclude_prefixes)
    exclude_exact = set(exclude_exact)
    rows: list[dict[str, Any]] = []
    for col_idx, name in enumerate(feature_names):
        col = np.asarray(X[:, col_idx], dtype=np.float64)
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            q01 = median = q99 = mean = std = min_v = max_v = np.nan
            zero_share = neg_share = q99_to_median = skew_proxy = np.nan
        else:
            q01, median, q99 = np.quantile(finite, [0.01, 0.50, 0.99])
            mean = float(np.mean(finite))
            std = float(np.std(finite, ddof=0))
            min_v = float(np.min(finite))
            max_v = float(np.max(finite))
            zero_share = float(np.mean(finite == 0.0))
            neg_share = float(np.mean(finite < 0.0))
            denom = max(abs(float(median)), eps)
            q99_to_median = float(q99 / denom)
            skew_proxy = float(q99_to_median - 1.0)
        candidate = (
            np.isfinite(zero_share)
            and np.isfinite(neg_share)
            and neg_share == 0.0
            and name not in exclude_exact
            and not name.startswith(exclude_prefixes)
            and (
                zero_share >= float(min_zero_share)
                or q99_to_median >= float(min_q99_to_median)
                or skew_proxy >= float(min_skew_proxy)
            )
        )
        rows.append(
            {
                "feature": name,
                "mean": mean,
                "std": std,
                "min": min_v,
                "q01": float(q01) if np.isfinite(q01) else np.nan,
                "median": float(median) if np.isfinite(median) else np.nan,
                "q99": float(q99) if np.isfinite(q99) else np.nan,
                "max": max_v,
                "zero_share": zero_share,
                "neg_share": neg_share,
                "q99_to_median": q99_to_median,
                "skew_proxy": skew_proxy,
                "log1p_candidate": bool(candidate),
            }
        )
    return pd.DataFrame(rows)


def suggest_log1p_features(
    audit_df: pd.DataFrame,
    *,
    min_zero_share: float = 0.05,
    min_q99_to_median: float = 20.0,
    min_skew_proxy: float = 20.0,
    exclude_prefixes: Iterable[str] = ("ret_vwap_", "ret_sum_", "ret_autocov_", "stdev_"),
    exclude_exact: Iterable[str] = (),
) -> list[str]:
    exclude_prefixes = tuple(exclude_prefixes)
    exclude_exact = set(exclude_exact)
    df = audit_df.copy()
    keep = (
        (df.get("neg_share", 1.0) == 0.0)
        & (
            (df.get("zero_share", 0.0) >= float(min_zero_share))
            | (df.get("q99_to_median", 0.0) >= float(min_q99_to_median))
            | (df.get("skew_proxy", 0.0) >= float(min_skew_proxy))
        )
    )
    df = df.loc[keep].copy()
    df = df.loc[~df["feature"].isin(exclude_exact)]
    df = df.loc[~df["feature"].str.startswith(exclude_prefixes)]
    return df["feature"].tolist()


def _build_logit_design(
    X: np.ndarray,
    feature_names: Iterable[str],
    *,
    log1p_features: Iterable[str] | None = None,
    interaction_pairs: Iterable[tuple[str, str]] | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, tuple[tuple[int, int], ...], tuple[str, ...]]:
    X = _sanitize_X(X)
    feature_names = list(feature_names)
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    log1p_set = set(log1p_features or [])
    log1p_mask = np.array([name in log1p_set for name in feature_names], dtype=bool)
    if np.any(log1p_mask):
        X = X.copy()
        X[:, log1p_mask] = np.log1p(np.maximum(X[:, log1p_mask], 0.0))

    design_feature_names = [f"log1p({name})" if mask else name for name, mask in zip(feature_names, log1p_mask)]
    interaction_pairs_idx: list[tuple[int, int]] = []
    interaction_terms: list[str] = []
    if interaction_pairs:
        parts = [X]
        for left_name, right_name in interaction_pairs:
            if left_name not in name_to_idx or right_name not in name_to_idx:
                continue
            left_idx = name_to_idx[left_name]
            right_idx = name_to_idx[right_name]
            parts.append((X[:, left_idx] * X[:, right_idx]).reshape(-1, 1))
            interaction_pairs_idx.append((left_idx, right_idx))
            interaction_terms.append(f"{design_feature_names[left_idx]} * {design_feature_names[right_idx]}")
        X = np.hstack(parts)
        design_feature_names.extend(interaction_terms)
    return X, design_feature_names, log1p_mask, tuple(interaction_pairs_idx), tuple(interaction_terms)


def fit_statsmodels_logit(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Iterable[str],
    max_iter: int = 4000,
    *,
    log1p_features: Iterable[str] | None = None,
    interaction_pairs: Iterable[tuple[str, str]] | None = None,
) -> tuple[StatsmodelsLogitModel, pd.DataFrame, dict[str, Any]]:
    X = _sanitize_X(X)
    y = np.asarray(y, dtype=np.int8)
    feature_names = list(feature_names)
    X_design, design_feature_names, log1p_mask, interaction_pairs_idx, interaction_terms = _build_logit_design(
        X,
        feature_names,
        log1p_features=log1p_features,
        interaction_pairs=interaction_pairs,
    )
    mean_ = X_design.mean(axis=0)
    scale_ = X_design.std(axis=0, ddof=0)
    keep_mask = scale_ > 1e-12
    safe_scale = np.where(scale_ > 1e-12, scale_, 1.0)
    Z = (X_design[:, keep_mask] - mean_[keep_mask]) / safe_scale[keep_mask]
    exog = sm.add_constant(Z, has_constant="add")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sm.Logit(y, exog).fit(method="lbfgs", maxiter=max_iter, disp=False)
    warning_messages = [f"{type(w.message).__name__}: {w.message}" for w in caught]
    term_names = ["const"] + [design_feature_names[i] for i, keep in enumerate(keep_mask) if keep]
    coef_table = pd.DataFrame(
        {
            "term": term_names,
            "coef": result.params,
            "std_err": result.bse,
            "z_value": result.tvalues,
            "p_value": result.pvalues,
        }
    )
    summary = {
        "converged": bool(result.mle_retvals.get("converged", getattr(result, "converged", True))),
        "iterations": int(result.mle_retvals.get("iterations", -1)),
        "llf": float(result.llf),
        "pseudo_r_squared": float(result.prsquared),
        "n_features_used": int(keep_mask.sum()),
        "n_design_features": int(len(design_feature_names)),
        "log1p_features": sorted(set(log1p_features or [])),
        "interaction_terms": list(interaction_terms),
        "uses_intercept": True,
        "uses_class_weight": False,
        "uses_sample_weights": False,
        "probability_source": "statsmodels_logit_predict",
        "warnings": warning_messages,
    }
    model = StatsmodelsLogitModel(
        result=result,
        feature_names=feature_names,
        design_feature_names=design_feature_names,
        keep_mask=keep_mask,
        mean_=mean_,
        scale_=safe_scale,
        coef_table_=coef_table,
        summary_=summary,
        log1p_mask_=log1p_mask,
        interaction_pairs_idx_=interaction_pairs_idx,
        interaction_terms_=interaction_terms,
    )
    return model, coef_table, summary
