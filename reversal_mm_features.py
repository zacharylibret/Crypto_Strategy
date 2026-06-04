from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from hbt_state_sampler import DayState
from paper_workflow import (
    ModelBundle,
    compute_next_mid_change_return_bp,
    compute_post_fill_markout_bp,
    trade_intensity_count_per_second,
)
from reversal_mm_constants import BP_SCALE, BUY, FILLED, NS_PER_SECOND, SELL

SCALE_LABELS = {
    90_000_000: "90ms",
    1_000_000_000: "1s",
    5_000_000_000: "5s",
    30_000_000_000: "30s",
    300_000_000_000: "300s",
}

BASE_FEATURE_COUNT = 173
CALENDAR_FEATURE_NAMES = ["is_weekend"]


@dataclass(slots=True)
class PrecomputedDay:
    state: DayState
    scales_ns: tuple[int, ...]
    vol_return_interval_ns: int
    vol_lookbacks: tuple[int, ...]
    roll_price_max: dict[int, np.ndarray]
    roll_price_min: dict[int, np.ndarray]
    roll_max_size: dict[int, np.ndarray]
    prefix_buy_count: np.ndarray
    prefix_sell_count: np.ndarray
    prefix_buy_qty: np.ndarray
    prefix_sell_qty: np.ndarray
    prefix_total_qty: np.ndarray
    prefix_notional: np.ndarray
    trade_ret: np.ndarray
    prefix_trade_ret: np.ndarray
    prefix_trade_ret_sq: np.ndarray
    prefix_trade_ret_cross: np.ndarray
    ret10: np.ndarray
    prefix_ret10: np.ndarray
    prefix_ret10_sq: np.ndarray
    step10: int

    @property
    def feature_names(self) -> list[str]:
        return _build_feature_names(self.scales_ns, self.vol_lookbacks)


@dataclass(slots=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray
    filled: np.ndarray
    side: np.ndarray
    post_ts: np.ndarray
    post_idx: np.ndarray
    fill_ts: np.ndarray
    fill_px: np.ndarray
    fill_prob: np.ndarray
    markout_5s_bp: np.ndarray
    next_change_ret_bp: np.ndarray
    feature_names: list[str]


def _build_feature_names(scales: tuple[int, ...], lookbacks: tuple[int, ...]) -> list[str]:
    names: list[str] = []
    for lb in lookbacks:
        names.append(f"stdev_{lb}")
    for sn in scales:
        sc = _scale_label(int(sn))
        for window in range(3):
            names += [f"amplitude_{sc}_w{window}", f"ret_vwap_{sc}_w{window}"]
    for sn in scales:
        sc = _scale_label(int(sn))
        for window in range(3):
            names += [
                f"max_size_{sc}_w{window}",
                f"avg_size_{sc}_w{window}",
                f"same_count_{sc}_w{window}",
                f"opp_count_{sc}_w{window}",
                f"total_same_{sc}_w{window}",
                f"total_opp_{sc}_w{window}",
            ]
    for sn in scales:
        sc = _scale_label(int(sn))
        for window in range(3):
            names += [
                f"ret_autocov_{sc}_w{window}",
                f"ret_sum_{sc}_w{window}",
                f"trade_intensity_{sc}_w{window}",
            ]
    names += ["top_near_liq", "top_opp_liq", "ob_near_half", "ob_opp_half", "totb_mean", "age"]
    if len(names) != BASE_FEATURE_COUNT:
        raise ValueError(f"Expected {BASE_FEATURE_COUNT} base features, got {len(names)}.")
    names += CALENDAR_FEATURE_NAMES
    return names


def _scale_label(scale_ns: int) -> str:
    scale_ns = int(scale_ns)
    if scale_ns in SCALE_LABELS:
        return SCALE_LABELS[scale_ns]
    if scale_ns < NS_PER_SECOND:
        ms = scale_ns / 1_000_000.0
        return f"{int(ms)}ms" if float(ms).is_integer() else f"{ms:g}ms"
    seconds = scale_ns / NS_PER_SECOND
    return f"{int(seconds)}s" if float(seconds).is_integer() else f"{seconds:g}s"


def price_range_bp(high: np.ndarray, low: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ref = np.maximum(np.asarray(ref, dtype=np.float64), 1e-300)
    return (np.asarray(high, dtype=np.float64) - np.asarray(low, dtype=np.float64)) / ref * BP_SCALE


def _roll_extreme(arr: np.ndarray, window: int, is_max: bool) -> np.ndarray:
    out = np.empty_like(arr)
    dq: deque[int] = deque()
    comp = (lambda a, b: a >= b) if is_max else (lambda a, b: a <= b)
    for i in range(arr.shape[0]):
        while dq and dq[0] <= i - window:
            dq.popleft()
        while dq and comp(arr[i], arr[dq[-1]]):
            dq.pop()
        dq.append(i)
        out[i] = arr[dq[0]]
    return out


def build_precomputed_day(
    state: DayState,
    scales_ns: list[int] | tuple[int, ...],
    volatility_return_interval_ns: int,
    volatility_lookbacks: list[int] | tuple[int, ...],
) -> PrecomputedDay:
    scales = tuple(int(x) for x in scales_ns)
    lookbacks = tuple(int(x) for x in volatility_lookbacks)
    step10 = volatility_return_interval_ns // state.interval_ns

    is_buy = (state.trade_side == BUY).astype(np.int64)
    is_sell = (state.trade_side == SELL).astype(np.int64)
    prefix_buy_count = np.concatenate([[0], np.cumsum(is_buy)])
    prefix_sell_count = np.concatenate([[0], np.cumsum(is_sell)])

    buy_qty = np.where(state.trade_side == BUY, state.trade_qty, 0.0)
    sell_qty = np.where(state.trade_side == SELL, state.trade_qty, 0.0)
    prefix_buy_qty = np.concatenate([[0.0], np.cumsum(buy_qty)])
    prefix_sell_qty = np.concatenate([[0.0], np.cumsum(sell_qty)])
    prefix_total_qty = np.concatenate([[0.0], np.cumsum(state.trade_qty)])
    prefix_notional = np.concatenate([[0.0], np.cumsum(state.trade_qty * state.trade_prices)])

    trade_ret = np.zeros(state.trade_prices.shape[0], dtype=np.float64)
    if state.trade_prices.shape[0] > 1:
        trade_ret[1:] = np.log(np.maximum(state.trade_prices[1:], 1e-300)) - np.log(
            np.maximum(state.trade_prices[:-1], 1e-300)
        )
    prefix_trade_ret = np.concatenate([[0.0], np.cumsum(trade_ret)])
    prefix_trade_ret_sq = np.concatenate([[0.0], np.cumsum(trade_ret * trade_ret)])
    trade_ret_cross = np.zeros_like(trade_ret)
    if trade_ret.shape[0] > 1:
        trade_ret_cross[1:] = trade_ret[1:] * trade_ret[:-1]
    prefix_trade_ret_cross = np.concatenate([[0.0], np.cumsum(trade_ret_cross)])

    roll_price_max: dict[int, np.ndarray] = {}
    roll_price_min: dict[int, np.ndarray] = {}
    roll_max_size: dict[int, np.ndarray] = {}
    for scale in scales:
        window = int(scale // state.interval_ns)
        roll_price_max[scale] = _roll_extreme(state.bucket_trade_price_max, window, True)
        roll_price_min[scale] = _roll_extreme(state.bucket_trade_price_min, window, False)
        roll_max_size[scale] = _roll_extreme(state.bucket_trade_max_size, window, True)

    base_prices = state.last_trade_price[step10 - 1 :: step10].astype(np.float64)
    ret10 = np.zeros(base_prices.shape[0], dtype=np.float64)
    if base_prices.shape[0] > 1:
        ret10[1:] = np.log(np.maximum(base_prices[1:], 1e-300)) - np.log(np.maximum(base_prices[:-1], 1e-300))
    prefix_ret10 = np.concatenate([[0.0], np.cumsum(ret10)])
    prefix_ret10_sq = np.concatenate([[0.0], np.cumsum(ret10 * ret10)])

    return PrecomputedDay(
        state=state,
        scales_ns=scales,
        vol_return_interval_ns=volatility_return_interval_ns,
        vol_lookbacks=lookbacks,
        roll_price_max=roll_price_max,
        roll_price_min=roll_price_min,
        roll_max_size=roll_max_size,
        prefix_buy_count=prefix_buy_count,
        prefix_sell_count=prefix_sell_count,
        prefix_buy_qty=prefix_buy_qty,
        prefix_sell_qty=prefix_sell_qty,
        prefix_total_qty=prefix_total_qty,
        prefix_notional=prefix_notional,
        trade_ret=trade_ret,
        prefix_trade_ret=prefix_trade_ret,
        prefix_trade_ret_sq=prefix_trade_ret_sq,
        prefix_trade_ret_cross=prefix_trade_ret_cross,
        ret10=ret10,
        prefix_ret10=prefix_ret10,
        prefix_ret10_sq=prefix_ret10_sq,
        step10=step10,
    )


def _take_prefix(prefix: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return prefix[b] - prefix[a]


def _stdev_k(pre: PrecomputedDay, idx: np.ndarray, lookback: int) -> np.ndarray:
    base_idx = np.maximum(((idx + 1) // pre.step10) - 1, 0)
    end = base_idx
    start = np.maximum(end - lookback + 1, 1)
    n = np.maximum(end - start + 1, 0)
    sx = pre.prefix_ret10[end + 1] - pre.prefix_ret10[start]
    sx2 = pre.prefix_ret10_sq[end + 1] - pre.prefix_ret10_sq[start]
    mean = np.divide(sx, np.maximum(n, 1), out=np.zeros_like(sx), where=n > 0)
    var = np.maximum(np.divide(sx2, np.maximum(n, 1), out=np.zeros_like(sx2), where=n > 0) - mean * mean, 0.0)
    out = np.zeros_like(var)
    mask = n > 1
    out[mask] = np.sqrt(var[mask])
    return out


def _window_bucket_bounds(idx: np.ndarray, window_size: int, lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    end = idx - lag * window_size
    valid = end >= 0
    end = np.clip(end, 0, None)
    start = np.clip(end - window_size + 1, 0, None)
    return start, end, valid


def _trade_bounds(state: DayState, start: np.ndarray, end: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hi = np.where(valid, state.trade_hi[end], 0)
    lo_idx = np.maximum(start - 1, 0)
    lo = np.where(start > 0, state.trade_hi[lo_idx], 0)
    lo = np.where(valid, lo, 0)
    return lo.astype(np.int64), hi.astype(np.int64)


def _is_weekend_value(day_key: str) -> np.float32:
    return np.float32(1.0 if datetime.strptime(day_key, "%Y-%m-%d").weekday() >= 5 else 0.0)


def _append_calendar_features(X: np.ndarray, day_key: str) -> np.ndarray:
    weekend_col = np.full((X.shape[0], 1), _is_weekend_value(day_key), dtype=np.float32)
    return np.hstack([np.asarray(X, dtype=np.float32), weekend_col])


def feature_block(pre: PrecomputedDay, idx: np.ndarray, side: int | np.ndarray, impute_missing: bool = True) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    side_arr = np.full(idx.shape[0], side, dtype=np.int8) if np.isscalar(side) else np.asarray(side, dtype=np.int8)
    state = pre.state
    n_rows = idx.shape[0]
    X = np.zeros((n_rows, BASE_FEATURE_COUNT), dtype=np.float64)
    col = 0
    sign = side_arr.astype(np.float64)

    near_px = np.where(side_arr == BUY, state.best_bid[idx], state.best_ask[idx])
    opp_px = np.where(side_arr == BUY, state.best_ask[idx], state.best_bid[idx])
    mid = 0.5 * (state.best_bid[idx] + state.best_ask[idx])
    mid = np.where(np.isfinite(mid), mid, np.where(np.isfinite(near_px), near_px, opp_px))

    for lookback in pre.vol_lookbacks:
        X[:, col] = _stdev_k(pre, idx, lookback)
        col += 1

    for scale in pre.scales_ns:
        window_size = int(scale // state.interval_ns)
        for lag in range(3):
            start, end, valid = _window_bucket_bounds(idx, window_size, lag)
            lo, hi = _trade_bounds(state, start, end, valid)
            n_trades = hi - lo

            pmax = pre.roll_price_max[scale][end]
            pmin = pre.roll_price_min[scale][end]
            amp = np.zeros(n_rows, dtype=np.float64)
            mask = valid & (n_trades > 0) & np.isfinite(pmax) & np.isfinite(pmin)
            amp[mask] = price_range_bp(pmax[mask], pmin[mask], np.where(np.isfinite(near_px[mask]), near_px[mask], mid[mask]))
            X[:, col] = amp
            col += 1

            total_qty = _take_prefix(pre.prefix_total_qty, lo, hi)
            total_notional = _take_prefix(pre.prefix_notional, lo, hi)
            vwap = np.array(mid, copy=True, dtype=np.float64)
            np.divide(total_notional, total_qty, out=vwap, where=total_qty > 0)
            ret_vwap = sign * (np.log(np.maximum(vwap, 1e-300)) - np.log(np.maximum(near_px, 1e-300)))
            ret_vwap[~valid] = 0.0
            X[:, col] = ret_vwap
            col += 1

    for scale in pre.scales_ns:
        window_size = int(scale // state.interval_ns)
        for lag in range(3):
            start, end, valid = _window_bucket_bounds(idx, window_size, lag)
            lo, hi = _trade_bounds(state, start, end, valid)
            buy_count = _take_prefix(pre.prefix_buy_count, lo, hi).astype(np.float64)
            sell_count = _take_prefix(pre.prefix_sell_count, lo, hi).astype(np.float64)
            buy_qty = _take_prefix(pre.prefix_buy_qty, lo, hi)
            sell_qty = _take_prefix(pre.prefix_sell_qty, lo, hi)
            total_qty = buy_qty + sell_qty
            total_count = buy_count + sell_count

            max_size = pre.roll_max_size[scale][end].copy()
            max_size[~valid] = 0.0
            avg_size = np.divide(total_qty, np.maximum(total_count, 1.0), out=np.zeros_like(total_qty), where=total_count > 0)
            same_count = np.where(side_arr == BUY, buy_count, sell_count)
            opp_count = np.where(side_arr == BUY, sell_count, buy_count)
            total_same = np.where(side_arr == BUY, buy_qty, sell_qty)
            total_opp = np.where(side_arr == BUY, sell_qty, buy_qty)
            X[:, col : col + 6] = np.column_stack([max_size, avg_size, same_count, opp_count, total_same, total_opp])
            col += 6

    for scale in pre.scales_ns:
        scale_seconds = scale / NS_PER_SECOND
        window_size = int(scale // state.interval_ns)
        for lag in range(3):
            start, end, valid = _window_bucket_bounds(idx, window_size, lag)
            lo, hi = _trade_bounds(state, start, end, valid)
            n_trades = hi - lo

            pair_count = np.maximum(n_trades - 2, 0)
            left = lo + 1
            right = hi - 1
            autocov = np.zeros(n_rows, dtype=np.float64)
            auto_mask = pair_count > 0
            if np.any(auto_mask):
                left_m = left[auto_mask]
                right_m = right[auto_mask]
                n_ret = n_trades[auto_mask] - 1
                sum_ret = pre.prefix_trade_ret[right_m + 1] - pre.prefix_trade_ret[left_m]
                mean_ret = sum_ret / np.maximum(n_ret, 1)
                cross = pre.prefix_trade_ret_cross[right_m + 1] - pre.prefix_trade_ret_cross[left_m + 1]
                sum_prev = pre.prefix_trade_ret[right_m] - pre.prefix_trade_ret[left_m]
                sum_next = pre.prefix_trade_ret[right_m + 1] - pre.prefix_trade_ret[left_m + 1]
                autocov[auto_mask] = (cross - mean_ret * (sum_prev + sum_next) + pair_count[auto_mask] * mean_ret * mean_ret) / np.maximum(
                    pair_count[auto_mask], 1
                )

            ret_sum = np.zeros(n_rows, dtype=np.float64)
            ret_mask = n_trades >= 2
            if np.any(ret_mask):
                first_px = state.trade_prices[lo[ret_mask]]
                last_px = state.trade_prices[hi[ret_mask] - 1]
                ret_sum[ret_mask] = sign[ret_mask] * (
                    np.log(np.maximum(last_px, 1e-300)) - np.log(np.maximum(first_px, 1e-300))
                )

            trade_intensity = np.where(valid, trade_intensity_count_per_second(n_trades, scale_seconds), 0.0).astype(np.float64)
            if np.any(ret_mask):
                first_ts = state.trade_times[lo[ret_mask]]
                last_ts = state.trade_times[hi[ret_mask] - 1]
                elapsed = (last_ts - first_ts) / NS_PER_SECOND
                trade_intensity[ret_mask] = trade_intensity_count_per_second(n_trades[ret_mask], scale_seconds, elapsed_seconds=elapsed)

            X[:, col : col + 3] = np.column_stack([autocov, ret_sum, trade_intensity])
            col += 3

    X[:, col] = np.where(side_arr == BUY, state.best_bid_notional[idx], state.best_ask_notional[idx])
    col += 1
    X[:, col] = np.where(side_arr == BUY, state.best_ask_notional[idx], state.best_bid_notional[idx])
    col += 1
    X[:, col] = np.where(side_arr == BUY, state.ob_bid_half_bp[idx], state.ob_ask_half_bp[idx])
    col += 1
    X[:, col] = np.where(side_arr == BUY, state.ob_ask_half_bp[idx], state.ob_bid_half_bp[idx])
    col += 1
    X[:, col] = state.totb_mean_s[idx]
    col += 1
    X[:, col] = state.age_s[idx]
    col += 1

    if col != BASE_FEATURE_COUNT:
        raise ValueError(f"Expected {BASE_FEATURE_COUNT} feature columns, got {col}.")

    if impute_missing:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32)


def grid_idx_from_ts(grid_times: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(grid_times, timestamps, side="right") - 1
    return np.clip(idx, 0, grid_times.shape[0] - 1).astype(np.int64)


def near_opp_notional(orders: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bid_notional = np.asarray(orders["post_best_bid_notional"], dtype=np.float64)
    ask_notional = np.asarray(orders["post_best_ask_notional"], dtype=np.float64)
    side = orders["side"]
    near = np.where(side == BUY, bid_notional, ask_notional)
    opp = np.where(side == BUY, ask_notional, bid_notional)
    return near.astype(np.float64), opp.astype(np.float64)


def label_orders(
    orders: np.ndarray,
    state: DayState,
    *,
    label_end_ts: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    filled = orders["status"] == FILLED
    side = orders["side"].astype(np.int8)
    post_ts = orders["post_local_ts"].astype(np.int64)
    fill_ts = np.where(filled, orders["fill_local_ts"], orders["terminal_local_ts"]).astype(np.int64)
    raw_fill_px = orders["fill_price"].astype(np.float64)
    post_px = orders["post_price"].astype(np.float64)
    fill_px = np.where(filled & np.isfinite(raw_fill_px) & (raw_fill_px > 0), raw_fill_px, post_px).astype(np.float64)

    next_change_ret = np.full(orders.shape[0], np.nan, dtype=np.float64)
    reversal = np.zeros(orders.shape[0], dtype=np.int8)
    signed_side = side.astype(np.float64)
    if state.mid_change_times.size > 0:
        next_idx = np.searchsorted(state.mid_change_times, fill_ts, side="right")
        next_change_ok = (
            filled
            & (next_idx >= 0)
            & (next_idx < state.mid_change_times.size)
            & np.isfinite(fill_px)
            & (fill_px > 0.0)
        )
        if label_end_ts is not None:
            next_change_ok &= state.mid_change_times[np.clip(next_idx, 0, state.mid_change_times.size - 1)] < int(label_end_ts)
        if np.any(next_change_ok):
            next_change_ret[next_change_ok] = compute_next_mid_change_return_bp(
                fill_ts[next_change_ok],
                fill_px[next_change_ok],
                signed_side[next_change_ok],
                state.mid_change_times,
                state.mid_change_best_bid,
                state.mid_change_best_ask,
                strict_after_fill=True,
                bp_scale=BP_SCALE,
            )
        use = filled & np.isfinite(next_change_ret)
        reversal[use] = (next_change_ret[use] > 0).astype(np.int8)

    mid_grid = np.where(
        np.isfinite(state.best_bid) & (state.best_bid > 0) & np.isfinite(state.best_ask) & (state.best_ask > 0),
        0.5 * (state.best_bid + state.best_ask),
        np.where(
            np.isfinite(state.best_bid) & (state.best_bid > 0),
            state.best_bid,
            np.where(np.isfinite(state.best_ask) & (state.best_ask > 0), state.best_ask, np.nan),
        ),
    )
    markout_5s = compute_post_fill_markout_bp(
        fill_ts,
        fill_px,
        side,
        state.grid_times,
        mid_grid,
        horizon_ns=5 * NS_PER_SECOND,
        lookup_side="left",
        bp_scale=BP_SCALE,
    )
    markout_5s = np.where(filled, markout_5s, np.nan)
    if label_end_ts is not None:
        markout_5s = np.where(fill_ts + (5 * NS_PER_SECOND) < int(label_end_ts), markout_5s, np.nan)
    post_idx = grid_idx_from_ts(state.grid_times, post_ts)
    label_resolved = (~filled) | np.isfinite(next_change_ret)
    return reversal, filled.astype(bool), markout_5s, next_change_ret, post_idx, fill_ts, fill_px, label_resolved


def build_dataset(
    orders: np.ndarray,
    state: DayState,
    feature_cfg,
    fill_surface,
    fill_prob_threshold: float,
    *,
    day_key: str,
    resolve_end_ts: int | None = None,
) -> Dataset:
    pre = build_precomputed_day(
        state,
        feature_cfg.scales_ns,
        feature_cfg.volatility_return_interval_ns,
        feature_cfg.volatility_lookbacks,
    )
    if resolve_end_ts is not None and orders.size > 0:
        orders = orders[orders["terminal_local_ts"].astype(np.int64) < int(resolve_end_ts)]
    near, opp = near_opp_notional(orders)
    fill_prob = fill_surface.predict(near, opp).astype(np.float64)
    reversal, filled, markout_5s, next_change_ret, post_idx, fill_ts, fill_px, label_resolved = label_orders(
        orders,
        state,
        label_end_ts=resolve_end_ts,
    )
    side = orders["side"].astype(np.int8)
    post_ts = orders["post_local_ts"].astype(np.int64)
    keep = (fill_prob >= float(fill_prob_threshold)) & label_resolved
    y = (filled & np.isfinite(next_change_ret) & (next_change_ret > 0)).astype(np.int8)
    X = feature_block(pre, post_idx[keep], side[keep], feature_cfg.impute_missing_with_zero)
    X = _append_calendar_features(X, day_key)
    return Dataset(
        X=X,
        y=y[keep],
        filled=filled[keep],
        side=side[keep],
        post_ts=post_ts[keep],
        post_idx=post_idx[keep],
        fill_ts=fill_ts[keep],
        fill_px=fill_px[keep],
        fill_prob=fill_prob[keep],
        markout_5s_bp=markout_5s[keep],
        next_change_ret_bp=next_change_ret[keep],
        feature_names=pre.feature_names,
    )


def merge_datasets(datasets: list[Dataset]) -> Dataset:
    feature_names = datasets[0].feature_names if datasets else []
    return Dataset(
        X=np.concatenate([d.X for d in datasets]) if datasets else np.empty((0, len(feature_names)), dtype=np.float32),
        y=np.concatenate([d.y for d in datasets]) if datasets else np.empty(0, dtype=np.int8),
        filled=np.concatenate([d.filled for d in datasets]) if datasets else np.empty(0, dtype=bool),
        side=np.concatenate([d.side for d in datasets]) if datasets else np.empty(0, dtype=np.int8),
        post_ts=np.concatenate([d.post_ts for d in datasets]) if datasets else np.empty(0, dtype=np.int64),
        post_idx=np.concatenate([d.post_idx for d in datasets]) if datasets else np.empty(0, dtype=np.int64),
        fill_ts=np.concatenate([d.fill_ts for d in datasets]) if datasets else np.empty(0, dtype=np.int64),
        fill_px=np.concatenate([d.fill_px for d in datasets]) if datasets else np.empty(0, dtype=np.float64),
        fill_prob=np.concatenate([d.fill_prob for d in datasets]) if datasets else np.empty(0, dtype=np.float64),
        markout_5s_bp=np.concatenate([d.markout_5s_bp for d in datasets]) if datasets else np.empty(0, dtype=np.float64),
        next_change_ret_bp=np.concatenate([d.next_change_ret_bp for d in datasets]) if datasets else np.empty(0, dtype=np.float64),
        feature_names=list(feature_names),
    )


def score_grid_signal(
    bundle: ModelBundle,
    state: DayState,
    orders: np.ndarray,
    feature_cfg,
    fill_prob_threshold: float,
    *,
    day_key: str,
    chunk_size: int = 50_000,
    grid_range: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    pre = build_precomputed_day(
        state,
        feature_cfg.scales_ns,
        feature_cfg.volatility_return_interval_ns,
        feature_cfg.volatility_lookbacks,
    )
    n_total = state.grid_times.shape[0]
    if grid_range is None:
        grid_start, grid_end = 0, n_total
    else:
        grid_start = int(grid_range[0])
        grid_end = int(grid_range[1])
        grid_start = max(0, min(grid_start, n_total))
        grid_end = max(grid_start, min(grid_end, n_total))
    n = grid_end - grid_start
    fill_prob_buy = bundle.fill_surface.predict(
        state.best_bid_notional[grid_start:grid_end],
        state.best_ask_notional[grid_start:grid_end],
    ).astype(np.float32)
    fill_prob_sell = bundle.fill_surface.predict(
        state.best_ask_notional[grid_start:grid_end],
        state.best_bid_notional[grid_start:grid_end],
    ).astype(np.float32)
    p_buy = np.full(n, -np.inf, dtype=np.float32)
    p_sell = np.full(n, -np.inf, dtype=np.float32)

    order_near, order_opp = near_opp_notional(orders)
    order_fill_prob = bundle.fill_surface.predict(order_near, order_opp).astype(np.float32)
    order_ts = orders["post_local_ts"].astype(np.int64)
    order_idx = grid_idx_from_ts(state.grid_times, order_ts)
    order_side = orders["side"].astype(np.int8)

    for side_value, out_prob in ((BUY, p_buy), (SELL, p_sell)):
        mask = (
            (order_side == side_value)
            & (order_fill_prob >= float(fill_prob_threshold))
            & (order_idx >= grid_start)
            & (order_idx < grid_end)
        )
        if not np.any(mask):
            continue
        idx_side = order_idx[mask]
        for start in range(0, idx_side.shape[0], chunk_size):
            end = min(start + chunk_size, idx_side.shape[0])
            idx_chunk = idx_side[start:end]
            X = feature_block(pre, idx_chunk, side_value, feature_cfg.impute_missing_with_zero)
            X = _append_calendar_features(X, day_key)
            prob_chunk = bundle.logistic.predict_proba(X)[:, 1].astype(np.float32)
            np.maximum.at(out_prob, idx_chunk - grid_start, prob_chunk)

    best_bid_qty = state.best_bid_qty[grid_start:grid_end]
    best_ask_qty = state.best_ask_qty[grid_start:grid_end]
    denom = best_bid_qty + best_ask_qty
    imbalance = np.divide(
        best_bid_qty - best_ask_qty,
        np.maximum(denom, 1e-300),
        out=np.zeros_like(best_bid_qty, dtype=np.float64),
        where=denom > 0,
    ).astype(np.float32)
    return {
        "ts": state.grid_times[grid_start:grid_end].astype(np.int64),
        "imbalance": imbalance,
        "fill_prob_buy": fill_prob_buy,
        "fill_prob_sell": fill_prob_sell,
        "p_buy": p_buy,
        "p_sell": p_sell,
        "best_bid": state.best_bid[grid_start:grid_end].astype(np.float32),
        "best_ask": state.best_ask[grid_start:grid_end].astype(np.float32),
    }
