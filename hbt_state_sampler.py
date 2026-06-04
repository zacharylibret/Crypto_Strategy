from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest, ROIVectorMarketDepthBacktest
from hftbacktest.data.utils.snapshot import create_last_snapshot


BUY_EVENT = np.uint64(536870912)
SELL_EVENT = np.uint64(268435456)
BUY = 1
SELL = -1
NS_PER_SECOND = 1_000_000_000
BP_SCALE = 10_000.0
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass(slots=True)
class DayState:
    interval_ns: int
    grid_times: np.ndarray
    best_bid: np.ndarray
    best_ask: np.ndarray
    best_bid_qty: np.ndarray
    best_ask_qty: np.ndarray
    best_bid_notional: np.ndarray
    best_ask_notional: np.ndarray
    ob_bid_half_bp: np.ndarray
    ob_ask_half_bp: np.ndarray
    age_s: np.ndarray
    totb_mean_s: np.ndarray
    last_trade_price: np.ndarray
    trade_hi: np.ndarray
    bucket_trade_price_max: np.ndarray
    bucket_trade_price_min: np.ndarray
    bucket_trade_max_size: np.ndarray
    trade_times: np.ndarray
    trade_prices: np.ndarray
    trade_qty: np.ndarray
    trade_side: np.ndarray
    mid_change_times: np.ndarray
    mid_change_best_bid: np.ndarray
    mid_change_best_ask: np.ndarray
    metadata: dict

    @property
    def n_buckets(self) -> int:
        return int(self.grid_times.shape[0])


@dataclass(slots=True)
class SnapshotChainStep:
    day_key: str
    data_path: Path
    snapshot_in: Path | None
    snapshot_out: Path


def infer_day_key(day_source: str | Path) -> str:
    stem = Path(day_source).stem
    match = DATE_RE.search(stem)
    if match is not None:
        return match.group(1)
    return stem[-5:]


def plan_snapshot_chain(
    day_sources: list[str | Path] | tuple[str | Path, ...],
    snapshot_dir: str | Path,
) -> list[SnapshotChainStep]:
    snapshot_dir = Path(snapshot_dir)
    ordered = sorted((Path(src) for src in day_sources), key=infer_day_key)
    steps: list[SnapshotChainStep] = []
    prev_snapshot: Path | None = None
    for src in ordered:
        day_key = infer_day_key(src)
        snapshot_out = snapshot_dir / f"snap_{day_key}.npz"
        steps.append(
            SnapshotChainStep(
                day_key=day_key,
                data_path=src,
                snapshot_in=prev_snapshot,
                snapshot_out=snapshot_out,
            )
        )
        prev_snapshot = snapshot_out
    return steps


def run_snapshot_chain(
    day_sources: list[str | Path] | tuple[str | Path, ...],
    snapshot_dir: str | Path,
    *,
    run_day: Callable[[SnapshotChainStep], Any],
) -> tuple[dict[str, Any], list[SnapshotChainStep]]:
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    finalized: list[SnapshotChainStep] = []
    prev_snapshot: Path | None = None
    for src in sorted((Path(day_source) for day_source in day_sources), key=infer_day_key):
        day_key = infer_day_key(src)
        planned = SnapshotChainStep(
            day_key=day_key,
            data_path=src,
            snapshot_in=prev_snapshot,
            snapshot_out=snapshot_dir / f"snap_{day_key}.npz",
        )
        result = run_day(planned)
        snapshot_out = planned.snapshot_out
        if isinstance(result, dict) and result.get("snapshot_path"):
            snapshot_out = Path(result["snapshot_path"])
        finalized_step = SnapshotChainStep(
            day_key=planned.day_key,
            data_path=planned.data_path,
            snapshot_in=planned.snapshot_in,
            snapshot_out=snapshot_out,
        )
        finalized.append(finalized_step)
        results[day_key] = result
        prev_snapshot = snapshot_out
    return results, finalized


def subset_start_snapshot(
    ordered_day_keys: list[str] | tuple[str, ...],
    subset_day_keys: list[str] | tuple[str, ...],
    snapshot_after_key: dict[str, str | Path],
) -> str | None:
    if not subset_day_keys:
        return None
    first_key = subset_day_keys[0]
    idx = list(ordered_day_keys).index(first_key)
    if idx == 0:
        return None
    prev_key = list(ordered_day_keys)[idx - 1]
    snapshot = snapshot_after_key[prev_key]
    return str(snapshot)


def create_hbt_eod_snapshot(
    snapshot_path: str | Path,
    data_path: str | Path,
    *,
    tick_size: float,
    lot_size: float,
    initial_snapshot_path: str | Path | None = None,
    force: bool = False,
) -> str:
    snapshot_path = Path(snapshot_path)
    if snapshot_path.exists() and not force:
        return str(snapshot_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    create_last_snapshot(
        [str(data_path)],
        tick_size=float(tick_size),
        lot_size=float(lot_size),
        initial_snapshot=str(initial_snapshot_path) if initial_snapshot_path else None,
        output_snapshot_filename=str(snapshot_path),
    )
    return str(snapshot_path)


def _coerce_data_files(data_path: str | Path | list[str | Path] | tuple[str | Path, ...]) -> list[str]:
    if isinstance(data_path, (str, Path)):
        return [str(data_path)]
    return [str(Path(p)) for p in data_path]


def build_hbt_asset(
    data_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    tick_size: float,
    lot_size: float,
    contract_size: float,
    maker_fee: float,
    taker_fee: float,
    entry_latency_ns: int,
    response_latency_ns: int,
    queue_model_power: int,
    last_trades_capacity: int,
    parallel_load: bool = True,
    initial_snapshot_path: str | Path | None = None,
) -> BacktestAsset:
    asset = BacktestAsset().data(_coerce_data_files(data_path))
    if initial_snapshot_path:
        asset = asset.initial_snapshot(str(initial_snapshot_path))
    asset = asset.linear_asset(float(contract_size))
    if hasattr(asset, "constant_order_latency"):
        asset = asset.constant_order_latency(int(entry_latency_ns), int(response_latency_ns))
    else:
        asset = asset.constant_latency(int(entry_latency_ns), int(response_latency_ns))
    asset = asset.power_prob_queue_model(int(queue_model_power))
    asset = asset.no_partial_fill_exchange()
    asset = asset.trading_value_fee_model(float(maker_fee), float(taker_fee))
    asset = asset.tick_size(float(tick_size)).lot_size(float(lot_size))
    if hasattr(asset, "last_trades_capacity"):
        asset = asset.last_trades_capacity(int(last_trades_capacity))
    if hasattr(asset, "parallel_load"):
        asset = asset.parallel_load(bool(parallel_load))
    return asset


def build_hft_backtester(
    data_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    engine: str,
    roi_lb: float,
    roi_ub: float,
    tick_size: float,
    lot_size: float,
    contract_size: float,
    maker_fee: float,
    taker_fee: float,
    entry_latency_ns: int,
    response_latency_ns: int,
    queue_model_power: int,
    last_trades_capacity: int,
    parallel_load: bool = True,
    initial_snapshot_path: str | Path | None = None,
):
    asset = build_hbt_asset(
        data_path,
        tick_size=tick_size,
        lot_size=lot_size,
        contract_size=contract_size,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        entry_latency_ns=entry_latency_ns,
        response_latency_ns=response_latency_ns,
        queue_model_power=queue_model_power,
        last_trades_capacity=last_trades_capacity,
        parallel_load=parallel_load,
        initial_snapshot_path=initial_snapshot_path,
    )
    engine_name = str(engine).lower()
    if engine_name == "roi_vector":
        asset = asset.roi_lb(float(roi_lb)).roi_ub(float(roi_ub))
        return ROIVectorMarketDepthBacktest([asset]), "ROIVectorMarketDepthBacktest"
    if engine_name == "hashmap":
        return HashMapMarketDepthBacktest([asset]), "HashMapMarketDepthBacktest"
    raise ValueError(f"Unsupported hftbacktest engine: {engine}")


def extract_midprice_changes_hbt(
    data_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    engine: str,
    roi_lb: float,
    roi_ub: float,
    tick_size: float,
    lot_size: float,
    contract_size: float,
    maker_fee: float,
    taker_fee: float,
    entry_latency_ns: int,
    response_latency_ns: int,
    queue_model_power: int,
    parallel_load: bool = True,
    initial_snapshot_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hbt, _ = build_hft_backtester(
        data_path,
        engine=engine,
        roi_lb=roi_lb,
        roi_ub=roi_ub,
        tick_size=tick_size,
        lot_size=lot_size,
        contract_size=contract_size,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        entry_latency_ns=entry_latency_ns,
        response_latency_ns=response_latency_ns,
        queue_model_power=queue_model_power,
        last_trades_capacity=0,
        parallel_load=parallel_load,
        initial_snapshot_path=initial_snapshot_path,
    )
    mid_change_times: list[int] = []
    mid_change_best_bid: list[float] = []
    mid_change_best_ask: list[float] = []
    prev_bid_tick: int | None = None
    prev_ask_tick: int | None = None
    try:
        while True:
            status = int(hbt.wait_next_feed(False, 10**18))
            if status == 1:
                break
            if status != 2:
                continue
            ts = int(hbt.current_timestamp)
            depth = hbt.depth(0)
            bid_tick = int(depth.best_bid_tick)
            ask_tick = int(depth.best_ask_tick)
            if prev_bid_tick is None or bid_tick != prev_bid_tick or ask_tick != prev_ask_tick:
                mid_change_times.append(ts)
                mid_change_best_bid.append(float(depth.best_bid) if np.isfinite(depth.best_bid) else np.nan)
                mid_change_best_ask.append(float(depth.best_ask) if np.isfinite(depth.best_ask) else np.nan)
                prev_bid_tick = bid_tick
                prev_ask_tick = ask_tick
    finally:
        hbt.close()
    return (
        np.asarray(mid_change_times, dtype=np.int64),
        np.asarray(mid_change_best_bid, dtype=np.float64),
        np.asarray(mid_change_best_ask, dtype=np.float64),
    )


def compute_exact_touch_time_features(
    grid_times: np.ndarray,
    mid_change_times: np.ndarray,
    *,
    lookback_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Flatten defensively so cached / loaded arrays cannot introduce column-vector
    # shape mismatches during boolean indexing.
    grid_times = np.asarray(grid_times, dtype=np.int64).reshape(-1)
    mid_change_times = np.asarray(mid_change_times, dtype=np.int64).reshape(-1)
    age_s = np.zeros(grid_times.shape[0], dtype=np.float64)
    totb_mean_s = np.zeros(grid_times.shape[0], dtype=np.float64)
    if grid_times.size == 0:
        return age_s, totb_mean_s
    if mid_change_times.size == 0:
        return age_s, totb_mean_s

    last_change_idx = np.searchsorted(mid_change_times, grid_times, side="right") - 1
    valid_last_change = last_change_idx >= 0
    if np.any(valid_last_change):
        age_s[valid_last_change] = (
            (grid_times[valid_last_change] - mid_change_times[last_change_idx[valid_last_change]]) / NS_PER_SECOND
        )

    if mid_change_times.size < 2:
        totb_mean_s[:] = age_s
        return age_s, totb_mean_s

    survival_end_times = mid_change_times[1:]
    survival_durations = np.diff(mid_change_times).astype(np.float64) / NS_PER_SECOND
    cum_survival = np.concatenate(([0.0], np.cumsum(survival_durations, dtype=np.float64)))

    left = np.searchsorted(survival_end_times, grid_times - int(lookback_ns), side="left")
    right = np.searchsorted(survival_end_times, grid_times, side="right")
    counts = np.asarray(right - left, dtype=np.int64).reshape(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.asarray(
            (cum_survival[right] - cum_survival[left]) / np.maximum(counts, 1),
            dtype=np.float64,
        ).reshape(-1)
    use_mean = counts > 0
    totb_mean_s = np.where(use_mean, means, age_s)
    return age_s, totb_mean_s


def _touch_qty(depth, *, is_bid: bool) -> float:
    price = float(depth.best_bid if is_bid else depth.best_ask)
    if not np.isfinite(price) or price <= 0:
        return 0.0
    tick = int(depth.best_bid_tick if is_bid else depth.best_ask_tick)
    qty = depth.bid_qty_at_tick(tick) if is_bid else depth.ask_qty_at_tick(tick)
    qty = float(qty)
    return qty if np.isfinite(qty) and qty > 0 else 0.0


def _half_depth_bp(depth, *, target_notional_usd: float, is_bid: bool) -> float:
    if target_notional_usd <= 0:
        return 0.0
    best_price = float(depth.best_bid if is_bid else depth.best_ask)
    if not np.isfinite(best_price) or best_price <= 0:
        return 0.0
    best_tick = int(depth.best_bid_tick if is_bid else depth.best_ask_tick)
    step = -1 if is_bid else 1
    remaining = float(target_notional_usd)
    qty_taken = 0.0

    if hasattr(depth, "bid_depth") and hasattr(depth, "ask_depth"):
        depth_arr = depth.bid_depth if is_bid else depth.ask_depth
        tick_lo = int(depth.roi_lb_tick)
        tick_hi = int(depth.roi_ub_tick)

        def level_qty(tick: int) -> float:
            if tick < tick_lo or tick > tick_hi:
                return 0.0
            qty = float(depth_arr[tick - tick_lo])
            return qty if np.isfinite(qty) and qty > 0 else 0.0

    else:
        snapshot = depth.snapshot()
        if snapshot.size == 0:
            return 0.0
        px = snapshot["px"].astype(np.float64)
        tick_lo = int(np.floor(np.nanmin(px) / float(depth.tick_size)))
        tick_hi = int(np.ceil(np.nanmax(px) / float(depth.tick_size)))

        def level_qty(tick: int) -> float:
            qty = depth.bid_qty_at_tick(tick) if is_bid else depth.ask_qty_at_tick(tick)
            qty = float(qty)
            return qty if np.isfinite(qty) and qty > 0 else 0.0

    tick = best_tick
    while remaining > 1e-12:
        if tick < tick_lo or tick > tick_hi:
            break
        level_price = tick * float(depth.tick_size)
        level_qty_value = level_qty(tick)
        if level_qty_value > 0 and np.isfinite(level_price) and level_price > 0:
            available = level_price * level_qty_value
            used = min(remaining, available)
            qty_taken += used / level_price
            remaining -= used
        tick += step

    if qty_taken <= 0:
        return 0.0
    executed_value = target_notional_usd - remaining
    vwap = executed_value / qty_taken
    if is_bid:
        return max(0.0, (best_price - vwap) / best_price * BP_SCALE)
    return max(0.0, (vwap - best_price) / best_price * BP_SCALE)


def build_day_state_hbt(
    data_path: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    tick_size: float,
    lot_size: float,
    contract_size: float,
    maker_fee: float,
    taker_fee: float,
    interval_ns: int,
    roi_lb: float,
    roi_ub: float,
    entry_latency_ns: int,
    response_latency_ns: int,
    queue_model_power: int,
    last_trades_capacity: int,
    parallel_load: bool = True,
    half_book_notional_usd: float,
    totb_mean_lookback_ns: int,
    engine: str = "roi_vector",
    initial_snapshot_path: str | Path | None = None,
) -> DayState:
    hbt, engine_name = build_hft_backtester(
        data_path,
        engine=engine,
        roi_lb=roi_lb,
        roi_ub=roi_ub,
        tick_size=tick_size,
        lot_size=lot_size,
        contract_size=contract_size,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        entry_latency_ns=entry_latency_ns,
        response_latency_ns=response_latency_ns,
        queue_model_power=queue_model_power,
        last_trades_capacity=last_trades_capacity,
        parallel_load=parallel_load,
        initial_snapshot_path=initial_snapshot_path,
    )

    grid_times: list[int] = []
    best_bid: list[float] = []
    best_ask: list[float] = []
    best_bid_qty: list[float] = []
    best_ask_qty: list[float] = []
    best_bid_notional: list[float] = []
    best_ask_notional: list[float] = []
    ob_bid_half_bp: list[float] = []
    ob_ask_half_bp: list[float] = []
    age_s: list[float] = []
    totb_mean_s: list[float] = []
    last_trade_price: list[float] = []
    trade_hi: list[int] = []
    bucket_trade_price_max: list[float] = []
    bucket_trade_price_min: list[float] = []
    bucket_trade_max_size: list[float] = []
    trade_times: list[int] = []
    trade_prices: list[float] = []
    trade_qty: list[float] = []
    trade_side: list[int] = []
    mid_change_times: list[int] = []
    mid_change_best_bid: list[float] = []
    mid_change_best_ask: list[float] = []

    prev_bid_tick: int | None = None
    prev_ask_tick: int | None = None
    latest_trade_price = np.nan
    trade_count = 0

    try:
        while int(hbt.elapse(int(interval_ns))) == 0:
            ts = int(hbt.current_timestamp)
            depth = hbt.depth(0)
            trades = np.asarray(hbt.last_trades(0)).copy()

            if trades.size > 0:
                trade_times.extend(trades["local_ts"].astype(np.int64).tolist())
                trade_prices.extend(trades["px"].astype(np.float64).tolist())
                trade_qty.extend(trades["qty"].astype(np.float64).tolist())
                trade_side.extend(
                    np.where((trades["ev"] & BUY_EVENT) == BUY_EVENT, BUY, SELL).astype(np.int8).tolist()
                )
                trade_count += int(trades.shape[0])
                latest_trade_price = float(trades["px"][-1])
                bucket_trade_price_max.append(float(np.max(trades["px"])))
                bucket_trade_price_min.append(float(np.min(trades["px"])))
                bucket_trade_max_size.append(float(np.max(trades["qty"])))
            else:
                bucket_trade_price_max.append(-np.inf)
                bucket_trade_price_min.append(np.inf)
                bucket_trade_max_size.append(0.0)
            if hasattr(hbt, "clear_last_trades"):
                hbt.clear_last_trades(0)

            bid_price = float(depth.best_bid)
            ask_price = float(depth.best_ask)
            bid_tick = int(depth.best_bid_tick)
            ask_tick = int(depth.best_ask_tick)
            bid_qty = _touch_qty(depth, is_bid=True)
            ask_qty = _touch_qty(depth, is_bid=False)

            touch_changed = prev_bid_tick is None or bid_tick != prev_bid_tick or ask_tick != prev_ask_tick
            if touch_changed:
                prev_bid_tick = bid_tick
                prev_ask_tick = ask_tick

            if not np.isfinite(latest_trade_price):
                if np.isfinite(bid_price) and np.isfinite(ask_price):
                    latest_trade_price = 0.5 * (bid_price + ask_price)
                elif np.isfinite(bid_price):
                    latest_trade_price = bid_price
                elif np.isfinite(ask_price):
                    latest_trade_price = ask_price
                else:
                    latest_trade_price = 0.0

            grid_times.append(ts)
            best_bid.append(bid_price if np.isfinite(bid_price) else np.nan)
            best_ask.append(ask_price if np.isfinite(ask_price) else np.nan)
            best_bid_qty.append(bid_qty)
            best_ask_qty.append(ask_qty)
            best_bid_notional.append(0.0 if not np.isfinite(bid_price) else bid_price * bid_qty)
            best_ask_notional.append(0.0 if not np.isfinite(ask_price) else ask_price * ask_qty)
            ob_bid_half_bp.append(_half_depth_bp(depth, target_notional_usd=half_book_notional_usd, is_bid=True))
            ob_ask_half_bp.append(_half_depth_bp(depth, target_notional_usd=half_book_notional_usd, is_bid=False))
            age_s.append(0.0)
            totb_mean_s.append(0.0)
            last_trade_price.append(float(latest_trade_price))
            trade_hi.append(trade_count)
    finally:
        hbt.close()

    metadata = {
        "state_source": "hftbacktest",
        "engine": engine_name,
        "initial_snapshot": Path(initial_snapshot_path).name if initial_snapshot_path else "",
        "touch_change_mode": "event_feed_exact",
        "state_version": f"hft_state_interval_{int(interval_ns)}",
        "n_grid": int(len(grid_times)),
        "n_trades": int(len(trade_times)),
        "roi_lb": float(roi_lb),
        "roi_ub": float(roi_ub),
        "last_trades_capacity": int(last_trades_capacity),
    }

    mid_change_times_arr, mid_change_best_bid_arr, mid_change_best_ask_arr = extract_midprice_changes_hbt(
        data_path,
        engine=engine,
        roi_lb=roi_lb,
        roi_ub=roi_ub,
        tick_size=tick_size,
        lot_size=lot_size,
        contract_size=contract_size,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        entry_latency_ns=entry_latency_ns,
        response_latency_ns=response_latency_ns,
        queue_model_power=queue_model_power,
        parallel_load=parallel_load,
        initial_snapshot_path=initial_snapshot_path,
    )
    age_s_arr, totb_mean_s_arr = compute_exact_touch_time_features(
        np.asarray(grid_times, dtype=np.int64),
        mid_change_times_arr,
        lookback_ns=int(totb_mean_lookback_ns),
    )

    return DayState(
        interval_ns=int(interval_ns),
        grid_times=np.asarray(grid_times, dtype=np.int64),
        best_bid=np.asarray(best_bid, dtype=np.float64),
        best_ask=np.asarray(best_ask, dtype=np.float64),
        best_bid_qty=np.asarray(best_bid_qty, dtype=np.float64),
        best_ask_qty=np.asarray(best_ask_qty, dtype=np.float64),
        best_bid_notional=np.asarray(best_bid_notional, dtype=np.float64),
        best_ask_notional=np.asarray(best_ask_notional, dtype=np.float64),
        ob_bid_half_bp=np.asarray(ob_bid_half_bp, dtype=np.float64),
        ob_ask_half_bp=np.asarray(ob_ask_half_bp, dtype=np.float64),
        age_s=age_s_arr,
        totb_mean_s=totb_mean_s_arr,
        last_trade_price=np.asarray(last_trade_price, dtype=np.float64),
        trade_hi=np.asarray(trade_hi, dtype=np.int64),
        bucket_trade_price_max=np.asarray(bucket_trade_price_max, dtype=np.float64),
        bucket_trade_price_min=np.asarray(bucket_trade_price_min, dtype=np.float64),
        bucket_trade_max_size=np.asarray(bucket_trade_max_size, dtype=np.float64),
        trade_times=np.asarray(trade_times, dtype=np.int64),
        trade_prices=np.asarray(trade_prices, dtype=np.float64),
        trade_qty=np.asarray(trade_qty, dtype=np.float64),
        trade_side=np.asarray(trade_side, dtype=np.int8),
        mid_change_times=mid_change_times_arr,
        mid_change_best_bid=mid_change_best_bid_arr,
        mid_change_best_ask=mid_change_best_ask_arr,
        metadata=metadata,
    )


def save_day_state(path: str | Path, state: DayState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        interval_ns=np.array([state.interval_ns], dtype=np.int64),
        grid_times=state.grid_times,
        best_bid=state.best_bid,
        best_ask=state.best_ask,
        best_bid_qty=state.best_bid_qty,
        best_ask_qty=state.best_ask_qty,
        best_bid_notional=state.best_bid_notional,
        best_ask_notional=state.best_ask_notional,
        ob_bid_half_bp=state.ob_bid_half_bp,
        ob_ask_half_bp=state.ob_ask_half_bp,
        age_s=state.age_s,
        totb_mean_s=state.totb_mean_s,
        last_trade_price=state.last_trade_price,
        trade_hi=state.trade_hi,
        bucket_trade_price_max=state.bucket_trade_price_max,
        bucket_trade_price_min=state.bucket_trade_price_min,
        bucket_trade_max_size=state.bucket_trade_max_size,
        trade_times=state.trade_times,
        trade_prices=state.trade_prices,
        trade_qty=state.trade_qty,
        trade_side=state.trade_side,
        mid_change_times=state.mid_change_times,
        mid_change_best_bid=state.mid_change_best_bid,
        mid_change_best_ask=state.mid_change_best_ask,
        metadata=np.array([state.metadata], dtype=object),
    )


def load_day_state(path: str | Path) -> DayState:
    with np.load(path, allow_pickle=True) as zf:
        return DayState(
            interval_ns=int(zf["interval_ns"][0]),
            grid_times=zf["grid_times"].copy(),
            best_bid=zf["best_bid"].copy(),
            best_ask=zf["best_ask"].copy(),
            best_bid_qty=zf["best_bid_qty"].copy(),
            best_ask_qty=zf["best_ask_qty"].copy(),
            best_bid_notional=zf["best_bid_notional"].copy(),
            best_ask_notional=zf["best_ask_notional"].copy(),
            ob_bid_half_bp=zf["ob_bid_half_bp"].copy(),
            ob_ask_half_bp=zf["ob_ask_half_bp"].copy(),
            age_s=zf["age_s"].copy(),
            totb_mean_s=zf["totb_mean_s"].copy(),
            last_trade_price=zf["last_trade_price"].copy(),
            trade_hi=zf["trade_hi"].copy(),
            bucket_trade_price_max=zf["bucket_trade_price_max"].copy(),
            bucket_trade_price_min=zf["bucket_trade_price_min"].copy(),
            bucket_trade_max_size=zf["bucket_trade_max_size"].copy(),
            trade_times=zf["trade_times"].copy(),
            trade_prices=zf["trade_prices"].copy(),
            trade_qty=zf["trade_qty"].copy(),
            trade_side=zf["trade_side"].copy(),
            mid_change_times=zf["mid_change_times"].copy(),
            mid_change_best_bid=zf["mid_change_best_bid"].copy(),
            mid_change_best_ask=zf["mid_change_best_ask"].copy(),
            metadata=dict(zf["metadata"][0]),
        )
