from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from hftbacktest import Recorder
from hftbacktest.recorder import record_dtype as HBT_RECORD_DTYPE
from hftbacktest.stats import (
    AnnualRet,
    DailyNumberOfTrades,
    DailyTradingValue,
    LinearAssetRecord,
    MaxDrawdown,
    MaxPositionValue,
    Ret,
    ReturnOverMDD,
    SR,
    Sortino,
)

from hbt_state_sampler import build_hft_backtester, infer_day_key
from paper_workflow import combine_signal_days
from reversal_mm_constants import (
    ACTIVE_STATUS,
    BP_SCALE,
    BUY,
    FILLED,
    GTX,
    HBT_STAT_NAMES,
    INACTIVE_STATUS,
    LIMIT,
    NS_PER_SECOND,
    PARTIALLY_FILLED,
    ROUNDTRIP_DTYPE,
    SAMPLED_ORDER_DTYPE,
    SELL,
)


@dataclass(slots=True)
class PostedOrder:
    order_id: int
    side: int
    post_local_ts: int
    post_price: float
    post_best_bid: float
    post_best_ask: float
    post_best_bid_qty: float
    post_best_ask_qty: float


@dataclass(slots=True)
class OpenPosition:
    side: int = 0
    open_price: float = 0.0
    open_local_ts: int = 0


@dataclass(slots=True)
class BacktestRun:
    roundtrips: np.ndarray
    record: np.ndarray


@dataclass(slots=True)
class StrategyEvaluation:
    roundtrips: np.ndarray
    record: np.ndarray
    stats_obj: Any | None
    stats_table: pd.DataFrame
    stats_summary: dict[str, float]


def _build_hbt(data_files, market_cfg, hbt_cfg, snapshot_path=None, *, trade_buffer: bool = False):
    hbt, _ = build_hft_backtester(
        list(data_files),
        engine=hbt_cfg.engine,
        roi_lb=hbt_cfg.roi_lb,
        roi_ub=hbt_cfg.roi_ub,
        tick_size=market_cfg.tick_size,
        lot_size=market_cfg.lot_size,
        contract_size=market_cfg.contract_size,
        maker_fee=market_cfg.maker_fee,
        taker_fee=market_cfg.taker_fee,
        entry_latency_ns=hbt_cfg.entry_latency_ns,
        response_latency_ns=hbt_cfg.response_latency_ns,
        queue_model_power=hbt_cfg.queue_model_power,
        last_trades_capacity=hbt_cfg.last_trades_capacity if trade_buffer else 0,
        parallel_load=getattr(hbt_cfg, "parallel_load", True),
        initial_snapshot_path=snapshot_path,
    )
    return hbt


def _coerce_data_files(data_path: str | Path | Iterable[str | Path]) -> list[str]:
    if isinstance(data_path, (str, Path)):
        return [str(Path(data_path))]
    return [str(Path(p)) for p in data_path]


def _is_inactive(status: int) -> bool:
    return int(status) in INACTIVE_STATUS


def _is_active(status: int) -> bool:
    return int(status) in ACTIVE_STATUS


def _signal_index(sig_ts: np.ndarray, cur_ts: int, last: int) -> int:
    if sig_ts.size == 0:
        return -1
    if last < 0:
        idx = np.searchsorted(sig_ts, cur_ts, side="right") - 1
        return int(np.clip(idx, -1, sig_ts.size - 1))
    while last + 1 < sig_ts.size and int(sig_ts[last + 1]) <= cur_ts:
        last += 1
    return last


def _reversal_score(side: int, p_buy: float, p_sell: float, threshold: float) -> bool:
    return p_buy > threshold if side == BUY else p_sell > threshold


def _fill_prob_gate(side: int, fill_prob_buy: float, fill_prob_sell: float, threshold: float) -> bool:
    return fill_prob_buy >= threshold if side == BUY else fill_prob_sell >= threshold


def _post_price(side: int, depth) -> float:
    return float(depth.best_bid if side == BUY else depth.best_ask)


def _submit_maker(hbt, asset_no: int, order_id: int, side: int, price: float, qty: float) -> int:
    if side == BUY:
        return int(hbt.submit_buy_order(asset_no, order_id, price, qty, GTX, LIMIT, False))
    return int(hbt.submit_sell_order(asset_no, order_id, price, qty, GTX, LIMIT, False))


def _touch_qty(depth, side: int) -> float:
    price = float(depth.best_bid if side == BUY else depth.best_ask)
    if not np.isfinite(price) or price <= 0:
        return 0.0
    tick = int(depth.best_bid_tick if side == BUY else depth.best_ask_tick)
    qty = depth.bid_qty_at_tick(tick) if side == BUY else depth.ask_qty_at_tick(tick)
    qty = float(qty)
    return qty if np.isfinite(qty) and qty > 0 else 0.0


def _notional(price: float, qty: float) -> float:
    return float(price * qty) if np.isfinite(price) and np.isfinite(qty) else 0.0


def _reversal_adverse(side: int, imbalance: float, threshold: float = 0.2) -> bool:
    return imbalance < -threshold if side == BUY else imbalance > threshold


def _reversal_cancel(side: int, imbalance: float, threshold: float = 0.0) -> bool:
    return imbalance >= -threshold if side == BUY else imbalance <= threshold


def _quote_sides_for_step(inventory: int, signal: dict[str, float], allow_dual_quote_same_step: bool, *, current_ts: int = 0) -> list[int]:
    if inventory > 0:
        return [SELL]
    if inventory < 0:
        return [BUY]
    if allow_dual_quote_same_step:
        return [BUY, SELL]
    buy_score = float(signal.get("p_buy", -np.inf))
    sell_score = float(signal.get("p_sell", -np.inf))
    if np.isfinite(buy_score) and np.isfinite(sell_score):
        if buy_score > sell_score:
            return [BUY, SELL]
        if sell_score > buy_score:
            return [SELL, BUY]
        return [BUY, SELL] if ((int(current_ts) // NS_PER_SECOND) % 2 == 0) else [SELL, BUY]
    return [BUY, SELL]


def _should_quote(mode: str, side: int, inventory: int, signal: dict[str, float], strategy_cfg, model_fill_prob_threshold: float) -> bool:
    inventory_side = BUY if inventory > 0 else SELL if inventory < 0 else 0
    if inventory_side != 0 and side == inventory_side:
        return False
    gate = (not strategy_cfg.enable_fill_prob_gate) or _fill_prob_gate(
        side,
        signal["fill_prob_buy"],
        signal["fill_prob_sell"],
        model_fill_prob_threshold,
    )
    if not gate:
        return False
    if mode != "reversal_logistic":
        raise ValueError(f"Unsupported mode: {mode}")
    return _reversal_adverse(side, signal["imbalance"], float(strategy_cfg.imbalance_post_threshold)) and _reversal_score(
        side,
        signal["p_buy"],
        signal["p_sell"],
        strategy_cfg.model_threshold,
    )


def _tracked_order_ids(
    working: dict[int, int | None],
    cancel_pending: dict[int, list[int]],
    side: int,
) -> list[int]:
    ids: list[int] = []
    working_id = working[side]
    if working_id is not None:
        ids.append(int(working_id))
    ids.extend(int(x) for x in cancel_pending[side])
    return ids


def _drop_tracked_order(
    working: dict[int, int | None],
    cancel_pending: dict[int, list[int]],
    side: int,
    order_id: int,
) -> None:
    if working[side] == order_id:
        working[side] = None
        return
    pending = cancel_pending[side]
    for idx, pending_id in enumerate(pending):
        if pending_id == order_id:
            pending.pop(idx)
            return


def _move_working_to_cancel_pending(
    working: dict[int, int | None],
    cancel_pending: dict[int, list[int]],
    side: int,
) -> int | None:
    order_id = working[side]
    if order_id is None:
        return None
    cancel_pending[side].append(int(order_id))
    working[side] = None
    return int(order_id)


def _process_fills(
    events: list[tuple[int, int, int, float]],
    inventory: int,
    open_positions: deque[OpenPosition],
    fee_bp: float,
    roundtrips: list[tuple],
) -> tuple[int, deque[OpenPosition]]:
    for _, fill_local_ts, side, price in events:
        inventory_side = BUY if inventory > 0 else SELL if inventory < 0 else 0
        if inventory_side == 0 or inventory_side == side:
            inventory += side
            open_positions.append(OpenPosition(side=side, open_price=price, open_local_ts=fill_local_ts))
            continue

        open_pos = open_positions.popleft()
        gross_ret_bp = open_pos.side * ((price / max(open_pos.open_price, 1e-300)) - 1.0) * BP_SCALE
        net_ret_bp = gross_ret_bp - 2.0 * fee_bp
        holding_s = (fill_local_ts - open_pos.open_local_ts) / NS_PER_SECOND
        roundtrips.append(
            (
                int(open_pos.side),
                int(open_pos.open_local_ts),
                float(open_pos.open_price),
                int(fill_local_ts),
                float(price),
                float(gross_ret_bp),
                float(net_ret_bp),
                float(holding_s),
            )
        )
        inventory += side
    return inventory, open_positions


def run_sampler(data_path, market_cfg, hbt_cfg, order_qty: float, snapshot_path=None, mdl_ns: int = 0) -> np.ndarray:
    hbt = _build_hbt(_coerce_data_files(data_path), market_cfg, hbt_cfg, snapshot_path, trade_buffer=False)
    asset_no = 0
    working: dict[int, int | None] = {BUY: None, SELL: None}
    cancel_pending: dict[int, list[int]] = {BUY: [], SELL: []}
    posted: dict[int, PostedOrder] = {}
    rows: list[tuple] = []
    order_id = 1
    try:
        while int(hbt.elapse(int(hbt_cfg.control_interval_ns))) == 0:
            current_ts = int(hbt.current_timestamp)
            depth = hbt.depth(asset_no)
            orders = hbt.orders(asset_no)
            for side in (BUY, SELL):
                for tracked_id in list(_tracked_order_ids(working, cancel_pending, side)):
                    order = orders.get(tracked_id)
                    if order is None:
                        _drop_tracked_order(working, cancel_pending, side, tracked_id)
                        posted.pop(tracked_id, None)
                        continue
                    status = int(order.status)
                    if _is_inactive(status):
                        meta = posted.pop(tracked_id)
                        exch_ts = int(getattr(order, "exch_timestamp", 0))
                        terminal_local_ts = int(exch_ts + mdl_ns) if exch_ts > 0 else current_ts
                        fill_price = float(order.exec_price) if status in (FILLED, PARTIALLY_FILLED) else np.nan
                        fill_local_ts = terminal_local_ts if np.isfinite(fill_price) else -1
                        rows.append(
                            (
                                int(meta.order_id),
                                int(meta.side),
                                int(meta.post_local_ts),
                                float(meta.post_price),
                                float(meta.post_best_bid),
                                float(meta.post_best_ask),
                                float(meta.post_best_bid_qty),
                                float(meta.post_best_ask_qty),
                                _notional(meta.post_best_bid, meta.post_best_bid_qty),
                                _notional(meta.post_best_ask, meta.post_best_ask_qty),
                                int(terminal_local_ts),
                                int(status),
                                int(fill_local_ts),
                                float(fill_price),
                            )
                        )
                        _drop_tracked_order(working, cancel_pending, side, tracked_id)
            for side in (BUY, SELL):
                active_id = working[side]
                if active_id is None:
                    continue
                order = orders.get(active_id)
                if order is None or not _is_active(int(order.status)) or not bool(order.cancellable):
                    continue
                stale = (
                    (side == BUY and float(depth.best_bid) > float(order.price))
                    or (side == SELL and float(depth.best_ask) < float(order.price))
                )
                if stale:
                    hbt.cancel(asset_no, active_id, False)
                    _move_working_to_cancel_pending(working, cancel_pending, side)

            budget = 2 if hbt_cfg.allow_dual_quote_same_step else 1
            for side in _quote_sides_for_step(
                0,
                {"p_buy": 0.0, "p_sell": 0.0},
                bool(hbt_cfg.allow_dual_quote_same_step),
                current_ts=current_ts,
            ):
                if budget <= 0:
                    break
                if working[side] is not None:
                    continue
                price = _post_price(side, depth)
                if not np.isfinite(price) or price <= 0:
                    continue
                rc = _submit_maker(hbt, asset_no, order_id, side, price, order_qty)
                if rc == 0:
                    working[side] = order_id
                    posted[order_id] = PostedOrder(
                        order_id=order_id,
                        side=side,
                        post_local_ts=current_ts,
                        post_price=price,
                        post_best_bid=float(depth.best_bid),
                        post_best_ask=float(depth.best_ask),
                        post_best_bid_qty=_touch_qty(depth, BUY),
                        post_best_ask_qty=_touch_qty(depth, SELL),
                    )
                    order_id += 1
                    budget -= 1
            hbt.clear_inactive_orders(asset_no)
    finally:
        hbt.close()
    return np.asarray(rows, dtype=SAMPLED_ORDER_DTYPE) if rows else np.empty(0, dtype=SAMPLED_ORDER_DTYPE)


def run_backtest(
    mode: str,
    data_files,
    signal_grid: dict[str, np.ndarray],
    market_cfg,
    hbt_cfg,
    strategy_cfg,
    model_fill_prob_threshold: float,
    snapshot_path=None,
    *,
    return_record: bool = False,
) -> BacktestRun | np.ndarray:
    hbt = _build_hbt(data_files, market_cfg, hbt_cfg, snapshot_path, trade_buffer=False)
    asset_no = 0
    working: dict[int, int | None] = {BUY: None, SELL: None}
    cancel_pending: dict[int, list[int]] = {BUY: [], SELL: []}
    order_id = 1
    inventory = 0
    open_positions: deque[OpenPosition] = deque()
    roundtrips: list[tuple] = []
    recorder = Recorder(1, max(int(signal_grid["ts"].shape[0]) + 4096, 8192)) if return_record else None

    sig_ts = signal_grid["ts"].astype(np.int64)
    signal_idx = -1
    fee_bp = market_cfg.maker_fee * BP_SCALE
    try:
        while int(hbt.elapse(int(hbt_cfg.control_interval_ns))) == 0:
            current_ts = int(hbt.current_timestamp)
            depth = hbt.depth(asset_no)
            orders = hbt.orders(asset_no)
            signal_idx = _signal_index(sig_ts, current_ts, signal_idx)
            if signal_idx < 0:
                signal = {
                    "imbalance": 0.0,
                    "fill_prob_buy": 0.0,
                    "fill_prob_sell": 0.0,
                    "p_buy": -np.inf,
                    "p_sell": -np.inf,
                }
            else:
                signal = {
                    "imbalance": float(signal_grid["imbalance"][signal_idx]),
                    "fill_prob_buy": float(signal_grid["fill_prob_buy"][signal_idx]),
                    "fill_prob_sell": float(signal_grid["fill_prob_sell"][signal_idx]),
                    "p_buy": float(signal_grid["p_buy"][signal_idx]),
                    "p_sell": float(signal_grid["p_sell"][signal_idx]),
                }

            fills: list[tuple[int, int, int, float]] = []
            for side in (BUY, SELL):
                for tracked_id in list(_tracked_order_ids(working, cancel_pending, side)):
                    order = orders.get(tracked_id)
                    if order is None:
                        _drop_tracked_order(working, cancel_pending, side, tracked_id)
                        continue
                    status = int(order.status)
                    if _is_inactive(status):
                        if status in (FILLED, PARTIALLY_FILLED):
                            fills.append(
                                (
                                    int(getattr(order, "exch_timestamp", 0)),
                                    int(getattr(order, "local_timestamp", current_ts)),
                                    side,
                                    float(order.exec_price),
                                )
                            )
                        _drop_tracked_order(working, cancel_pending, side, tracked_id)
            if fills:
                fills.sort(key=lambda row: (row[0], row[1], row[2]))
                inventory, open_positions = _process_fills(fills, inventory, open_positions, fee_bp, roundtrips)

            for side in (BUY, SELL):
                active_id = working[side]
                if active_id is None:
                    continue
                order = orders.get(active_id)
                if order is None or not _is_active(int(order.status)) or not bool(order.cancellable):
                    continue
                stale = (
                    (side == BUY and float(depth.best_bid) > float(order.price))
                    or (side == SELL and float(depth.best_ask) < float(order.price))
                )
                cancel_on_imbalance = inventory == 0 and _reversal_cancel(
                    side,
                    signal["imbalance"],
                    float(strategy_cfg.imbalance_cancel_threshold),
                )
                if stale or cancel_on_imbalance:
                    hbt.cancel(asset_no, active_id, False)
                    _move_working_to_cancel_pending(working, cancel_pending, side)

            budget = 2 if hbt_cfg.allow_dual_quote_same_step else 1
            quote_sides = _quote_sides_for_step(
                inventory,
                signal,
                bool(hbt_cfg.allow_dual_quote_same_step),
                current_ts=current_ts,
            )
            for side in quote_sides:
                if budget <= 0:
                    break
                if working[side] is not None:
                    continue
                if not _should_quote(mode, side, inventory, signal, strategy_cfg, model_fill_prob_threshold):
                    continue
                price = _post_price(side, depth)
                if not np.isfinite(price) or price <= 0:
                    continue
                rc = _submit_maker(hbt, asset_no, order_id, side, price, strategy_cfg.order_qty)
                if rc == 0:
                    working[side] = order_id
                    order_id += 1
                    budget -= 1

            hbt.clear_inactive_orders(asset_no)
            if recorder is not None:
                recorder.recorder.record(hbt)
        if recorder is not None:
            recorder.recorder.record(hbt)
    finally:
        hbt.close()

    roundtrip_array = np.asarray(roundtrips, dtype=ROUNDTRIP_DTYPE) if roundtrips else np.empty(0, dtype=ROUNDTRIP_DTYPE)
    if not return_record:
        return roundtrip_array
    record = recorder.get(asset_no).copy() if recorder is not None else np.empty(0, dtype=HBT_RECORD_DTYPE)
    return BacktestRun(roundtrips=roundtrip_array, record=record)

def _book_size_from_record(record: np.ndarray, market_cfg, order_qty: float) -> float:
    if record.size == 0:
        return np.nan
    price = float(record["price"][0])
    if not np.isfinite(price) or price <= 0:
        return np.nan
    return float(max(price * order_qty * market_cfg.contract_size, 1e-9))


def build_hbt_stats(record: np.ndarray, market_cfg, order_qty: float, *, resample: str = "1h", trading_days_per_year: int = 365):
    if record.size == 0:
        return None, pd.DataFrame(), {
            HBT_STAT_NAMES["sharpe"]: np.nan,
            HBT_STAT_NAMES["sortino"]: np.nan,
            HBT_STAT_NAMES["return"]: np.nan,
            HBT_STAT_NAMES["annual_return"]: np.nan,
            HBT_STAT_NAMES["max_drawdown"]: np.nan,
            HBT_STAT_NAMES["return_over_mdd"]: np.nan,
            HBT_STAT_NAMES["daily_trades"]: np.nan,
            HBT_STAT_NAMES["daily_trading_value"]: np.nan,
            HBT_STAT_NAMES["max_position_value"]: np.nan,
            "BookSize": np.nan,
        }
    book_size = _book_size_from_record(record, market_cfg, order_qty)
    base_record = LinearAssetRecord(record).contract_size(float(market_cfg.contract_size)).resample(resample)
    prepared = base_record.stats([])
    entire = prepared.entire
    equity = (entire["equity_wo_fee"] - entire["fee"]).to_numpy().astype(np.float64)
    pnl = np.diff(equity)
    pnl = pnl[np.isfinite(pnl)]
    downside = np.minimum(0.0, pnl)
    downside_risk = float(np.sqrt(np.mean(downside * downside))) if downside.size > 0 else np.nan
    max_equity = np.maximum.accumulate(equity) if equity.size else np.empty(0, dtype=np.float64)
    drawdown = max_equity - equity if equity.size else np.empty(0, dtype=np.float64)
    max_drawdown = float(np.max(drawdown)) if drawdown.size else 0.0

    metrics = [
        Ret(HBT_STAT_NAMES["return"], book_size=book_size if np.isfinite(book_size) else None),
        AnnualRet(HBT_STAT_NAMES["annual_return"], book_size=book_size if np.isfinite(book_size) else None, trading_days_per_year=trading_days_per_year),
        MaxDrawdown(HBT_STAT_NAMES["max_drawdown"], book_size=book_size if np.isfinite(book_size) else None),
        DailyNumberOfTrades(HBT_STAT_NAMES["daily_trades"]),
        DailyTradingValue(HBT_STAT_NAMES["daily_trading_value"], book_size=book_size if np.isfinite(book_size) else None),
        MaxPositionValue(HBT_STAT_NAMES["max_position_value"]),
    ]
    if pnl.size > 1 and float(np.std(pnl, ddof=0)) > 1e-12:
        metrics.insert(0, SR(HBT_STAT_NAMES["sharpe"], trading_days_per_year=trading_days_per_year))
    if np.isfinite(downside_risk) and downside_risk > 1e-12:
        metrics.insert(1 if metrics and isinstance(metrics[0], SR) else 0, Sortino(HBT_STAT_NAMES["sortino"], trading_days_per_year=trading_days_per_year))
    if max_drawdown > 1e-12:
        metrics.append(ReturnOverMDD(HBT_STAT_NAMES["return_over_mdd"]))

    stats_obj = LinearAssetRecord(record).contract_size(float(market_cfg.contract_size)).resample(resample).stats(metrics)
    stats_table = pd.DataFrame(stats_obj.summary().to_dicts())
    summary = stats_table.iloc[-1].to_dict() if not stats_table.empty else {}
    summary.setdefault(HBT_STAT_NAMES["sharpe"], np.nan)
    summary.setdefault(HBT_STAT_NAMES["sortino"], np.nan)
    summary.setdefault(HBT_STAT_NAMES["return_over_mdd"], np.nan)
    summary["BookSize"] = book_size
    return stats_obj, stats_table, summary


def run_strategy_days(
    mode: str,
    day_sources,
    signal_map: dict[str, dict[str, np.ndarray]],
    data_files_by_key: dict[str, str],
    market_cfg,
    hbt_cfg,
    strategy_cfg,
    model_fill_prob_threshold: float,
    *,
    start_snapshot=None,
) -> StrategyEvaluation:
    combined_signal = combine_signal_days(day_sources, signal_map)
    ordered_files = [data_files_by_key[infer_day_key(src)] for src in day_sources]
    run = run_backtest(
        mode,
        ordered_files,
        combined_signal,
        market_cfg,
        hbt_cfg,
        strategy_cfg,
        model_fill_prob_threshold,
        snapshot_path=start_snapshot,
        return_record=True,
    )
    stats_obj, stats_table, stats_summary = build_hbt_stats(run.record, market_cfg, strategy_cfg.order_qty)
    return StrategyEvaluation(
        roundtrips=run.roundtrips,
        record=run.record,
        stats_obj=stats_obj,
        stats_table=stats_table,
        stats_summary=stats_summary,
    )


def threshold_backtest_table(
    mode: str,
    thresholds: Iterable[float],
    day_sources,
    signal_map: dict[str, dict[str, np.ndarray]],
    data_files_by_key: dict[str, str],
    market_cfg,
    hbt_cfg,
    strategy_template,
    model_fill_prob_threshold: float,
    *,
    start_snapshot,
    min_roundtrips: int,
    min_daily_trades: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        strategy_cfg = replace(strategy_template, model_threshold=float(threshold))
        result = run_strategy_days(
            mode,
            day_sources,
            signal_map,
            data_files_by_key,
            market_cfg,
            hbt_cfg,
            strategy_cfg,
            model_fill_prob_threshold,
            start_snapshot=start_snapshot,
        )
        sharpe = float(result.stats_summary.get(HBT_STAT_NAMES["sharpe"], np.nan))
        daily_trades = float(result.stats_summary.get(HBT_STAT_NAMES["daily_trades"], np.nan))
        roundtrip_count = int(result.roundtrips.shape[0])
        valid = (
            np.isfinite(sharpe)
            and np.isfinite(daily_trades)
            and daily_trades >= float(min_daily_trades)
            and roundtrip_count >= int(min_roundtrips)
        )
        rows.append(
            {
                "threshold": float(threshold),
                HBT_STAT_NAMES["sharpe"]: sharpe,
                HBT_STAT_NAMES["sortino"]: float(result.stats_summary.get(HBT_STAT_NAMES["sortino"], np.nan)),
                HBT_STAT_NAMES["return"]: float(result.stats_summary.get(HBT_STAT_NAMES["return"], np.nan)),
                HBT_STAT_NAMES["annual_return"]: float(result.stats_summary.get(HBT_STAT_NAMES["annual_return"], np.nan)),
                HBT_STAT_NAMES["max_drawdown"]: float(result.stats_summary.get(HBT_STAT_NAMES["max_drawdown"], np.nan)),
                HBT_STAT_NAMES["return_over_mdd"]: float(result.stats_summary.get(HBT_STAT_NAMES["return_over_mdd"], np.nan)),
                HBT_STAT_NAMES["daily_trades"]: daily_trades,
                HBT_STAT_NAMES["daily_trading_value"]: float(result.stats_summary.get(HBT_STAT_NAMES["daily_trading_value"], np.nan)),
                HBT_STAT_NAMES["max_position_value"]: float(result.stats_summary.get(HBT_STAT_NAMES["max_position_value"], np.nan)),
                "roundtrip_count": roundtrip_count,
                "valid_for_selection": bool(valid),
            }
        )
    return pd.DataFrame(rows)
