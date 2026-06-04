from __future__ import annotations

import argparse
import os
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np

_MPL_DIR = Path(__file__).resolve().parent / "artifacts" / "mplconfig"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib

matplotlib.use("Agg")

from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest, Recorder, ROIVectorMarketDepthBacktest
from hftbacktest.stats import (
    DailyNumberOfTrades,
    DailyTradingValue,
    LinearAssetRecord,
    MaxDrawdown,
    MaxPositionValue,
    Ret,
    ReturnOverMDD,
    ReturnOverTrade,
    SR,
    Sortino,
)

from paper_workflow import combine_signal_days
from reversal_mm_config import default_configs
from reversal_mm_constants import BUY, FILLED, PARTIALLY_FILLED, SELL
from reversal_mm_features import score_grid_signal
from reversal_mm_pipeline import build_console_report, run_full_pipeline
from reversal_mm_simulation import (
    _drop_tracked_order,
    _is_active,
    _is_inactive,
    _move_working_to_cancel_pending,
    _post_price,
    _process_fills,
    _reversal_adverse,
    _reversal_cancel,
    _signal_index,
    _tracked_order_ids,
    _quote_sides_for_step,
    _submit_maker,
    _touch_qty,
    OpenPosition,
)


def _build_asset(data_files, snapshot_path, market_cfg, hbt_cfg) -> BacktestAsset:
    asset = BacktestAsset().data([str(Path(p)) for p in data_files])
    if snapshot_path:
        asset = asset.initial_snapshot(str(snapshot_path))
    asset = asset.linear_asset(float(market_cfg.contract_size))
    if hasattr(asset, "constant_order_latency"):
        asset = asset.constant_order_latency(int(hbt_cfg.entry_latency_ns), int(hbt_cfg.response_latency_ns))
    else:
        asset = asset.constant_latency(int(hbt_cfg.entry_latency_ns), int(hbt_cfg.response_latency_ns))
    asset = asset.power_prob_queue_model(int(hbt_cfg.queue_model_power))
    asset = asset.no_partial_fill_exchange()
    asset = asset.trading_value_fee_model(float(market_cfg.maker_fee), float(market_cfg.taker_fee))
    asset = asset.tick_size(float(market_cfg.tick_size)).lot_size(float(market_cfg.lot_size))
    if hasattr(asset, "last_trades_capacity"):
        asset = asset.last_trades_capacity(0)
    if hasattr(asset, "parallel_load"):
        asset = asset.parallel_load(bool(hbt_cfg.parallel_load))
    return asset


def _build_hbt(data_files, snapshot_path, market_cfg, hbt_cfg):
    asset = _build_asset(data_files, snapshot_path, market_cfg, hbt_cfg)
    if str(hbt_cfg.engine).lower() == "roi_vector":
        asset = asset.roi_lb(float(hbt_cfg.roi_lb)).roi_ub(float(hbt_cfg.roi_ub))
        return ROIVectorMarketDepthBacktest([asset])
    if str(hbt_cfg.engine).lower() == "hashmap":
        return HashMapMarketDepthBacktest([asset])
    raise ValueError(f"Unsupported hftbacktest engine: {hbt_cfg.engine}")


def _result_gate_threshold(results, gate_name: str, fallback: float) -> float:
    gates = getattr(results, "fill_prob_gates", None)
    if gates is None or getattr(gates, "empty", True):
        return float(fallback)
    match = gates.loc[gates["name"] == gate_name, "threshold"]
    if match.empty:
        return float(fallback)
    return float(match.iloc[0])


def _load_or_build_signal(
    path: Path,
    bundle,
    prepared,
    key: str,
    feature_cfg,
    dataset_fill_prob_threshold: float,
    *,
    force: bool,
) -> dict[str, np.ndarray]:
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as zf:
            return {name: zf[name] for name in zf.files}
    window = prepared.day_windows[key]
    signal = score_grid_signal(
        bundle,
        prepared.state,
        prepared.samples[key],
        feature_cfg,
        dataset_fill_prob_threshold,
        day_key=key,
        chunk_size=feature_cfg.score_chunk_size,
        grid_range=(window.grid_start, window.grid_end),
    )
    np.savez_compressed(path, **signal)
    return signal


def _build_test_signal(results, feature_cfg, model_cfg, *, force: bool) -> dict[str, np.ndarray]:
    paths = results.paths
    prepared = results.prepared
    dataset_gate = _result_gate_threshold(results, "dataset_gate_refit", model_cfg.fill_prob_threshold)
    signal_map: dict[str, dict[str, np.ndarray]] = {}
    for key in paths.test_keys:
        signal_map[key] = _load_or_build_signal(
            paths.signal_test_dir / f"sig_{key}.npz",
            results.training.bundle,
            prepared,
            key,
            feature_cfg,
            dataset_gate,
            force=force,
        )
    day_sources = [paths.data_files[key] for key in paths.test_keys]
    return combine_signal_days(day_sources, signal_map)


def _should_quote(side: int, inventory: int, signal: dict[str, float], strategy_cfg, model_fill_prob_threshold: float) -> bool:
    inventory_side = BUY if inventory > 0 else SELL if inventory < 0 else 0
    if inventory_side != 0 and side == inventory_side:
        return False
    fill_prob = signal["fill_prob_buy"] if side == BUY else signal["fill_prob_sell"]
    if strategy_cfg.enable_fill_prob_gate and fill_prob < model_fill_prob_threshold:
        return False
    if not _reversal_adverse(side, signal["imbalance"], float(strategy_cfg.imbalance_post_threshold)):
        return False
    score = signal["p_buy"] if side == BUY else signal["p_sell"]
    return score > float(strategy_cfg.model_threshold)


def market_making_algo(hbt, signal_grid: dict[str, np.ndarray], market_cfg, hbt_cfg, strategy_cfg, *, model_fill_prob_threshold: float, recorder: Recorder | None = None) -> bool:
    asset_no = 0
    working: dict[int, int | None] = {BUY: None, SELL: None}
    cancel_pending: dict[int, list[int]] = {BUY: [], SELL: []}
    order_id = 1
    inventory = 0
    open_positions: deque[OpenPosition] = deque()
    fee_bp = float(market_cfg.maker_fee) * 10_000.0
    sig_ts = signal_grid["ts"].astype(np.int64)
    signal_idx = -1
    roundtrips: list[tuple] = []

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
            if not _should_quote(side, inventory, signal, strategy_cfg, model_fill_prob_threshold):
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
    return True


def _book_size(record: np.ndarray, strategy_cfg, market_cfg) -> float | None:
    if record.size == 0:
        return None
    price = float(record["price"][0])
    if not np.isfinite(price) or price <= 0:
        return None
    return float(max(price * strategy_cfg.order_qty * market_cfg.contract_size, 1e-9))


def _build_stats(record: np.ndarray, strategy_cfg, market_cfg):
    if record.size == 0:
        return None
    book_size = _book_size(record, strategy_cfg, market_cfg)
    metrics = [
        SR("SR", trading_days_per_year=365),
        Sortino("Sortino", trading_days_per_year=365),
        Ret("Return", book_size=book_size),
        MaxDrawdown("MaxDrawdown", book_size=book_size),
        ReturnOverMDD("ReturnOverMDD"),
        ReturnOverTrade("ReturnOverTrade"),
        DailyTradingValue("DailyTradingValue", book_size=book_size),
        DailyNumberOfTrades("DailyNumberOfTrades"),
        MaxPositionValue("MaxPositionValue"),
    ]
    return LinearAssetRecord(record).contract_size(float(market_cfg.contract_size)).resample("1h").stats(metrics)


def main() -> int:
    parser = argparse.ArgumentParser(description="Direct HftBacktest runner for the standalone logistic reversal MM project.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root of the standalone final_sub package.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional manual threshold override. Defaults to the validation-selected threshold.",
    )
    parser.add_argument(
        "--start-window",
        type=int,
        default=None,
        help="Optional 1-based sliding-window number to start from.",
    )
    parser.add_argument(
        "--end-window",
        type=int,
        default=None,
        help="Optional 1-based sliding-window number to stop at.",
    )
    args = parser.parse_args()

    pipeline_cfg, market_cfg, hbt_cfg, feature_cfg, model_cfg, strategy_cfg = default_configs(args.project_root)
    results = run_full_pipeline(
        pipeline_cfg,
        market_cfg,
        hbt_cfg,
        feature_cfg,
        model_cfg,
        strategy_cfg,
        start_window=args.start_window,
        end_window=args.end_window,
    )
    print(build_console_report(results))

    threshold = float(args.threshold) if args.threshold is not None else results.logistic_selected_threshold
    if threshold is None:
        print("No threshold selected from validation. Use --threshold to run a direct test replay anyway.")
        return 0

    test_signal = _build_test_signal(results, feature_cfg, model_cfg, force=False)
    test_files = [results.paths.data_files[key] for key in results.paths.test_keys]
    selected_strategy = replace(strategy_cfg, model_threshold=float(threshold))
    trading_gate = _result_gate_threshold(results, "trading_gate_test", model_cfg.fill_prob_threshold)

    hbt = _build_hbt(test_files, results.prepared.test_start_snapshot, market_cfg, hbt_cfg)

    recorder = Recorder(1, max(int(test_signal["ts"].shape[0]) + 4096, 8192))
    ok = False
    try:
        ok = market_making_algo(
            hbt,
            test_signal,
            market_cfg,
            hbt_cfg,
            selected_strategy,
            model_fill_prob_threshold=trading_gate,
            recorder=recorder,
        )
    finally:
        _ = hbt.close()

    if ok:
        record = recorder.get(0).copy()
        stats = _build_stats(record, selected_strategy, market_cfg)
        if stats is None:
            print("No recorder output available for stats.")
            return 0

        print(stats.summary())
        summary = stats.summary()
        print("Sharpe:", summary["SR"][0])
        print("Sortino:", summary["Sortino"][0])
        print("Return:", summary["Return"][0])
        print("MaxDrawdown:", summary["MaxDrawdown"][0])
        print("ReturnOverMDD:", summary["ReturnOverMDD"][0])
        print("ReturnOverTrade:", summary["ReturnOverTrade"][0])
        print("DailyTradingValue:", summary["DailyTradingValue"][0])
        print("DailyNumberOfTrades:", summary["DailyNumberOfTrades"][0])
        print("MaxPositionValue:", summary["MaxPositionValue"][0])
        print(stats.entire.head())
        plot_obj = stats.plot()
        plot_path = results.paths.reports / "backtest_selected_threshold.png"
        if hasattr(plot_obj, "figure"):
            plot_obj.figure.savefig(plot_path, dpi=160, bbox_inches="tight")
        else:
            import matplotlib.pyplot as plt

            plt.gcf().savefig(plot_path, dpi=160, bbox_inches="tight")
            plt.close("all")
        print(f"Saved plot -> {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
