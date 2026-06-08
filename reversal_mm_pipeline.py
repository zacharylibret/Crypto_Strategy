from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from paper_workflow import build_empirical_fill_surface, fit_fill_surface_ols
from reversal_mm_constants import FILLED, HBT_STAT_NAMES, MODE_REVERSAL
from reversal_mm_data import PreparedData, ProjectPaths, build_project_paths, prepare_market_data
from reversal_mm_features import Dataset, build_dataset, merge_datasets, near_opp_notional, score_grid_signal
from reversal_mm_modeling import ModelTrainingResult, evaluate_probabilistic_classifier, fit_reversal_models
from reversal_mm_simulation import run_strategy_days, threshold_backtest_table
from reversal_mm_utils import ensure_dir, format_table
from reversal_mm_config import synchronize_configs


@dataclass(slots=True)
class PipelineResults:
    paths: ProjectPaths
    prepared: PreparedData
    fill_surface: object
    fill_surface_ols: dict
    training: ModelTrainingResult
    train_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset
    logistic_metrics: pd.DataFrame
    logistic_validation_backtest: pd.DataFrame
    logistic_test_threshold_backtest: pd.DataFrame
    logistic_test_backtest: pd.DataFrame
    logistic_shadow_test_threshold_backtest: pd.DataFrame
    logistic_shadow_test_backtest: pd.DataFrame
    logistic_selected_threshold: float | None
    logistic_selected_threshold_quantile: float | None
    logistic_validation_selected_threshold: float | None
    logistic_threshold_grid: tuple[float, ...]
    fill_prob_gates: pd.DataFrame
    selected_test_daily_pnl: pd.DataFrame


@dataclass(slots=True)
class SlidingWindowSpec:
    index: int
    train_keys: tuple[str, ...]
    validation_keys: tuple[str, ...]
    test_keys: tuple[str, ...]
    name: str


@dataclass(slots=True)
class FillProbGate:
    name: str
    threshold: float
    selected_quantile: float
    total_count: int
    kept_count: int
    kept_share: float
    min_count_floor: int
    kept_fill_count: int | None = None
    min_fill_floor: int | None = None
    floor_met: bool = True


@dataclass(slots=True)
class CalibrationSource:
    mode: str
    window_name: str
    window_index: int | None = None
    day_keys: tuple[str, ...] = ()
    report_path: str | None = None


@dataclass(slots=True)
class CalibrationSelection:
    source: CalibrationSource
    validation_backtest: pd.DataFrame
    threshold_candidates: pd.DataFrame
    threshold_grid: tuple[float, ...]
    selected_threshold_quantile: float | None
    calibration_selected_threshold: float | None
    validation_selected_threshold: float | None
    selected_row: dict | None
    criterion: str


def _window_step_days(validation_days: int, test_days: int) -> int:
    validation_days = int(validation_days)
    test_days = int(test_days)
    if validation_days <= 0 or test_days <= 0:
        raise ValueError("validation_days and test_days must both be positive.")
    return test_days


WINDOW_TRAIN_DAYS = 21
WINDOW_VALIDATION_DAYS = 7
WINDOW_TEST_DAYS = 1
WINDOW_STEP_DAYS = _window_step_days(WINDOW_VALIDATION_DAYS, WINDOW_TEST_DAYS)
WINDOW_REFIT_DAYS = WINDOW_TRAIN_DAYS


def _window_name(index: int, train_keys: tuple[str, ...], test_keys: tuple[str, ...]) -> str:
    return f"window_{index:03d}_{train_keys[0]}_{test_keys[-1]}"


def _window_ordered_keys(spec: SlidingWindowSpec) -> tuple[str, ...]:
    return (*spec.train_keys, *spec.validation_keys, *spec.test_keys)


def _window_number(spec: SlidingWindowSpec) -> int:
    return int(spec.index) + 1


def _sliding_window_specs(
    ordered_keys: tuple[str, ...],
    *,
    train_days: int = WINDOW_TRAIN_DAYS,
    validation_days: int = WINDOW_VALIDATION_DAYS,
    test_days: int = WINDOW_TEST_DAYS,
    step_days: int | None = None,
) -> list[SlidingWindowSpec]:
    total_days = int(train_days) + int(validation_days) + int(test_days)
    if len(ordered_keys) < total_days:
        raise ValueError(
            f"Need at least {total_days} day files for a train/validation/test window, got {len(ordered_keys)}."
        )
    step = _window_step_days(validation_days, test_days) if step_days is None else int(step_days)
    if step <= 0:
        raise ValueError(f"step_days must be positive, got {step}.")
    # Advance by the test block size so each new window rolls forward by the
    # next out-of-sample segment and test coverage stays continuous.
    starts = list(range(0, len(ordered_keys) - total_days + 1, step))

    specs: list[SlidingWindowSpec] = []
    for index, start in enumerate(starts):
        train_keys = tuple(ordered_keys[start : start + train_days])
        validation_keys = tuple(ordered_keys[start + train_days : start + train_days + validation_days])
        test_keys = tuple(ordered_keys[start + train_days + validation_days : start + total_days])
        specs.append(
            SlidingWindowSpec(
                index=index,
                train_keys=train_keys,
                validation_keys=validation_keys,
                test_keys=test_keys,
                name=_window_name(index, train_keys, test_keys),
            )
        )
    return specs


def _select_window_specs(
    window_specs: list[SlidingWindowSpec],
    *,
    start_window: int | None,
    end_window: int | None,
) -> tuple[list[SlidingWindowSpec], int, int]:
    if not window_specs:
        raise ValueError("No sliding windows were generated.")
    total_windows = len(window_specs)
    start = 1 if start_window is None else int(start_window)
    end = total_windows if end_window is None else int(end_window)
    if start < 1 or start > total_windows:
        raise ValueError(f"start_window must be between 1 and {total_windows}, got {start}.")
    if end < 1 or end > total_windows:
        raise ValueError(f"end_window must be between 1 and {total_windows}, got {end}.")
    if end < start:
        raise ValueError(f"end_window ({end}) must be >= start_window ({start}).")
    selected = [spec for spec in window_specs if start <= _window_number(spec) <= end]
    if not selected:
        raise ValueError(f"No windows selected for range {start}..{end}.")
    return selected, start, end


def _load_existing_summary_rows(summary_path: Path, selected_specs: list[SlidingWindowSpec]) -> list[dict[str, object]]:
    if not summary_path.exists():
        return []
    try:
        existing = pd.read_csv(summary_path)
    except Exception as exc:
        print(f"Existing summary read failed for {summary_path.name}: {exc}. Rebuilding selected summary rows.")
        return []
    if existing.empty:
        return []

    selected_indices = {int(spec.index) for spec in selected_specs}
    selected_names = {spec.name for spec in selected_specs}
    if "window_index" in existing.columns:
        mask = ~existing["window_index"].astype("Int64").isin(selected_indices)
        existing = existing.loc[mask].copy()
    elif "window_name" in existing.columns:
        existing = existing.loc[~existing["window_name"].isin(selected_names)].copy()
    else:
        print(f"Existing summary {summary_path.name} has no window key columns; ignoring previous rows.")
        return []

    if "window_index" in existing.columns:
        existing = existing.sort_values("window_index").reset_index(drop=True)
    return existing.to_dict(orient="records")


def _window_paths(base_paths: ProjectPaths, spec: SlidingWindowSpec) -> ProjectPaths:
    window_keys = _window_ordered_keys(spec)
    return ProjectPaths(
        project_root=base_paths.project_root,
        artifacts=base_paths.artifacts,
        reports=ensure_dir(base_paths.reports / spec.name),
        state_dir=ensure_dir(base_paths.state_dir / spec.name),
        snapshot_dir=ensure_dir(base_paths.snapshot_dir / spec.name),
        sample_dir=ensure_dir(base_paths.sample_dir / spec.name),
        signal_val_dir=ensure_dir(base_paths.signal_val_dir / spec.name),
        signal_test_dir=ensure_dir(base_paths.signal_test_dir / spec.name),
        ordered_keys=window_keys,
        train_keys=spec.train_keys,
        validation_keys=spec.validation_keys,
        test_keys=spec.test_keys,
        data_files={key: base_paths.data_files[key] for key in window_keys},
    )


def _load_previous_test_calibration_table(
    base_paths: ProjectPaths,
    previous_spec: SlidingWindowSpec | None,
    current_spec: SlidingWindowSpec,
) -> tuple[pd.DataFrame | None, CalibrationSource | None]:
    if previous_spec is None:
        return None, None
    if len(previous_spec.test_keys) != len(current_spec.validation_keys):
        return None, None
    if tuple(previous_spec.test_keys) != tuple(current_spec.validation_keys):
        return None, None
    path = base_paths.reports / previous_spec.name / "logistic_test_threshold_backtest.csv"
    if not path.exists():
        return None, None
    try:
        table = pd.read_csv(path)
    except Exception as exc:
        print(f"Previous test calibration read failed for {previous_spec.name}: {exc}. Falling back to validation sweep.")
        return None, None
    if table.empty:
        return None, None
    return table, CalibrationSource(
        mode="previous_test_sweep",
        window_name=previous_spec.name,
        window_index=int(previous_spec.index),
        day_keys=tuple(previous_spec.test_keys),
        report_path=str(path),
    )


def _refit_keys_for_test(paths: ProjectPaths) -> tuple[str, ...]:
    pretest_keys = (*paths.train_keys, *paths.validation_keys)
    if not pretest_keys:
        return tuple()
    return tuple(pretest_keys[-WINDOW_REFIT_DAYS:])


def _quantile_grid(low: float, high: float, n: int = 12) -> np.ndarray:
    lo = float(np.clip(low, 0.0, 1.0))
    hi = float(np.clip(high, 0.0, 1.0))
    if hi < lo:
        lo, hi = hi, lo
    if np.isclose(lo, hi):
        return np.array([lo], dtype=np.float64)
    return np.linspace(lo, hi, n, dtype=np.float64)


def _resolved_fill_surface_cap_quantile(model_cfg) -> float:
    quantile = getattr(model_cfg, "fill_surface_notional_cap_quantile")
    quantile = float(quantile)
    if not 0.0 < quantile <= 1.0:
        raise ValueError(
            "fill_surface_notional_cap_quantile must be in (0, 1], "
            f"got {quantile}."
        )
    return quantile


def _resolved_fill_surface_cap(
    near: np.ndarray,
    opp: np.ndarray,
    model_cfg,
) -> tuple[float | None, float | None, str]:
    cap_quantile = _resolved_fill_surface_cap_quantile(model_cfg)
    near_vals = np.asarray(near, dtype=np.float64)
    opp_vals = np.asarray(opp, dtype=np.float64)
    finite = np.concatenate([near_vals[np.isfinite(near_vals)], opp_vals[np.isfinite(opp_vals)]])
    if finite.size == 0:
        return None, cap_quantile, "window_quantile"
    return float(np.quantile(finite, cap_quantile)), cap_quantile, "window_quantile"


def _fill_surface_summary(
    surface,
    *,
    raw_order_count: int,
    fill_surface_quantile: float,
    cap_mode: str,
    cap_quantile: float | None,
) -> dict[str, object]:
    raw_count = int(raw_order_count)
    used_count = int(surface.n_orders_used)
    dropped_count = max(raw_count - used_count, 0)
    return {
        "fill_surface_quantile": float(fill_surface_quantile),
        "max_notional_cap": surface.max_notional_cap if surface.max_notional_cap is not None else np.nan,
        "max_notional_cap_mode": cap_mode,
        "max_notional_cap_quantile": cap_quantile if cap_quantile is not None else np.nan,
        "n_orders_before_cap": raw_count,
        "n_orders_used": used_count,
        "n_orders_dropped_by_cap": dropped_count,
        "cap_keep_share": float(used_count / raw_count) if raw_count > 0 else np.nan,
    }


def _selected_test_evaluation(
    threshold: float | None,
    bundle,
    prepared: PreparedData,
    paths: ProjectPaths,
    market_cfg,
    hbt_cfg,
    strategy_cfg,
    feature_cfg,
    dataset_fill_prob_threshold: float,
    trading_fill_prob_threshold: float,
) -> object | None:
    if threshold is None:
        return None
    signal_map = _signal_map_for_keys(
        bundle,
        prepared,
        paths,
        paths.test_keys,
        paths.signal_test_dir,
        feature_cfg,
        dataset_fill_prob_threshold,
        force=False,
    )
    return run_strategy_days(
        MODE_REVERSAL,
        [paths.data_files[key] for key in paths.test_keys],
        signal_map,
        {key: str(paths.data_files[key]) for key in paths.ordered_keys},
        market_cfg,
        hbt_cfg,
        replace(strategy_cfg, model_threshold=float(threshold)),
        trading_fill_prob_threshold,
        start_snapshot=prepared.test_start_snapshot,
    )


def _daily_pnl_frame_from_record(record: np.ndarray, market_cfg, *, window_index: int, window_name: str) -> pd.DataFrame:
    if record.size == 0:
        return pd.DataFrame(columns=["window_index", "window_name", "timestamp", "daily_pnl", "cumulative_window_pnl"])

    from hftbacktest.stats import LinearAssetRecord

    # Anchor daily PnL to the same 1-hour-resampled equity series used by the
    # backtest summary metrics so cumulative daily PnL matches ReturnPct.
    base_stats = LinearAssetRecord(record).contract_size(float(market_cfg.contract_size)).resample("1h").stats([])
    base = pd.DataFrame(base_stats.entire.to_dict(as_series=False))
    if base.empty:
        return pd.DataFrame(columns=["window_index", "window_name", "timestamp", "daily_pnl", "cumulative_window_pnl"])

    base = base.copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"])
    base["equity"] = base["equity_wo_fee"] - base["fee"]

    daily = (
        base.assign(timestamp=base["timestamp"].dt.floor("1d"))
        .groupby("timestamp", as_index=False)["equity"]
        .last()
    )
    initial_equity = float(base["equity"].iloc[0])
    daily["daily_pnl"] = daily["equity"].diff()
    daily.loc[daily.index[0], "daily_pnl"] = daily.loc[daily.index[0], "equity"] - initial_equity
    daily["cumulative_window_pnl"] = daily["daily_pnl"].cumsum()
    out = daily.loc[:, ["timestamp", "daily_pnl", "cumulative_window_pnl"]].copy()
    out.insert(0, "window_name", window_name)
    out.insert(0, "window_index", int(window_index))
    return out


def _write_root_pnl_report_outputs(reports_dir: Path, updated: pd.DataFrame) -> None:
    pnl_csv = reports_dir / "pnl_daily.csv"
    pnl_png = reports_dir / "pnl.png"

    if updated.empty:
        if pnl_csv.exists():
            pnl_csv.unlink()
        if pnl_png.exists():
            pnl_png.unlink()
        return

    updated = updated.sort_values(["timestamp", "window_index"]).reset_index(drop=True)
    daily = (
        updated.groupby("timestamp", as_index=False)["daily_pnl"]
        .sum()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    daily["cumulative_pnl"] = daily["daily_pnl"].cumsum()

    updated.to_csv(pnl_csv, index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily["timestamp"], daily["cumulative_pnl"], color="steelblue", linewidth=1.8)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Cumulative Daily PnL Across Completed Windows")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(pnl_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _load_completed_window_daily_pnl(reports_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for window_dir in sorted(path for path in reports_dir.glob("window_*") if path.is_dir()):
        daily_path = window_dir / "selected_test_daily_pnl.csv"
        if not daily_path.exists():
            continue
        frame = pd.read_csv(daily_path, parse_dates=["timestamp"])
        if frame.empty:
            continue
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["window_index", "window_name", "timestamp", "daily_pnl", "cumulative_window_pnl"])
    return pd.concat(frames, ignore_index=True)


def _update_root_pnl_report(reports_dir: Path, spec: SlidingWindowSpec, daily_pnl: pd.DataFrame) -> None:
    updated = _load_completed_window_daily_pnl(reports_dir)
    if updated.empty and daily_pnl is not None and not daily_pnl.empty:
        updated = daily_pnl.copy()
    _write_root_pnl_report_outputs(reports_dir, updated)


def _select_gate_from_values(
    values: np.ndarray,
    *,
    name: str,
    quantile_low: float,
    quantile_high: float,
    min_count: int,
    fill_flags: np.ndarray | None = None,
    min_fill_count: int | None = None,
) -> FillProbGate:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return FillProbGate(
            name=name,
            threshold=float("inf"),
            selected_quantile=float(quantile_low),
            total_count=0,
            kept_count=0,
            kept_share=float("nan"),
            min_count_floor=int(min_count),
            kept_fill_count=None if fill_flags is None else 0,
            min_fill_floor=None if min_fill_count is None else int(min_fill_count),
            floor_met=False,
        )

    fills = None
    if fill_flags is not None:
        fills = np.asarray(fill_flags, dtype=bool)
        if fills.shape[0] != values.shape[0]:
            raise ValueError(f"Fill flags for {name} do not align with gate values.")
        fills = fills[np.isfinite(values)]

    quantiles = _quantile_grid(quantile_low, quantile_high)
    selected: FillProbGate | None = None
    for q in quantiles[::-1]:
        threshold = float(np.quantile(vals, q))
        keep = vals >= threshold
        kept_count = int(np.sum(keep))
        kept_fill_count = None if fills is None else int(np.sum(fills[keep]))
        floor_met = kept_count >= int(min_count) and (
            min_fill_count is None or (kept_fill_count is not None and kept_fill_count >= int(min_fill_count))
        )
        selected = FillProbGate(
            name=name,
            threshold=threshold,
            selected_quantile=float(q),
            total_count=int(vals.size),
            kept_count=kept_count,
            kept_share=float(kept_count / max(int(vals.size), 1)),
            min_count_floor=int(min_count),
            kept_fill_count=kept_fill_count,
            min_fill_floor=None if min_fill_count is None else int(min_fill_count),
            floor_met=bool(floor_met),
        )
        if floor_met:
            return selected

    assert selected is not None
    return selected


def _gate_from_fixed_quantile(
    values: np.ndarray,
    *,
    name: str,
    selected_quantile: float,
    min_count: int,
    fill_flags: np.ndarray | None = None,
    min_fill_count: int | None = None,
) -> FillProbGate:
    raw_values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(raw_values)
    vals = raw_values[finite]
    q = float(np.clip(selected_quantile, 0.0, 1.0))
    if vals.size == 0:
        return FillProbGate(
            name=name,
            threshold=float("inf"),
            selected_quantile=q,
            total_count=0,
            kept_count=0,
            kept_share=float("nan"),
            min_count_floor=int(min_count),
            kept_fill_count=None if fill_flags is None else 0,
            min_fill_floor=None if min_fill_count is None else int(min_fill_count),
            floor_met=False,
        )

    fills = None
    if fill_flags is not None:
        fills = np.asarray(fill_flags, dtype=bool)
        if fills.shape[0] != raw_values.shape[0]:
            raise ValueError(f"Fill flags for {name} do not align with gate values.")
        fills = fills[finite]

    threshold = float(np.quantile(vals, q))
    keep = vals >= threshold
    kept_count = int(np.sum(keep))
    kept_fill_count = None if fills is None else int(np.sum(fills[keep]))
    floor_met = kept_count >= int(min_count) and (
        min_fill_count is None or (kept_fill_count is not None and kept_fill_count >= int(min_fill_count))
    )
    return FillProbGate(
        name=name,
        threshold=threshold,
        selected_quantile=q,
        total_count=int(vals.size),
        kept_count=kept_count,
        kept_share=float(kept_count / max(int(vals.size), 1)),
        min_count_floor=int(min_count),
        kept_fill_count=kept_fill_count,
        min_fill_floor=None if min_fill_count is None else int(min_fill_count),
        floor_met=bool(floor_met),
    )


def _order_fill_prob_values(orders: np.ndarray, fill_surface) -> np.ndarray:
    if orders.size == 0:
        return np.empty(0, dtype=np.float64)
    near, opp = near_opp_notional(orders)
    return fill_surface.predict(near, opp).astype(np.float64)


def _grid_fill_prob_values(prepared: PreparedData, day_keys: tuple[str, ...], fill_surface) -> np.ndarray:
    values: list[np.ndarray] = []
    for key in day_keys:
        window = prepared.day_windows[key]
        start = int(window.grid_start)
        end = int(window.grid_end)
        if end <= start:
            continue
        bid_notional = np.asarray(prepared.state.best_bid_notional[start:end], dtype=np.float64)
        ask_notional = np.asarray(prepared.state.best_ask_notional[start:end], dtype=np.float64)
        eligible = np.isfinite(bid_notional) & np.isfinite(ask_notional) & (bid_notional > 0.0) & (ask_notional > 0.0)
        if not np.any(eligible):
            continue
        bid_eligible = bid_notional[eligible]
        ask_eligible = ask_notional[eligible]
        values.append(fill_surface.predict(bid_eligible, ask_eligible).astype(np.float64))
        values.append(fill_surface.predict(ask_eligible, bid_eligible).astype(np.float64))
    if not values:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(values)


def _select_dataset_gate(orders: np.ndarray, fill_surface, model_cfg, *, name: str) -> FillProbGate:
    values = _order_fill_prob_values(orders, fill_surface)
    filled = orders["status"] == FILLED if orders.size > 0 else np.empty(0, dtype=bool)
    return _select_gate_from_values(
        values,
        name=name,
        quantile_low=model_cfg.dataset_gate_quantile_low,
        quantile_high=model_cfg.dataset_gate_quantile_high,
        min_count=model_cfg.dataset_gate_min_orders,
        fill_flags=filled,
        min_fill_count=model_cfg.dataset_gate_min_fills,
    )


def _select_trading_gate(prepared: PreparedData, day_keys: tuple[str, ...], fill_surface, model_cfg, *, name: str) -> FillProbGate:
    values = _grid_fill_prob_values(prepared, day_keys, fill_surface)
    return _select_gate_from_values(
        values,
        name=name,
        quantile_low=model_cfg.trading_gate_quantile_low,
        quantile_high=model_cfg.trading_gate_quantile_high,
        min_count=model_cfg.trading_gate_min_orders,
    )


def _remap_dataset_gate(
    orders: np.ndarray,
    fill_surface,
    model_cfg,
    reference_gate: FillProbGate,
    *,
    name: str,
) -> FillProbGate:
    values = _order_fill_prob_values(orders, fill_surface)
    filled = orders["status"] == FILLED if orders.size > 0 else np.empty(0, dtype=bool)
    return _gate_from_fixed_quantile(
        values,
        name=name,
        selected_quantile=reference_gate.selected_quantile,
        min_count=model_cfg.dataset_gate_min_orders,
        fill_flags=filled,
        min_fill_count=model_cfg.dataset_gate_min_fills,
    )


def _remap_trading_gate(
    prepared: PreparedData,
    day_keys: tuple[str, ...],
    fill_surface,
    model_cfg,
    reference_gate: FillProbGate,
    *,
    name: str,
) -> FillProbGate:
    values = _grid_fill_prob_values(prepared, day_keys, fill_surface)
    return _gate_from_fixed_quantile(
        values,
        name=name,
        selected_quantile=reference_gate.selected_quantile,
        min_count=model_cfg.trading_gate_min_orders,
    )


def _datasets_for_keys(prepared: PreparedData, day_keys: tuple[str, ...], feature_cfg, fill_surface, fill_prob_threshold: float) -> list[Dataset]:
    if not day_keys:
        return []
    resolve_end_ts = int(prepared.day_windows[day_keys[-1]].local_ts_end_exclusive)
    datasets: list[Dataset] = []
    for key in day_keys:
        orders = prepared.samples[key]
        if orders.size == 0:
            continue
        datasets.append(
            build_dataset(
                orders,
                prepared.state,
                feature_cfg,
                fill_surface,
                fill_prob_threshold,
                day_key=key,
                resolve_end_ts=resolve_end_ts,
            )
        )
    return datasets


def _orders_for_keys(prepared: PreparedData, day_keys: tuple[str, ...]) -> np.ndarray:
    blocks = [prepared.samples[key] for key in day_keys if prepared.samples[key].size > 0]
    return np.concatenate(blocks) if blocks else np.empty(0, dtype=prepared.samples[next(iter(prepared.samples))].dtype)


def _resolved_orders_for_keys(prepared: PreparedData, day_keys: tuple[str, ...]) -> np.ndarray:
    orders = _orders_for_keys(prepared, day_keys)
    if orders.size == 0 or not day_keys:
        return orders
    resolve_end_ts = int(prepared.day_windows[day_keys[-1]].local_ts_end_exclusive)
    return orders[orders["terminal_local_ts"].astype(np.int64) < resolve_end_ts]


def _select_stable_supported_threshold_quantile(
    table: pd.DataFrame,
    *,
    min_threshold_quantile: float | None = None,
) -> tuple[float | None, float | None, dict | None, str]:
    if table.empty:
        return None, None, None, "empty HftBacktest validation table"
    order_col = "threshold_quantile" if "threshold_quantile" in table.columns else "threshold"
    df = table.copy().sort_values(order_col).reset_index(drop=True)
    sharpe_col = HBT_STAT_NAMES["sharpe"]
    support = df["valid_for_selection"].astype(bool) & np.isfinite(df[sharpe_col])
    supported = df.loc[support].copy().reset_index(drop=True)
    if supported.empty:
        return None, None, None, f"no supported HftBacktest {sharpe_col} thresholds"
    floor_text = ""
    floor_value: float | None = None
    if min_threshold_quantile is not None and "threshold_quantile" in supported.columns:
        floor_value = float(min_threshold_quantile)
        supported = supported.loc[
            supported["threshold_quantile"].astype(np.float64) >= floor_value
        ].copy().reset_index(drop=True)
        floor_text = f" with threshold_quantile >= {floor_value:.3f}"
        if supported.empty:
            return None, None, None, f"no supported HftBacktest {sharpe_col} thresholds{floor_text}"
    positive = supported.loc[supported[sharpe_col] > 0.0].copy().reset_index(drop=True)
    if positive.empty:
        return None, None, None, f"no supported positive HftBacktest {sharpe_col} threshold{floor_text}"
    raw_sharpe = positive[sharpe_col].astype(np.float64).to_numpy()
    smooth_sharpe = np.empty_like(raw_sharpe)
    for idx in range(raw_sharpe.size):
        lo = max(0, idx - 1)
        hi = min(raw_sharpe.size, idx + 2)
        smooth_sharpe[idx] = float(np.median(raw_sharpe[lo:hi]))
    positive["smoothed_sharpe"] = smooth_sharpe
    top_k = int(min(3, positive.shape[0]))
    candidates = (
        positive.sort_values(["smoothed_sharpe", sharpe_col, order_col], ascending=[False, False, True])
        .head(top_k)
        .sort_values(order_col)
        .reset_index(drop=True)
    )
    selected_index = int(candidates.shape[0] // 2)
    selected_threshold = float(candidates.loc[selected_index, "threshold"])
    selected_quantile = (
        float(candidates.loc[selected_index, "threshold_quantile"]) if "threshold_quantile" in candidates.columns else None
    )
    payload = {
        "selected_threshold_quantile": selected_quantile,
        "selected_threshold": selected_threshold,
        "candidate_count": top_k,
        "supported_positive_count": int(positive.shape[0]),
        "threshold_quantile_floor": floor_value,
        "candidate_rows": candidates.to_dict(orient="records"),
    }
    criterion = (
        f"select median {order_col} row among top {top_k} supported positive thresholds ranked by "
        f"3-point median-smoothed HftBacktest {sharpe_col}{floor_text}"
    )
    return selected_quantile, selected_threshold, payload, criterion


def _threshold_for_quantile(candidates: pd.DataFrame, threshold_quantile: float | None) -> float | None:
    if threshold_quantile is None or candidates.empty or "threshold_quantile" not in candidates.columns:
        return None
    quantiles = candidates["threshold_quantile"].astype(np.float64).to_numpy()
    matches = np.isclose(quantiles, float(threshold_quantile), rtol=0.0, atol=1e-12)
    if np.any(matches):
        return float(candidates.loc[matches, "threshold"].iloc[0])
    idx = int(np.argmin(np.abs(quantiles - float(threshold_quantile))))
    return float(candidates.iloc[idx]["threshold"])


def _threshold_reference_signal_dir(paths: ProjectPaths) -> Path:
    return ensure_dir(paths.signal_test_dir / "_refit_threshold_reference")


def _shadow_consistency_signal_dir(paths: ProjectPaths) -> Path:
    return ensure_dir(paths.signal_test_dir / "_validation_policy_shadow")


def _score_signal_grid(
    path: Path,
    bundle,
    prepared: PreparedData,
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


def _signal_map_for_keys(
    bundle,
    prepared: PreparedData,
    paths: ProjectPaths,
    day_keys: tuple[str, ...],
    signal_dir: Path,
    feature_cfg,
    dataset_fill_prob_threshold: float,
    *,
    force: bool,
) -> dict[str, dict[str, np.ndarray]]:
    signal_map: dict[str, dict[str, np.ndarray]] = {}
    for key in day_keys:
        signal_map[key] = _score_signal_grid(
            signal_dir / f"sig_{key}.npz",
            bundle,
            prepared,
            key,
            feature_cfg,
            dataset_fill_prob_threshold,
            force=force,
        )
    return signal_map


def _threshold_candidates_from_signal_map(
    signal_map: dict[str, dict[str, np.ndarray]],
    quantile_levels: tuple[float, ...],
) -> pd.DataFrame:
    levels = np.asarray(quantile_levels, dtype=np.float64)
    levels = levels[np.isfinite(levels)]
    levels = np.clip(levels, 0.0, 1.0)
    if levels.size == 0:
        return pd.DataFrame(columns=["threshold_quantile", "threshold"])
    levels = np.unique(levels)
    score_blocks: list[np.ndarray] = []
    for signal in signal_map.values():
        for name in ("p_buy", "p_sell"):
            vals = np.asarray(signal[name], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                score_blocks.append(finite)
    if not score_blocks:
        return pd.DataFrame(columns=["threshold_quantile", "threshold"])
    scores = np.concatenate(score_blocks)
    thresholds = np.quantile(scores, levels).astype(np.float64)
    candidates = pd.DataFrame(
        {
            "threshold_quantile": levels.astype(np.float64),
            "threshold": thresholds,
        }
    )
    candidates["threshold"] = candidates["threshold"].round(12)
    return candidates


def _resolve_calibration_selection(
    bundle,
    prepared: PreparedData,
    paths: ProjectPaths,
    market_cfg,
    hbt_cfg,
    strategy_cfg,
    model_cfg,
    feature_cfg,
    dataset_fill_prob_threshold: float,
    trading_fill_prob_threshold: float,
    *,
    calibration_table: pd.DataFrame | None,
    calibration_source: CalibrationSource | None,
    force: bool,
) -> CalibrationSelection:
    validation_signal_map = _signal_map_for_keys(
        bundle,
        prepared,
        paths,
        paths.validation_keys,
        paths.signal_val_dir,
        feature_cfg,
        dataset_fill_prob_threshold,
        force=force,
    )
    threshold_candidates = _threshold_candidates_from_signal_map(validation_signal_map, model_cfg.threshold_quantiles)
    threshold_grid = tuple(float(x) for x in threshold_candidates["threshold"].astype(np.float64).to_numpy())
    default_source = CalibrationSource(
        mode="validation_sweep",
        window_name=paths.reports.name,
        window_index=int(paths.reports.name.split("_")[1]),
        day_keys=tuple(paths.validation_keys),
    )
    if calibration_table is not None:
        source = calibration_source if calibration_source is not None else default_source
        print(f"Reusing previous window test-days threshold sweep as calibration from {source.window_name}...")
        validation_backtest = calibration_table.copy()
    else:
        source = default_source
        print("Running validation balanced-inventory backtest sweeps...")
        thresholds = tuple(float(x) for x in threshold_candidates["threshold"].astype(np.float64).to_numpy())
        data_files_by_key = {key: str(paths.data_files[key]) for key in paths.ordered_keys}
        day_sources = [paths.data_files[key] for key in paths.validation_keys]
        validation_backtest = threshold_backtest_table(
            MODE_REVERSAL,
            thresholds,
            day_sources,
            validation_signal_map,
            data_files_by_key,
            market_cfg,
            hbt_cfg,
            strategy_cfg,
            trading_fill_prob_threshold,
            start_snapshot=prepared.validation_start_snapshot,
            min_roundtrips=model_cfg.min_validation_roundtrips,
            min_daily_trades=model_cfg.min_validation_daily_trades,
        )
        if threshold_candidates.shape[0] == validation_backtest.shape[0]:
            validation_backtest = validation_backtest.copy()
            validation_backtest.insert(
                0,
                "threshold_quantile",
                threshold_candidates["threshold_quantile"].to_numpy(dtype=np.float64),
            )
    (
        selected_threshold_quantile,
        calibration_selected_threshold,
        selected_row,
        criterion,
    ) = _select_stable_supported_threshold_quantile(
        validation_backtest,
        min_threshold_quantile=model_cfg.selection_threshold_quantile_floor,
    )
    validation_selected_threshold = _threshold_for_quantile(
        threshold_candidates,
        selected_threshold_quantile,
    )
    if validation_selected_threshold is None:
        validation_selected_threshold = calibration_selected_threshold
    return CalibrationSelection(
        source=source,
        validation_backtest=validation_backtest,
        threshold_candidates=threshold_candidates,
        threshold_grid=threshold_grid,
        selected_threshold_quantile=selected_threshold_quantile,
        calibration_selected_threshold=calibration_selected_threshold,
        validation_selected_threshold=validation_selected_threshold,
        selected_row=selected_row,
        criterion=criterion,
    )


def _test_backtest_table(
    threshold: float | None,
    threshold_quantile: float | None,
    bundle,
    prepared: PreparedData,
    paths: ProjectPaths,
    market_cfg,
    hbt_cfg,
    strategy_cfg,
    model_cfg,
    feature_cfg,
    dataset_fill_prob_threshold: float,
    trading_fill_prob_threshold: float,
    *,
    force: bool,
    signal_dir: Path | None = None,
) -> pd.DataFrame:
    if threshold is None:
        return pd.DataFrame()
    effective_signal_dir = signal_dir if signal_dir is not None else paths.signal_test_dir
    signal_map = _signal_map_for_keys(
        bundle,
        prepared,
        paths,
        paths.test_keys,
        effective_signal_dir,
        feature_cfg,
        dataset_fill_prob_threshold,
        force=force,
    )
    data_files_by_key = {key: str(paths.data_files[key]) for key in paths.ordered_keys}
    day_sources = [paths.data_files[key] for key in paths.test_keys]
    table = threshold_backtest_table(
        MODE_REVERSAL,
        [float(threshold)],
        day_sources,
        signal_map,
        data_files_by_key,
        market_cfg,
        hbt_cfg,
        strategy_cfg,
        trading_fill_prob_threshold,
        start_snapshot=prepared.test_start_snapshot,
        min_roundtrips=model_cfg.min_validation_roundtrips,
        min_daily_trades=model_cfg.min_validation_daily_trades,
    )
    if threshold_quantile is not None and not table.empty:
        table = table.copy()
        table.insert(0, "threshold_quantile", float(threshold_quantile))
    return table


def _test_threshold_backtest_table(
    bundle,
    prepared: PreparedData,
    paths: ProjectPaths,
    market_cfg,
    hbt_cfg,
    strategy_cfg,
    model_cfg,
    feature_cfg,
    threshold_candidates: pd.DataFrame,
    thresholds: tuple[float, ...],
    dataset_fill_prob_threshold: float,
    trading_fill_prob_threshold: float,
    *,
    force: bool,
    signal_dir: Path | None = None,
) -> pd.DataFrame:
    effective_signal_dir = signal_dir if signal_dir is not None else paths.signal_test_dir
    signal_map = _signal_map_for_keys(
        bundle,
        prepared,
        paths,
        paths.test_keys,
        effective_signal_dir,
        feature_cfg,
        dataset_fill_prob_threshold,
        force=force,
    )
    data_files_by_key = {key: str(paths.data_files[key]) for key in paths.ordered_keys}
    day_sources = [paths.data_files[key] for key in paths.test_keys]
    table = threshold_backtest_table(
        MODE_REVERSAL,
        thresholds,
        day_sources,
        signal_map,
        data_files_by_key,
        market_cfg,
        hbt_cfg,
        strategy_cfg,
        trading_fill_prob_threshold,
        start_snapshot=prepared.test_start_snapshot,
        min_roundtrips=model_cfg.min_validation_roundtrips,
        min_daily_trades=model_cfg.min_validation_daily_trades,
    )
    if threshold_candidates.shape[0] == table.shape[0]:
        table = table.copy()
        table.insert(0, "threshold_quantile", threshold_candidates["threshold_quantile"].to_numpy(dtype=np.float64))
    return table


def _selection_and_final_metric_table(selection_training: ModelTrainingResult, final_training: ModelTrainingResult, test_dataset: Dataset) -> pd.DataFrame:
    X_test = np.nan_to_num(test_dataset.X, nan=0.0, posinf=0.0, neginf=0.0)
    y_test = test_dataset.y
    logistic_probs_test = final_training.bundle.logistic.predict_proba(X_test)[:, 1]
    rows = []
    if "train" in selection_training.summary.get("logit_eval_rows", {}):
        rows.append(selection_training.summary["logit_eval_rows"]["train"])
    if "validation" in selection_training.summary.get("logit_eval_rows", {}):
        rows.append(selection_training.summary["logit_eval_rows"]["validation"])
    rows.append(evaluate_probabilistic_classifier(y_test, logistic_probs_test, "test"))
    return pd.DataFrame(rows)


def _save_tables(results: PipelineResults, report_dir: Path | None = None) -> None:
    reports = report_dir if report_dir is not None else results.paths.reports
    results.fill_surface_ols["coef_table"].to_csv(reports / "fill_surface_ols_coefficients.csv", index=False)
    pd.DataFrame([results.fill_surface_ols["summary"]]).to_csv(reports / "fill_surface_ols_summary.csv", index=False)
    refit_fill_surface_summary = results.training.summary.get("refit_fill_surface")
    if isinstance(refit_fill_surface_summary, dict):
        pd.DataFrame([refit_fill_surface_summary]).to_csv(reports / "refit_fill_surface_summary.csv", index=False)
    results.fill_prob_gates.to_csv(reports / "fill_prob_gates.csv", index=False)
    results.training.logistic_coefficients.to_csv(reports / "logistic_coefficients.csv", index=False)
    feature_audit = results.training.summary.get("feature_audit")
    if isinstance(feature_audit, pd.DataFrame):
        feature_audit.to_csv(reports / "feature_audit.csv", index=False)
    pd.DataFrame({"log1p_feature": results.training.summary.get("log1p_features", [])}).to_csv(
        reports / "log1p_features.csv", index=False
    )
    pd.DataFrame(results.training.summary.get("interaction_pairs", []), columns=["left_feature", "right_feature"]).to_csv(
        reports / "logistic_interactions.csv", index=False
    )
    results.logistic_metrics.to_csv(reports / "logistic_train_val_test_metrics.csv", index=False)
    results.logistic_validation_backtest.to_csv(reports / "logistic_validation_backtest.csv", index=False)
    results.logistic_test_threshold_backtest.to_csv(reports / "logistic_test_threshold_backtest.csv", index=False)
    results.logistic_shadow_test_threshold_backtest.to_csv(
        reports / "logistic_shadow_test_threshold_backtest.csv", index=False
    )
    selected_daily_path = reports / "selected_test_daily_pnl.csv"
    if not results.selected_test_daily_pnl.empty:
        results.selected_test_daily_pnl.to_csv(selected_daily_path, index=False)
    elif selected_daily_path.exists():
        selected_daily_path.unlink()
    pd.DataFrame(
        [
            {
                "calibration_source_mode": results.training.summary.get("logit_validation_backtest_selection", {}).get(
                    "calibration_source_mode"
                ),
                "calibration_window_name": results.training.summary.get("logit_validation_backtest_selection", {}).get(
                    "calibration_window_name"
                ),
                "selected_threshold": results.logistic_selected_threshold,
                "selected_threshold_quantile": results.logistic_selected_threshold_quantile,
                "validation_selected_threshold": results.logistic_validation_selected_threshold,
                "criterion": results.training.summary.get("logit_validation_backtest_selection", {}).get("criterion"),
                "dataset_gate_train": float(
                    results.fill_prob_gates.loc[results.fill_prob_gates["name"] == "dataset_gate_train", "threshold"].iloc[0]
                ),
                "trading_gate_validation": float(
                    results.fill_prob_gates.loc[results.fill_prob_gates["name"] == "trading_gate_validation", "threshold"].iloc[0]
                ),
                "dataset_gate_refit": float(
                    results.fill_prob_gates.loc[results.fill_prob_gates["name"] == "dataset_gate_refit", "threshold"].iloc[0]
                ),
                "trading_gate_test": float(
                    results.fill_prob_gates.loc[results.fill_prob_gates["name"] == "trading_gate_test", "threshold"].iloc[0]
                ),
            }
        ]
    ).to_csv(reports / "selection_summary.csv", index=False)
    test_path = reports / "logistic_test_backtest.csv"
    if not results.logistic_test_backtest.empty:
        results.logistic_test_backtest.to_csv(test_path, index=False)
    elif test_path.exists():
        test_path.unlink()
    shadow_test_path = reports / "logistic_shadow_test_backtest.csv"
    if not results.logistic_shadow_test_backtest.empty:
        results.logistic_shadow_test_backtest.to_csv(shadow_test_path, index=False)
    elif shadow_test_path.exists():
        shadow_test_path.unlink()


def _write_console_report(results: PipelineResults, report_dir: Path | None = None) -> None:
    reports = report_dir if report_dir is not None else results.paths.reports
    (reports / "console_report.txt").write_text(build_console_report(results) + "\n", encoding="utf-8")


def _window_summary_row(spec: SlidingWindowSpec, results: PipelineResults) -> dict[str, object]:
    selected = results.logistic_test_backtest.iloc[0].to_dict() if not results.logistic_test_backtest.empty else {}
    gates = results.fill_prob_gates.set_index("name")
    train_fill_surface_summary = results.fill_surface_ols.get("summary", {})
    refit_fill_surface_summary = results.training.summary.get("refit_fill_surface", {})
    selection_meta = results.training.summary.get("logit_validation_backtest_selection", {})
    return {
        "window_index": spec.index,
        "window_name": spec.name,
        "train_start": spec.train_keys[0],
        "train_end": spec.train_keys[-1],
        "validation_start": spec.validation_keys[0],
        "validation_end": spec.validation_keys[-1],
        "test_start": spec.test_keys[0],
        "test_end": spec.test_keys[-1],
        "calibration_source_mode": selection_meta.get("calibration_source_mode"),
        "calibration_window_name": selection_meta.get("calibration_window_name"),
        "selected_threshold": results.logistic_selected_threshold,
        "selected_threshold_quantile": results.logistic_selected_threshold_quantile,
        "validation_selected_threshold": results.logistic_validation_selected_threshold,
        "dataset_gate_train": float(gates.loc["dataset_gate_train", "threshold"]),
        "trading_gate_validation": float(gates.loc["trading_gate_validation", "threshold"]),
        "dataset_gate_refit": float(gates.loc["dataset_gate_refit", "threshold"]),
        "trading_gate_test": float(gates.loc["trading_gate_test", "threshold"]),
        "fill_surface_cap_mode": train_fill_surface_summary.get("max_notional_cap_mode", ""),
        "fill_surface_cap_quantile": train_fill_surface_summary.get("max_notional_cap_quantile", np.nan),
        "fill_surface_cap_train": train_fill_surface_summary.get("max_notional_cap", np.nan),
        "fill_surface_cap_keep_share_train": train_fill_surface_summary.get("cap_keep_share", np.nan),
        "fill_surface_cap_refit": refit_fill_surface_summary.get("max_notional_cap", np.nan),
        "fill_surface_cap_keep_share_refit": refit_fill_surface_summary.get("cap_keep_share", np.nan),
        HBT_STAT_NAMES["sharpe"]: selected.get(HBT_STAT_NAMES["sharpe"], np.nan),
        HBT_STAT_NAMES["sortino"]: selected.get(HBT_STAT_NAMES["sortino"], np.nan),
        HBT_STAT_NAMES["return"]: selected.get(HBT_STAT_NAMES["return"], np.nan),
        HBT_STAT_NAMES["annual_return"]: selected.get(HBT_STAT_NAMES["annual_return"], np.nan),
        HBT_STAT_NAMES["max_drawdown"]: selected.get(HBT_STAT_NAMES["max_drawdown"], np.nan),
        HBT_STAT_NAMES["return_over_mdd"]: selected.get(HBT_STAT_NAMES["return_over_mdd"], np.nan),
        HBT_STAT_NAMES["daily_trades"]: selected.get(HBT_STAT_NAMES["daily_trades"], np.nan),
    }


def _run_single_window_pipeline(
    paths: ProjectPaths,
    prepared: PreparedData,
    market_cfg,
    hbt_cfg,
    feature_cfg,
    model_cfg,
    strategy_cfg,
    *,
    calibration_table: pd.DataFrame | None = None,
    calibration_source: CalibrationSource | None = None,
    force_rescore_signals: bool,
) -> PipelineResults:
    print("Building train-only fill surface...")
    train_orders = _resolved_orders_for_keys(prepared, paths.train_keys)
    train_filled = train_orders["status"] == FILLED
    train_near, train_opp = near_opp_notional(train_orders)
    train_fill_surface_cap, train_fill_surface_cap_quantile, train_fill_surface_cap_mode = _resolved_fill_surface_cap(
        train_near,
        train_opp,
        model_cfg,
    )
    fill_surface = build_empirical_fill_surface(
        train_near,
        train_opp,
        train_filled,
        bins=model_cfg.fill_surface_bins,
        quantile=model_cfg.fill_surface_quantile,
        max_notional_cap=train_fill_surface_cap,
    )
    fill_surface_ols = fit_fill_surface_ols(fill_surface)
    fill_surface_ols["summary"].update(
        _fill_surface_summary(
            fill_surface,
            raw_order_count=int(train_filled.shape[0]),
            fill_surface_quantile=model_cfg.fill_surface_quantile,
            cap_mode=train_fill_surface_cap_mode,
            cap_quantile=train_fill_surface_cap_quantile,
        )
    )
    dataset_gate_train = _select_dataset_gate(train_orders, fill_surface, model_cfg, name="dataset_gate_train")
    validation_trading_gate = _select_trading_gate(
        prepared,
        paths.train_keys,
        fill_surface,
        model_cfg,
        name="trading_gate_validation",
    )

    print("Building train / validation / test datasets...")
    train_dataset = merge_datasets(_datasets_for_keys(prepared, paths.train_keys, feature_cfg, fill_surface, dataset_gate_train.threshold))
    validation_dataset = merge_datasets(
        _datasets_for_keys(prepared, paths.validation_keys, feature_cfg, fill_surface, dataset_gate_train.threshold)
    )

    print("Fitting logistic regression model...")
    selection_training = fit_reversal_models(train_dataset, validation_dataset, model_cfg, fill_surface)

    calibration = _resolve_calibration_selection(
        selection_training.bundle,
        prepared,
        paths,
        market_cfg,
        hbt_cfg,
        strategy_cfg,
        model_cfg,
        feature_cfg,
        dataset_gate_train.threshold,
        validation_trading_gate.threshold,
        calibration_table=calibration_table,
        calibration_source=calibration_source,
        force=force_rescore_signals,
    )
    logistic_validation_backtest = calibration.validation_backtest
    validation_threshold_candidates = calibration.threshold_candidates
    logistic_threshold_grid = calibration.threshold_grid
    logistic_selected_threshold_quantile = calibration.selected_threshold_quantile
    logistic_validation_selected_threshold = calibration.validation_selected_threshold
    selection_training.bundle.selected_threshold = logistic_validation_selected_threshold
    selection_training.summary["selected_threshold_quantile"] = logistic_selected_threshold_quantile
    selection_training.summary["selected_threshold"] = logistic_validation_selected_threshold
    selection_training.summary["logit_validation_backtest_selection"] = {
        "selected_threshold_quantile": calibration.selected_threshold_quantile,
        "validation_selected_threshold": calibration.validation_selected_threshold,
        "calibration_selected_threshold": calibration.calibration_selected_threshold,
        "selected_row": calibration.selected_row,
        "criterion": calibration.criterion,
        "calibration_source_mode": calibration.source.mode,
        "calibration_window_name": calibration.source.window_name,
        "calibration_window_index": calibration.source.window_index,
        "calibration_day_keys": calibration.source.day_keys,
        "calibration_report_path": calibration.source.report_path,
    }
    print(
        "Calibration threshold selection from HftBacktest: "
        f"logistic_quantile={logistic_selected_threshold_quantile}, "
        f"validation_threshold={logistic_validation_selected_threshold}"
    )

    refit_keys = _refit_keys_for_test(paths)
    refit_days = len(refit_keys)
    print(
        "Refitting fill surface and logistic model on the most recent pre-test days "
        f"(last {refit_days} pre-test days)..."
    )
    refit_orders = _resolved_orders_for_keys(prepared, refit_keys)
    refit_filled = refit_orders["status"] == FILLED
    refit_near, refit_opp = near_opp_notional(refit_orders)
    refit_fill_surface_cap, refit_fill_surface_cap_quantile, refit_fill_surface_cap_mode = _resolved_fill_surface_cap(
        refit_near,
        refit_opp,
        model_cfg,
    )
    final_fill_surface = build_empirical_fill_surface(
        refit_near,
        refit_opp,
        refit_filled,
        bins=model_cfg.fill_surface_bins,
        quantile=model_cfg.fill_surface_quantile,
        max_notional_cap=refit_fill_surface_cap,
    )
    dataset_gate_refit = _remap_dataset_gate(
        refit_orders,
        final_fill_surface,
        model_cfg,
        dataset_gate_train,
        name="dataset_gate_refit",
    )
    test_trading_gate = _remap_trading_gate(
        prepared,
        refit_keys,
        final_fill_surface,
        model_cfg,
        validation_trading_gate,
        name="trading_gate_test",
    )
    refit_dataset = merge_datasets(
        _datasets_for_keys(prepared, refit_keys, feature_cfg, final_fill_surface, dataset_gate_refit.threshold)
    )
    final_training = fit_reversal_models(refit_dataset, None, model_cfg, final_fill_surface)
    final_training.summary["refit_fill_surface"] = _fill_surface_summary(
        final_fill_surface,
        raw_order_count=int(refit_filled.shape[0]),
        fill_surface_quantile=model_cfg.fill_surface_quantile,
        cap_mode=refit_fill_surface_cap_mode,
        cap_quantile=refit_fill_surface_cap_quantile,
    )
    refit_threshold_signal_map = _signal_map_for_keys(
        final_training.bundle,
        prepared,
        paths,
        refit_keys,
        _threshold_reference_signal_dir(paths),
        feature_cfg,
        dataset_gate_refit.threshold,
        force=force_rescore_signals,
    )
    refit_threshold_candidates = _threshold_candidates_from_signal_map(refit_threshold_signal_map, model_cfg.threshold_quantiles)
    logistic_selected_threshold = _threshold_for_quantile(
        refit_threshold_candidates,
        logistic_selected_threshold_quantile,
    )
    final_training.bundle.selected_threshold = logistic_selected_threshold
    final_training.summary["selected_threshold_quantile"] = logistic_selected_threshold_quantile
    final_training.summary["selected_threshold"] = logistic_selected_threshold
    final_training.summary["logit_validation_backtest_selection"] = dict(
        selection_training.summary.get("logit_validation_backtest_selection", {})
    )
    final_training.summary["logit_validation_backtest_selection"]["selected_threshold_quantile"] = (
        logistic_selected_threshold_quantile
    )
    final_training.summary["logit_validation_backtest_selection"]["validation_selected_threshold"] = (
        logistic_validation_selected_threshold
    )
    final_training.summary["logit_validation_backtest_selection"]["selected_threshold"] = logistic_selected_threshold
    final_training.summary["fill_prob_gates"] = {
        gate.name: asdict(gate)
        for gate in (dataset_gate_train, validation_trading_gate, dataset_gate_refit, test_trading_gate)
    }

    print("Scoring test split for classifier metrics...")
    test_dataset = merge_datasets(
        _datasets_for_keys(prepared, paths.test_keys, feature_cfg, final_fill_surface, dataset_gate_refit.threshold)
    )
    logistic_metrics = _selection_and_final_metric_table(selection_training, final_training, test_dataset)

    print("Running test-days threshold sweep backtests...")
    logistic_test_threshold_backtest = _test_threshold_backtest_table(
        final_training.bundle,
        prepared,
        paths,
        market_cfg,
        hbt_cfg,
        strategy_cfg,
        model_cfg,
        feature_cfg,
        refit_threshold_candidates,
        tuple(float(x) for x in refit_threshold_candidates["threshold"].astype(np.float64).to_numpy()),
        dataset_gate_refit.threshold,
        test_trading_gate.threshold,
        force=force_rescore_signals,
    )

    print("Running shadow consistency sweep with validation-stage policy objects on test days...")
    logistic_shadow_test_threshold_backtest = _test_threshold_backtest_table(
        selection_training.bundle,
        prepared,
        paths,
        market_cfg,
        hbt_cfg,
        strategy_cfg,
        model_cfg,
        feature_cfg,
        validation_threshold_candidates,
        logistic_threshold_grid,
        dataset_gate_train.threshold,
        validation_trading_gate.threshold,
        force=force_rescore_signals,
        signal_dir=_shadow_consistency_signal_dir(paths),
    )

    print("Running selected-threshold test balanced-inventory backtests...")
    logistic_test_backtest = _test_backtest_table(
        logistic_selected_threshold,
        logistic_selected_threshold_quantile,
        final_training.bundle,
        prepared,
        paths,
        market_cfg,
        hbt_cfg,
        strategy_cfg,
        model_cfg,
        feature_cfg,
        dataset_gate_refit.threshold,
        test_trading_gate.threshold,
        force=force_rescore_signals,
    )
    logistic_shadow_test_backtest = _test_backtest_table(
        logistic_validation_selected_threshold,
        logistic_selected_threshold_quantile,
        selection_training.bundle,
        prepared,
        paths,
        market_cfg,
        hbt_cfg,
        strategy_cfg,
        model_cfg,
        feature_cfg,
        dataset_gate_train.threshold,
        validation_trading_gate.threshold,
        force=force_rescore_signals,
        signal_dir=_shadow_consistency_signal_dir(paths),
    )
    selected_test_evaluation = _selected_test_evaluation(
        logistic_selected_threshold,
        final_training.bundle,
        prepared,
        paths,
        market_cfg,
        hbt_cfg,
        strategy_cfg,
        feature_cfg,
        dataset_gate_refit.threshold,
        test_trading_gate.threshold,
    )
    selected_test_daily_pnl = (
        _daily_pnl_frame_from_record(
            selected_test_evaluation.record,
            market_cfg,
            window_index=int(paths.reports.name.split("_")[1]),
            window_name=paths.reports.name,
        )
        if selected_test_evaluation is not None
        else pd.DataFrame(columns=["window_index", "window_name", "timestamp", "daily_pnl", "cumulative_window_pnl"])
    )

    print("Saving report tables...")
    gate_summary = pd.DataFrame(
        [asdict(dataset_gate_train), asdict(validation_trading_gate), asdict(dataset_gate_refit), asdict(test_trading_gate)]
    )
    results = PipelineResults(
        paths=paths,
        prepared=prepared,
        fill_surface=final_fill_surface,
        fill_surface_ols=fill_surface_ols,
        training=final_training,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        logistic_metrics=logistic_metrics,
        logistic_validation_backtest=logistic_validation_backtest,
        logistic_test_threshold_backtest=logistic_test_threshold_backtest,
        logistic_test_backtest=logistic_test_backtest,
        logistic_shadow_test_threshold_backtest=logistic_shadow_test_threshold_backtest,
        logistic_shadow_test_backtest=logistic_shadow_test_backtest,
        logistic_selected_threshold=logistic_selected_threshold,
        logistic_selected_threshold_quantile=logistic_selected_threshold_quantile,
        logistic_validation_selected_threshold=logistic_validation_selected_threshold,
        logistic_threshold_grid=logistic_threshold_grid,
        fill_prob_gates=gate_summary,
        selected_test_daily_pnl=selected_test_daily_pnl,
    )
    return results


def run_full_pipeline(
    pipeline_cfg,
    market_cfg,
    hbt_cfg,
    feature_cfg,
    model_cfg,
    strategy_cfg,
    *,
    start_window: int | None = None,
    end_window: int | None = None,
) -> PipelineResults:
    pipeline_cfg, market_cfg, hbt_cfg, feature_cfg, model_cfg, strategy_cfg = synchronize_configs(
        pipeline_cfg, market_cfg, hbt_cfg, feature_cfg, model_cfg, strategy_cfg
    )
    base_paths = build_project_paths(pipeline_cfg, market_cfg)
    window_specs = _sliding_window_specs(base_paths.ordered_keys)
    window_specs_by_index = {int(spec.index): spec for spec in window_specs}
    selected_specs, selected_start, selected_end = _select_window_specs(
        window_specs,
        start_window=start_window,
        end_window=end_window,
    )
    print(
        f"Running sliding-window trial across {len(base_paths.ordered_keys)} day files "
        f"with {len(window_specs)} windows ({WINDOW_TRAIN_DAYS} train / {WINDOW_VALIDATION_DAYS} validation / {WINDOW_TEST_DAYS} test)."
    )
    if len(selected_specs) != len(window_specs):
        print(f"Selected window range: {selected_start} -> {selected_end} ({len(selected_specs)} windows).")

    summary_path = base_paths.reports / "sliding_window_summary.csv"
    summary_rows = _load_existing_summary_rows(summary_path, selected_specs)
    if summary_rows:
        print(f"Loaded {len(summary_rows)} preserved summary rows from {summary_path.name}.")

    last_results: PipelineResults | None = None

    for offset, spec in enumerate(selected_specs, start=1):
        print("")
        print(
            f"[Window {offset}/{len(selected_specs)} | global {spec.index + 1}/{len(window_specs)}] "
            f"train {spec.train_keys[0]} -> {spec.train_keys[-1]}, "
            f"validation {spec.validation_keys[0]} -> {spec.validation_keys[-1]}, "
            f"test {spec.test_keys[0]} -> {spec.test_keys[-1]}"
        )
        window_paths = _window_paths(base_paths, spec)
        previous_spec = window_specs_by_index.get(int(spec.index) - 1)
        calibration_table, calibration_source = _load_previous_test_calibration_table(base_paths, previous_spec, spec)
        print("Preparing window data, continuous state, snapshots, and sampled orders...")
        window_prepared = prepare_market_data(window_paths, market_cfg, hbt_cfg, feature_cfg, strategy_cfg, pipeline_cfg)
        results = _run_single_window_pipeline(
            window_paths,
            window_prepared,
            market_cfg,
            hbt_cfg,
            feature_cfg,
            model_cfg,
            strategy_cfg,
            calibration_table=calibration_table,
            calibration_source=calibration_source,
            force_rescore_signals=pipeline_cfg.force_rescore_signals,
        )
        print("Saving report tables...")
        _save_tables(results, report_dir=window_paths.reports)
        _write_console_report(results, report_dir=window_paths.reports)
        _update_root_pnl_report(base_paths.reports, spec, results.selected_test_daily_pnl)
        summary_rows.append(_window_summary_row(spec, results))
        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty and "window_index" in summary_df.columns:
            summary_df = summary_df.sort_values("window_index").reset_index(drop=True)
        summary_df.to_csv(summary_path, index=False)
        last_results = results

    if last_results is None:
        raise RuntimeError("Sliding-window pipeline produced no results for the selected window range.")
    return last_results


def build_console_report(results: PipelineResults) -> str:
    ols_summary = results.fill_surface_ols["summary"]
    gate_table = results.fill_prob_gates.copy()
    selection_meta = results.training.summary.get("logit_validation_backtest_selection", {})
    calibration_mode = selection_meta.get("calibration_source_mode", "validation_sweep")
    calibration_window_name = selection_meta.get("calibration_window_name")
    calibration_heading = "Validation balanced-inventory backtest (Logistic):"
    if calibration_mode == "previous_test_sweep" and calibration_window_name:
        calibration_heading = f"Calibration threshold sweep reused from {calibration_window_name}:"
    lines = [
        "Fill surface OLS:",
        f"  R^2 = {ols_summary['r_squared']:.6f}",
        format_table(results.fill_surface_ols["coef_table"]),
        "",
        "Fill-probability gates:",
        format_table(gate_table),
        "",
        "Logistic train / validation / test metrics:",
        format_table(results.logistic_metrics),
        "",
        calibration_heading,
        format_table(results.logistic_validation_backtest),
        "",
        f"Validation score-grid thresholds: {results.logistic_threshold_grid}",
        "",
        (
            "Selected threshold quantile: "
            f"logistic={results.logistic_selected_threshold_quantile}, "
            f"validation_threshold={results.logistic_validation_selected_threshold}"
        ),
        "",
        "Shadow consistency threshold sweep (validation-stage model/gates on test days):",
        format_table(results.logistic_shadow_test_threshold_backtest),
        "",
        "Test balanced-inventory threshold sweep (Logistic, refit on pre-test window):",
        format_table(results.logistic_test_threshold_backtest),
        "",
        "Logistic coefficients (refit on pre-test window):",
        format_table(results.training.logistic_coefficients),
        "",
        "Shadow consistency backtest (validation-stage selected threshold on test days):",
        format_table(results.logistic_shadow_test_backtest),
        "",
        f"Applied refit-period threshold for test: logistic={results.logistic_selected_threshold}",
        "",
        "Test balanced-inventory backtest (Logistic, selected threshold, refit on pre-test window):",
        format_table(results.logistic_test_backtest),
    ]
    return "\n".join(lines)
