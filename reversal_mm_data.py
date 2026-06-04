from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hbt_state_sampler import (
    DayState,
    build_day_state_hbt,
    create_hbt_eod_snapshot,
    infer_day_key,
    load_day_state,
    save_day_state,
    subset_start_snapshot,
)
from reversal_mm_simulation import run_sampler
from reversal_mm_utils import ensure_dir, load_hbt_npz


@dataclass(slots=True)
class ProjectPaths:
    project_root: Path
    artifacts: Path
    reports: Path
    state_dir: Path
    snapshot_dir: Path
    sample_dir: Path
    signal_val_dir: Path
    signal_test_dir: Path
    ordered_keys: tuple[str, ...]
    train_keys: tuple[str, ...]
    validation_keys: tuple[str, ...]
    test_keys: tuple[str, ...]
    data_files: dict[str, Path]


@dataclass(slots=True)
class DayWindow:
    local_ts_start: int
    local_ts_end_exclusive: int
    grid_start: int
    grid_end: int


@dataclass(slots=True)
class PreparedData:
    paths: ProjectPaths
    state: DayState
    day_windows: dict[str, DayWindow]
    samples: dict[str, np.ndarray]
    snapshot_after_key: dict[str, str]
    validation_start_snapshot: str | None
    test_start_snapshot: str | None


def build_project_paths(pipeline_cfg, market_cfg) -> ProjectPaths:
    project_root = Path(pipeline_cfg.project_root).resolve()
    artifacts = ensure_dir(project_root / pipeline_cfg.artifact_dirname)
    reports = ensure_dir(artifacts / "reports")
    state_dir = ensure_dir(artifacts / "states")
    snapshot_dir = ensure_dir(artifacts / "snapshots")
    sample_dir = ensure_dir(artifacts / "samples")
    signal_val_dir = ensure_dir(artifacts / "signals_val")
    signal_test_dir = ensure_dir(artifacts / "signals_test")

    data_dir = project_root / "data"
    discovered = sorted((Path(p) for p in data_dir.glob("*.npz")), key=infer_day_key)
    if not discovered:
        raise FileNotFoundError(f"No .npz files found under {data_dir}")
    ordered_keys = tuple(infer_day_key(path) for path in discovered)
    if len(set(ordered_keys)) != len(ordered_keys):
        raise ValueError("Discovered duplicate day keys in data directory.")
    train_keys: tuple[str, ...] = ()
    validation_keys: tuple[str, ...] = ()
    test_keys: tuple[str, ...] = ()

    data_files = {infer_day_key(path): path for path in discovered}
    return ProjectPaths(
        project_root=project_root,
        artifacts=artifacts,
        reports=reports,
        state_dir=state_dir,
        snapshot_dir=snapshot_dir,
        sample_dir=sample_dir,
        signal_val_dir=signal_val_dir,
        signal_test_dir=signal_test_dir,
        ordered_keys=ordered_keys,
        train_keys=train_keys,
        validation_keys=validation_keys,
        test_keys=test_keys,
        data_files=data_files,
    )


def validate_data_files(paths: ProjectPaths) -> dict[str, Path]:
    for key in paths.ordered_keys:
        load_hbt_npz(paths.data_files[key])
    return paths.data_files


def _state_cache_matches(
    state: DayState,
    expected_state_version: str,
    expected_interval_ns: int,
    expected_day_keys: tuple[str, ...],
) -> bool:
    meta = getattr(state, "metadata", {}) or {}
    return (
        meta.get("state_source") == "hftbacktest"
        and meta.get("state_scope") == "continuous"
        and meta.get("touch_change_mode") == "event_feed_exact"
        and meta.get("state_version") == expected_state_version
        and tuple(meta.get("source_day_keys", ())) == tuple(expected_day_keys)
        and int(getattr(state, "interval_ns", -1)) == int(expected_interval_ns)
    )


def _load_or_build_continuous_state(
    paths: ProjectPaths,
    market_cfg,
    hbt_cfg,
    feature_cfg,
    pipeline_cfg,
) -> DayState:
    state_path = paths.state_dir / f"continuous_{pipeline_cfg.state_version}.npz"
    if state_path.exists() and not pipeline_cfg.force_rebuild_states:
        try:
            cached = load_day_state(state_path)
        except Exception as exc:
            print(f"  Cache read failed for {state_path.name}: {exc}. Rebuilding state.")
        else:
            if _state_cache_matches(cached, pipeline_cfg.state_version, feature_cfg.interval_ns, paths.ordered_keys):
                return cached
    data_files = [paths.data_files[key] for key in paths.ordered_keys]
    state = build_day_state_hbt(
        data_files,
        tick_size=market_cfg.tick_size,
        lot_size=market_cfg.lot_size,
        contract_size=market_cfg.contract_size,
        maker_fee=market_cfg.maker_fee,
        taker_fee=market_cfg.taker_fee,
        interval_ns=feature_cfg.interval_ns,
        roi_lb=hbt_cfg.roi_lb,
        roi_ub=hbt_cfg.roi_ub,
        entry_latency_ns=hbt_cfg.entry_latency_ns,
        response_latency_ns=hbt_cfg.response_latency_ns,
        queue_model_power=hbt_cfg.queue_model_power,
        last_trades_capacity=hbt_cfg.last_trades_capacity,
        parallel_load=hbt_cfg.parallel_load,
        half_book_notional_usd=market_cfg.half_book_notional_usd,
        totb_mean_lookback_ns=feature_cfg.totb_mean_lookback_ns,
        engine=hbt_cfg.engine,
        initial_snapshot_path=None,
    )
    state.metadata = dict(state.metadata)
    state.metadata["state_version"] = pipeline_cfg.state_version
    state.metadata["state_scope"] = "continuous"
    state.metadata["source_day_keys"] = tuple(paths.ordered_keys)
    save_day_state(state_path, state)
    return state


def _build_day_windows(paths: ProjectPaths, state: DayState) -> dict[str, DayWindow]:
    raw_bounds: dict[str, tuple[int, int]] = {}
    for key in paths.ordered_keys:
        day_data = load_hbt_npz(paths.data_files[key])
        if day_data.shape[0] == 0:
            raise ValueError(f"Day file {paths.data_files[key].name} is empty.")
        local_ts = day_data["local_ts"].astype(np.int64)
        raw_bounds[key] = (int(np.min(local_ts)), int(np.max(local_ts)))

    max_i64 = np.iinfo(np.int64).max
    windows: dict[str, DayWindow] = {}
    for idx, key in enumerate(paths.ordered_keys):
        start_ts, end_ts = raw_bounds[key]
        if idx + 1 < len(paths.ordered_keys):
            next_key = paths.ordered_keys[idx + 1]
            end_exclusive = int(raw_bounds[next_key][0])
        else:
            end_exclusive = int(end_ts + 1) if end_ts < max_i64 else int(end_ts)
        grid_start = int(np.searchsorted(state.grid_times, start_ts, side="left"))
        grid_end = int(np.searchsorted(state.grid_times, end_exclusive, side="left"))
        windows[key] = DayWindow(
            local_ts_start=int(start_ts),
            local_ts_end_exclusive=int(max(end_exclusive, start_ts)),
            grid_start=max(grid_start, 0),
            grid_end=max(grid_end, grid_start),
        )
    return windows


def _load_or_sample_continuous_orders(
    paths: ProjectPaths,
    market_cfg,
    hbt_cfg,
    strategy_cfg,
    pipeline_cfg,
) -> np.ndarray:
    sample_path = paths.sample_dir / f"continuous_{pipeline_cfg.sample_version}.npz"
    if sample_path.exists() and not pipeline_cfg.force_resample_orders:
        try:
            with np.load(sample_path, allow_pickle=False) as zf:
                cached_keys = tuple(str(x) for x in zf["ordered_keys"].tolist()) if "ordered_keys" in zf.files else ()
                if cached_keys != paths.ordered_keys:
                    raise ValueError(
                        f"sample cache keys {cached_keys!r} do not match configured keys {paths.ordered_keys!r}"
                    )
                return zf["orders"]
        except Exception as exc:
            print(f"  Cache read failed for {sample_path.name}: {exc}. Resampling orders.")
    data_files = [str(paths.data_files[key]) for key in paths.ordered_keys]
    orders = run_sampler(
        data_files,
        market_cfg,
        hbt_cfg,
        order_qty=strategy_cfg.order_qty,
        snapshot_path=None,
        mdl_ns=hbt_cfg.market_data_latency_ns,
    )
    np.savez_compressed(sample_path, orders=orders, ordered_keys=np.asarray(paths.ordered_keys, dtype="U16"))
    return orders


def _split_orders_by_day(orders: np.ndarray, day_windows: dict[str, DayWindow], ordered_keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    if orders.size == 0:
        return {key: np.empty(0, dtype=orders.dtype) for key in ordered_keys}
    post_ts = orders["post_local_ts"].astype(np.int64)
    samples: dict[str, np.ndarray] = {}
    for key in ordered_keys:
        window = day_windows[key]
        mask = (post_ts >= window.local_ts_start) & (post_ts < window.local_ts_end_exclusive)
        samples[key] = orders[mask]
    return samples


def _build_snapshot_after_key(paths: ProjectPaths, market_cfg) -> dict[str, str]:
    snapshot_after_key: dict[str, str] = {}
    prev_snapshot: str | None = None
    for key in paths.ordered_keys:
        prev_snapshot = create_hbt_eod_snapshot(
            paths.snapshot_dir / f"snap_{key}.npz",
            paths.data_files[key],
            tick_size=market_cfg.tick_size,
            lot_size=market_cfg.lot_size,
            initial_snapshot_path=prev_snapshot,
            force=False,
        )
        snapshot_after_key[key] = prev_snapshot
    return snapshot_after_key


def prepare_market_data(paths: ProjectPaths, market_cfg, hbt_cfg, feature_cfg, strategy_cfg, pipeline_cfg) -> PreparedData:
    print("Validating input day files...")
    validate_data_files(paths)

    print("Loading/building continuous market state across all configured day files...")
    state = _load_or_build_continuous_state(paths, market_cfg, hbt_cfg, feature_cfg, pipeline_cfg)

    print("Mapping configured day windows into the continuous state...")
    day_windows = _build_day_windows(paths, state)

    print("Loading/sampling continuous maker-order stream across all configured day files...")
    all_orders = _load_or_sample_continuous_orders(paths, market_cfg, hbt_cfg, strategy_cfg, pipeline_cfg)
    samples = _split_orders_by_day(all_orders, day_windows, paths.ordered_keys)

    print("Preparing end-of-day snapshots for subset backtests...")
    snapshot_after_key = _build_snapshot_after_key(paths, market_cfg)

    validation_start_snapshot = subset_start_snapshot(paths.ordered_keys, paths.validation_keys, snapshot_after_key)
    test_start_snapshot = subset_start_snapshot(paths.ordered_keys, paths.test_keys, snapshot_after_key)

    return PreparedData(
        paths=paths,
        state=state,
        day_windows=day_windows,
        samples=samples,
        snapshot_after_key=snapshot_after_key,
        validation_start_snapshot=validation_start_snapshot,
        test_start_snapshot=test_start_snapshot,
    )
