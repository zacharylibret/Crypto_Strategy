from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class MarketConfig:
    venue_prefix: str = "binance-futures"
    symbol: str = "BTCUSDT"
    data_year: str = "2026"
    tick_size: float = 0.1
    lot_size: float = 0.001
    contract_size: float = 1.0
    maker_fee: float = -0.00005
    taker_fee: float = 0.00015
    half_book_notional_usd: float = 500_000.0


@dataclass(slots=True)
class HbtConfig:
    engine: str = "roi_vector"
    roi_lb: float = 50_000.0
    roi_ub: float = 120_000.0
    control_interval_ns: int = 80_000_000
    market_data_latency_ns: int = 3_000_000
    entry_latency_ns: int = 3_000_000
    response_latency_ns: int = 3_000_000
    queue_model_power: int = 2
    parallel_load: bool = True
    last_trades_capacity: int = 100_000
    allow_dual_quote_same_step: bool = True


@dataclass(slots=True)
class FeatureConfig:
    interval_ns: int | None = None
    scales_ns: list[int] = field(default_factory=list)
    volatility_return_interval_ns: int = 10_000_000_000
    volatility_lookbacks: list[int] = field(default_factory=lambda: [100, 500])
    totb_mean_lookback_ns: int = 600_000_000_000
    impute_missing_with_zero: bool = True
    score_chunk_size: int = 50_000


@dataclass(slots=True)
class ModelConfig:
    fill_surface_bins: int = 20
    fill_surface_quantile: float = 0.99
    fill_surface_notional_cap_quantile: float = 0.96
    fill_prob_threshold: float = 0.82
    dataset_gate_quantile_low: float = 0.6
    dataset_gate_quantile_high: float = 0.8
    dataset_gate_min_orders: int = 250
    dataset_gate_min_fills: int = 25
    trading_gate_quantile_low: float = 0.90
    trading_gate_quantile_high: float = 0.98
    trading_gate_min_orders: int = 250
    logistic_max_iter: int = 4000
    threshold_quantiles: tuple[float, ...] = tuple(np.concatenate([np.arange(0.8, 0.96, 0.05), np.array([0.975, 0.98, 0.985, 0.855, 0.9, 0.935, 0.95, 0.975, 0.98, 0.985, 0.987, 0.99, 1.0])]))
    selection_threshold_quantile_floor: float = 0.945
    min_validation_roundtrips: int = 50
    min_validation_daily_trades: float = 50


@dataclass(slots=True)
class StrategyConfig:
    order_qty: float = 0.001
    imbalance_post_threshold: float = 0.5
    imbalance_cancel_threshold: float = 0.0
    model_threshold: float = 0.24
    enable_fill_prob_gate: bool = True


@dataclass(slots=True)
class PipelineConfig:
    project_root: Path = Path(".")
    artifact_dirname: str = "artifacts"
    state_version: str = ""
    sample_version: str = ""
    force_rebuild_states: bool = False
    force_resample_orders: bool = False
    force_rescore_signals: bool = True


DEFAULT_HORIZON_SCALES_NS = (
    1_000_000_000,
    5_000_000_000,
    30_000_000_000,
    300_000_000_000,
)


def _config_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _resolved_scales(feature_cfg: FeatureConfig, hbt_cfg: HbtConfig) -> list[int]:
    interval_ns = int(hbt_cfg.control_interval_ns)
    tail = [int(x) for x in (feature_cfg.scales_ns[1:] if feature_cfg.scales_ns else DEFAULT_HORIZON_SCALES_NS)]
    return [interval_ns, *tail]


def synchronize_configs(
    pipeline_cfg: PipelineConfig,
    market_cfg: MarketConfig,
    hbt_cfg: HbtConfig,
    feature_cfg: FeatureConfig,
    model_cfg: ModelConfig | None = None,
    strategy_cfg: StrategyConfig | None = None,
) -> tuple[PipelineConfig, MarketConfig, HbtConfig, FeatureConfig, ModelConfig | None, StrategyConfig | None]:
    feature_cfg.interval_ns = int(hbt_cfg.control_interval_ns)
    feature_cfg.scales_ns = _resolved_scales(feature_cfg, hbt_cfg)

    state_payload = {
        "venue_prefix": market_cfg.venue_prefix,
        "symbol": market_cfg.symbol,
        "data_year": market_cfg.data_year,
        "tick_size": market_cfg.tick_size,
        "lot_size": market_cfg.lot_size,
        "contract_size": market_cfg.contract_size,
        "maker_fee": market_cfg.maker_fee,
        "taker_fee": market_cfg.taker_fee,
        "half_book_notional_usd": market_cfg.half_book_notional_usd,
        "engine": hbt_cfg.engine,
        "roi_lb": hbt_cfg.roi_lb,
        "roi_ub": hbt_cfg.roi_ub,
        "control_interval_ns": hbt_cfg.control_interval_ns,
        "entry_latency_ns": hbt_cfg.entry_latency_ns,
        "response_latency_ns": hbt_cfg.response_latency_ns,
        "queue_model_power": hbt_cfg.queue_model_power,
        "parallel_load": hbt_cfg.parallel_load,
        "last_trades_capacity": hbt_cfg.last_trades_capacity,
        "interval_ns": feature_cfg.interval_ns,
        "scales_ns": feature_cfg.scales_ns,
        "volatility_return_interval_ns": feature_cfg.volatility_return_interval_ns,
        "volatility_lookbacks": feature_cfg.volatility_lookbacks,
        "totb_mean_lookback_ns": feature_cfg.totb_mean_lookback_ns,
        "impute_missing_with_zero": feature_cfg.impute_missing_with_zero,
    }
    sample_payload = {
        **state_payload,
        "sampler_logic_version": "same_step_replace_v1",
        "market_data_latency_ns": hbt_cfg.market_data_latency_ns,
        "allow_dual_quote_same_step": hbt_cfg.allow_dual_quote_same_step,
        "order_qty": None if strategy_cfg is None else strategy_cfg.order_qty,
    }
    pipeline_cfg.state_version = f"state_{_config_hash(state_payload)}"
    pipeline_cfg.sample_version = f"samples_{_config_hash(sample_payload)}"
    return pipeline_cfg, market_cfg, hbt_cfg, feature_cfg, model_cfg, strategy_cfg


def default_configs(project_root: str | Path = ".") -> tuple[
    PipelineConfig,
    MarketConfig,
    HbtConfig,
    FeatureConfig,
    ModelConfig,
    StrategyConfig,
]:
    root = Path(project_root).resolve()
    pipeline_cfg = PipelineConfig(project_root=root)
    market_cfg = MarketConfig()
    hbt_cfg = HbtConfig()
    feature_cfg = FeatureConfig()
    model_cfg = ModelConfig()
    strategy_cfg = StrategyConfig()
    synchronize_configs(pipeline_cfg, market_cfg, hbt_cfg, feature_cfg, model_cfg, strategy_cfg)
    return pipeline_cfg, market_cfg, hbt_cfg, feature_cfg, model_cfg, strategy_cfg
