from __future__ import annotations

import numpy as np

BUY = 1
SELL = -1

NONE = 0
NEW = 1
EXPIRED = 2
FILLED = 3
CANCELED = 4
PARTIALLY_FILLED = 5
REJECTED = 6

GTC = 0
GTX = 1
FOK = 2
IOC = 3

LIMIT = 0
MARKET = 1

ACTIVE_STATUS = {NEW, PARTIALLY_FILLED}
INACTIVE_STATUS = {NONE, EXPIRED, FILLED, CANCELED, REJECTED}

NS_PER_SECOND = 1_000_000_000
BP_SCALE = 10_000.0

EXCH_EVENT = np.uint64(2147483648)
LOCAL_EVENT = np.uint64(1073741824)
BUY_EVENT = np.uint64(536870912)
SELL_EVENT = np.uint64(268435456)

DEPTH_EVENT = 1
TRADE_EVENT = 2
DEPTH_CLEAR_EVENT = 3
DEPTH_SNAPSHOT_EVENT = 4

EVENT_DTYPE = np.dtype(
    [
        ("ev", "<u8"),
        ("exch_ts", "<i8"),
        ("local_ts", "<i8"),
        ("px", "<f8"),
        ("qty", "<f8"),
        ("order_id", "<u8"),
        ("ival", "<i8"),
        ("fval", "<f8"),
    ],
    align=True,
)

SAMPLED_ORDER_DTYPE = np.dtype(
    [
        ("order_id", "<u8"),
        ("side", "i1"),
        ("post_local_ts", "<i8"),
        ("post_price", "<f8"),
        ("post_best_bid", "<f8"),
        ("post_best_ask", "<f8"),
        ("post_best_bid_qty", "<f8"),
        ("post_best_ask_qty", "<f8"),
        ("post_best_bid_notional", "<f8"),
        ("post_best_ask_notional", "<f8"),
        ("terminal_local_ts", "<i8"),
        ("status", "u1"),
        ("fill_local_ts", "<i8"),
        ("fill_price", "<f8"),
    ],
    align=True,
)

ROUNDTRIP_DTYPE = np.dtype(
    [
        ("open_side", "i1"),
        ("open_local_ts", "<i8"),
        ("open_price", "<f8"),
        ("close_local_ts", "<i8"),
        ("close_price", "<f8"),
        ("gross_ret_bp", "<f8"),
        ("net_ret_bp", "<f8"),
        ("holding_s", "<f8"),
    ],
    align=True,
)

MODE_REVERSAL = "reversal_logistic"

HBT_STAT_NAMES = {
    "sharpe": "Sharpe1H365",
    "sortino": "Sortino1H365",
    "return": "ReturnPct",
    "annual_return": "AnnualReturnPct",
    "max_drawdown": "MaxDrawdownPct",
    "return_over_mdd": "ReturnOverMDD",
    "daily_trades": "DailyTrades",
    "daily_trading_value": "DailyTradingValuePct",
    "max_position_value": "MaxPositionValue",
}


def event_type(ev: np.uint64) -> int:
    return int(ev) & 0xFF


def has_flag(ev: np.uint64, flag: np.uint64) -> bool:
    return (int(ev) & int(flag)) == int(flag)
