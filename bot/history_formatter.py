"""
bot/history_formatter.py - Compact Mobile-Friendly HTML Trade History Formatter

Renders Telegram trade history messages optimized for ~38-char mobile screens using
HTML parse_mode and <pre> fixed-width columns. All dynamic strings are escaped
with html.escape().
"""

import html
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# Default fallback symbol mapping when broker asset cache is unavailable
DEFAULT_SYMBOL_MAP: Dict[str, str] = {
    "1": "EURUSD",
    "2": "GBPUSD",
    "4": "EURJPY",
    "5": "GBPUSD",
    "6": "USDJPY",
    "8": "NZDUSD",
    "10": "EURGBP",
    "54": "XAGUSD",
    "74": "XAUUSD",
    "99": "AUDUSD",
    "100": "USDCAD",
    "105": "NZDUSD",
    "108": "USDCHF",
    "212": "EURCAD",
    "816": "BTCUSD",
    "824": "ETHUSD",
    "836": "XRPUSD",
    "886": "SOLUSD",
    "969": "BRENT",
    "971": "WTI",
    "1470": "US500",
    "1471": "NAS100",
    "1487": "XAGUSD",
    "1979": "BCHUSD",
}

_NAME_NORMALIZATIONS: Dict[str, str] = {
    "GOLD": "XAUUSD",
    "XAU/USD": "XAUUSD",
    "GOLD (XAUUSD)": "XAUUSD",
    "SILVER": "XAGUSD",
    "XAG/USD": "XAGUSD",
    "SILVER (XAGUSD)": "XAGUSD",
    "US 100": "NAS100",
    "US100": "NAS100",
    "NAS 100": "NAS100",
    "US 500": "US500",
    "BITCOIN": "BTCUSD",
    "BITCOIN (BTCUSD)": "BTCUSD",
    "BTC/USD": "BTCUSD",
    "ETHEREUM": "ETHUSD",
    "ETHEREUM (ETHUSD)": "ETHUSD",
    "ETH/USD": "ETHUSD",
    "CRUDE OIL (BRENT)": "BRENT",
    "CRUDE OIL (WTI)": "WTI",
}


def build_symbol_map_from_assets(assets: Optional[List[Dict[str, Any]]]) -> Dict[str, str]:
    """Build an {asset_id: ticker} lookup map from broker asset metadata."""
    mapping: Dict[str, str] = {}
    for asset in (assets or []):
        aid = asset.get("asset_id") or asset.get("id")
        name = asset.get("name") or asset.get("ticker") or ""
        if aid is not None and name:
            mapping[str(aid)] = normalize_symbol_label(str(name))
    return mapping


def normalize_symbol_label(raw_name: str) -> str:
    """Normalize verbose broker instrument labels to compact uppercase tickers."""
    cleaned = raw_name.strip().upper()
    if cleaned in _NAME_NORMALIZATIONS:
        return _NAME_NORMALIZATIONS[cleaned]
    # Remove "(OTC)", slashes, hyphens, and spaces
    cleaned_no_otc = re.sub(r"\s*\(OTC\)\s*", "", cleaned).strip()
    if cleaned_no_otc in _NAME_NORMALIZATIONS:
        return _NAME_NORMALIZATIONS[cleaned_no_otc]
    compact = cleaned_no_otc.replace("/", "").replace("-", "").replace(" ", "")
    return compact or "UNKNOWN"


def resolve_symbol_name(raw_symbol: Any, symbol_map: Optional[Dict[str, str]] = None) -> str:
    """
    Resolve numeric broker asset IDs using symbol_map (cached broker map).
    Returns 'ID<num>' if a numeric ID cannot be resolved.
    """
    s = str(raw_symbol if raw_symbol is not None else "UNKNOWN").strip()
    if s.lower().startswith("asset #"):
        s = s[7:].strip()
    elif s.startswith("#"):
        s = s[1:].strip()

    if s.isdigit():
        if symbol_map and s in symbol_map and symbol_map[s]:
            return normalize_symbol_label(symbol_map[s])
        return f"ID{s}"

    return normalize_symbol_label(s)


def format_strategy_group_name(raw_strategy: Optional[str]) -> str:
    """Map internal copier/strategy identifiers to readable group header names."""
    s = str(raw_strategy or "Unknown").strip()
    low = s.lower()
    if "callisto" in low:
        return "Gold Zone"
    if "gold" in low and "breakout" not in low and "swing" not in low:
        return "Gold Scalp"
    if "gsociety" in low or "society" in low:
        return "Gold Breakout"
    if "king" in low:
        return "Gold Swing"
    if "polycarp" in low:
        return "Blitz Option"
    if low in ("cfd", "copier", "copiers"):
        return "Gold Setup"
    if low in ("manual", "unattributed", "manual/unattributed"):
        return "Manual/Unattributed"
    return s


def map_exit_reason(reason: Optional[str]) -> str:
    """Convert verbose exit reasons into 2-3 char codes: TP, SL, BE, MAN."""
    r = str(reason or "").strip().lower()
    if any(k in r for k in ("take_profit", "tp1", "tp2", "tp")) and "stop" not in r:
        return "TP"
    if any(k in r for k in ("stop_loss", "stop_lose", "margin_erase", "margin", "sl")):
        return "SL"
    if any(k in r for k in ("breakeven", "break_even")) or r == "be":
        return "BE"
    return "MAN"


def _parse_timestamp_parts(ts_raw: Any) -> Tuple[str, float]:
    """Return ('HH:MM', epoch_seconds) from ISO, SQL, or time-only strings."""
    s = str(ts_raw or "").strip()
    if not s or s == "--":
        return "--:--", 0.0

    # Time-only e.g. '08:24:22'
    if re.fullmatch(r"\d{2}:\d{2}(:\d{2})?", s):
        hhmm = s[:5]
        parts = s.split(":")
        secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + (int(parts[2]) if len(parts) > 2 else 0)
        return hhmm, float(secs)

    norm = s.replace("T", " ").replace("Z", "").replace(" UTC", "").strip()
    # Match YYYY-MM-DD HH:MM:SS(.ffffff)
    m = re.search(r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})", norm)
    if m:
        dt_str = f"{m.group(1)} {m.group(2)}"
        try:
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            return m.group(2)[:5], dt.timestamp()
        except ValueError:
            return m.group(2)[:5], 0.0

    m_hm = re.search(r"(\d{2}:\d{2})", norm)
    return (m_hm.group(1) if m_hm else "--:--"), 0.0


def _format_lots(lots: float) -> str:
    """Format lot size to 1 decimal without 'L' suffix, compacting large broker unit counts."""
    val = abs(float(lots or 0.0))
    if val >= 1000.0:
        k_val = val / 1000.0
        if k_val >= 10.0 or k_val.is_integer():
            return f"{k_val:.0f}k"
        return f"{k_val:.1f}k"
    if val >= 100.0:
        return f"{val:.0f}"
    return f"{val:.1f}"


def _format_pnl_compact(pnl: float, has_unconfirmed: bool = False) -> str:
    """Format signed PnL without '$' symbol, fitting within 7 chars."""
    if has_unconfirmed:
        return "Pending"
    if abs(pnl) >= 1000.0:
        return f"{pnl:+.0f}"
    return f"{pnl:+.2f}"


def format_history_row(
    icon: str,
    hhmm: str,
    symbol: str,
    side: str,
    lots: float,
    pnl: float,
    exit_reason: str,
    has_unconfirmed: bool = False,
    stack_count: int = 1,
) -> str:
    """Format a single fixed-width trade row (<= 38 chars) for mobile <pre> rendering."""
    sym_col = symbol[:7].ljust(7)
    side_u = str(side or "BUY").upper()
    side_c = "B" if side_u in ("BUY", "LONG", "CALL") else "S"
    lots_col = _format_lots(lots).ljust(4)
    pnl_col = _format_pnl_compact(pnl, has_unconfirmed).rjust(7)
    reason_col = map_exit_reason(exit_reason)
    stack_col = f" ×{stack_count}" if stack_count > 1 else ""
    row = f"{icon} {hhmm} {sym_col} {side_c} {lots_col} {pnl_col} {reason_col}{stack_col}"
    # Guard hard 38-char ceiling if unusually wide stack/lots
    if len(row) > 38 and stack_col:
        row = f"{icon} {hhmm} {sym_col} {side_c} {lots_col} {pnl_col} {reason_col[:2]}{stack_col}"
    return html.escape(row)


def _collapse_stacked_entries(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Display-only grouping for trades closing within 5s on the same symbol, side, and strategy.
    Does not mutate source stats or raw trade dicts.
    """
    if not trades:
        return []

    # Work newest-first so stacked pairs close to each other in time sit adjacent
    ordered = sorted(trades, key=lambda x: x["_ts_sec"], reverse=True)
    collapsed: List[Dict[str, Any]] = []

    for tr in ordered:
        merged = False
        for prev in collapsed:
            if (
                prev["resolved_symbol"] == tr["resolved_symbol"]
                and prev["side_char"] == tr["side_char"]
                and prev["strategy_group"] == tr["strategy_group"]
                and prev["_ts_sec"] > 0
                and tr["_ts_sec"] > 0
                and abs(prev["_ts_sec"] - tr["_ts_sec"]) <= 5.0
            ):
                prev["pnl"] += tr["pnl"]
                prev["lots"] += tr["lots"]
                prev["has_unconfirmed"] = prev["has_unconfirmed"] or tr["has_unconfirmed"]
                prev["stack_count"] += tr.get("stack_count", 1)
                merged = True
                break
        if not merged:
            collapsed.append(dict(tr))

    return collapsed


def format_history_html(
    account_type: str,
    category: str,
    period: str,
    page: int,
    total_pages: int,
    stats: Dict[str, Any],
    trades: List[Dict[str, Any]],
    symbol_map: Optional[Dict[str, str]] = None,
    blitz_summary: Optional[Dict[str, Any]] = None,
    sync_note: str = "",
) -> str:
    """
    Render complete HTML trade history message formatted for a ~38-char mobile screen.

    Includes compact stats header, optional Blitz summary directly beneath the header,
    strategy-grouped sections with subtotal headers, and fixed-width <pre> rows.
    """
    acct_lbl = html.escape((account_type or "training").upper())
    cat_lbl = html.escape("ALL" if (category or "all").lower() == "all" else str(category).upper())
    per_lbl = html.escape("ALL TIME" if (period or "all").lower() == "all" else str(period).upper())

    net_pnl = float(stats.get("net_pnl", 0.0))
    net_icon = "🟢" if net_pnl >= 0 else "🔴"
    net_sign = "+" if net_pnl >= 0 else "-"
    win_rate = float(stats.get("win_rate", 0.0))
    wins = int(stats.get("wins", 0))
    losses = int(stats.get("losses", 0))
    breakevens = int(stats.get("breakevens", 0))
    pf = float(stats.get("profit_factor", 0.0))
    max_dd = abs(float(stats.get("max_drawdown", 0.0)))
    total_trades = int(stats.get("total_trades", 0))
    gross_profit = float(stats.get("gross_profit", 0.0))
    gross_loss = abs(float(stats.get("gross_loss", 0.0)))

    lines: List[str] = [
        f"📜 <b>{acct_lbl} · {cat_lbl} · {per_lbl}</b> ({page}/{total_pages})",
        f"{net_icon} Net <b>{net_sign}${abs(net_pnl):.2f}</b> · 🎯 {win_rate:.1f}% ({wins}W {losses}L {breakevens}BE)",
        f"📈 PF {pf:.2f} · 📉 Max DD -${max_dd:.2f}",
        f"📊 {total_trades} trades (+${gross_profit:.2f} / -${gross_loss:.2f})",
    ]

    if blitz_summary and blitz_summary.get("count", 0) > 0:
        b_pnl = float(blitz_summary.get("pnl", 0.0))
        b_sign = "+" if b_pnl >= 0 else "-"
        b_wins = int(blitz_summary.get("wins", 0))
        b_losses = int(blitz_summary.get("losses", 0))
        lines.append(f"⚡ Blitz: {b_sign}${abs(b_pnl):.2f} ({b_wins}W {b_losses}L)")

    if sync_note:
        lines.append(f"⚠️ <i>{html.escape(sync_note.strip())}</i>")

    if not trades:
        lines.append("\n<i>No closed trades matching this filter.</i>")
        return "\n".join(lines)

    # Normalize trade dicts for grouping & rendering
    normalized: List[Dict[str, Any]] = []
    for t in trades:
        raw_sym = t.get("symbol", "UNKNOWN")
        resolved_sym = resolve_symbol_name(raw_sym, symbol_map=symbol_map)
        side_raw = str(t.get("side", "BUY")).upper()
        side_c = "B" if side_u_is_buy(side_raw) else "S"
        lots_val = float(t.get("total_lots") if t.get("total_lots") is not None else t.get("lots", 1.0))
        pnl_val = float(t.get("pnl") or 0.0)
        has_unconfirmed = bool(t.get("has_unconfirmed_pnl", False) or t.get("pnl") is None)
        close_ts = t.get("close_time_utc") or t.get("close_time") or t.get("time") or ""
        hhmm, ts_sec = _parse_timestamp_parts(close_ts)
        strat_group = format_strategy_group_name(t.get("strategy", "Unknown"))
        exit_reason = str(t.get("exit_reason") or "closed")

        normalized.append({
            "resolved_symbol": resolved_sym,
            "side": side_raw,
            "side_char": side_c,
            "lots": lots_val,
            "pnl": pnl_val,
            "has_unconfirmed": has_unconfirmed,
            "hhmm": hhmm,
            "_ts_sec": ts_sec,
            "strategy_group": strat_group,
            "exit_reason": exit_reason,
            "stack_count": 1,
        })

    # Group trades by strategy header (preserving newest-group-first order)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in sorted(normalized, key=lambda x: x["_ts_sec"], reverse=True):
        groups.setdefault(item["strategy_group"], []).append(item)

    for strat_name, grp_items in groups.items():
        grp_count = len(grp_items)
        grp_pnl = sum(i["pnl"] for i in grp_items)
        grp_sign = "+" if grp_pnl >= 0 else "-"
        trade_word = "trade" if grp_count == 1 else "trades"
        lines.append("")
        lines.append(
            f"<b>{html.escape(strat_name)}</b> · {grp_sign}${abs(grp_pnl):.2f} · {grp_count} {trade_word}"
        )

        display_rows = _collapse_stacked_entries(grp_items)
        pre_lines: List[str] = []
        for row_item in display_rows:
            r_pnl = row_item["pnl"]
            r_unconf = row_item["has_unconfirmed"]
            r_reason = row_item["exit_reason"]
            if r_unconf:
                r_icon = "⏳"
            elif abs(r_pnl) < 0.005 or "breakeven" in r_reason.lower():
                r_icon = "🛡️"
            elif r_pnl > 0:
                r_icon = "✅"
            else:
                r_icon = "❌"

            pre_lines.append(
                format_history_row(
                    icon=r_icon,
                    hhmm=row_item["hhmm"],
                    symbol=row_item["resolved_symbol"],
                    side=row_item["side"],
                    lots=row_item["lots"],
                    pnl=r_pnl,
                    exit_reason=r_reason,
                    has_unconfirmed=r_unconf,
                    stack_count=row_item.get("stack_count", 1),
                )
            )
        lines.append("<pre>" + "\n".join(pre_lines) + "</pre>")

    output = "\n".join(lines)
    if len(output) > 4000:
        output = output[:3940] + "\n<i>...(truncated)</i>"
    return output


def side_u_is_buy(side_raw: str) -> bool:
    return str(side_raw or "").upper() in ("BUY", "LONG", "CALL")


def format_blitz_history_html(
    account_type: str,
    page: int,
    total_pages: int,
    blitz_trades: List[Dict[str, Any]],
    total_count: int,
    wins: int,
    losses: int,
    ties: int,
    net_pnl: float,
    win_rate: float,
) -> str:
    """Render Blitz-only execution ledger in compact mobile HTML format."""
    acct_lbl = html.escape((account_type or "training").upper())
    pnl_icon = "🟢" if net_pnl >= 0 else "🔴"
    pnl_sign = "+" if net_pnl >= 0 else "-"

    lines: List[str] = [
        f"⚡ <b>{acct_lbl} · BLITZ · ALL</b> ({page}/{total_pages})",
        f"{pnl_icon} Net <b>{pnl_sign}${abs(net_pnl):.2f}</b> · 🎯 {win_rate:.1f}% ({wins}W {losses}L {ties}T)",
        f"📊 {total_count} options",
    ]

    if not blitz_trades:
        lines.append("\n<i>No Blitz option trades recorded in session.</i>")
        return "\n".join(lines)

    pre_lines: List[str] = []
    for t in blitz_trades:
        icon = "✅" if t.get("is_win") else ("🛡️" if t.get("is_equal") else "❌")
        hhmm, _ = _parse_timestamp_parts(t.get("time"))
        sym = normalize_symbol_label(str(t.get("asset", "OTC")))[:7].ljust(7)
        side_c = "B" if side_u_is_buy(t.get("direction", "CALL")) else "S"
        stake_s = f"{float(t.get('stake', 0.0)):.1f}".ljust(4)
        pnl_val = float(t.get("pnl", 0.0))
        pnl_s = f"{pnl_val:+.2f}".rjust(7)
        res_code = "WIN" if t.get("is_win") else ("TIE" if t.get("is_equal") else "LOS")
        pre_lines.append(html.escape(f"{icon} {hhmm} {sym} {side_c} {stake_s} {pnl_s} {res_code}"))

    lines.append("")
    lines.append("<pre>" + "\n".join(pre_lines) + "</pre>")
    return "\n".join(lines)
