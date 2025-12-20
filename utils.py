import re
import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)

def load_signals(file_path="signals.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning(f"⚠️ No signals file found at {file_path}")
        return ""

def parse_signals(text: str):
    pattern = re.compile(r"(\d{2}:\d{2});([A-Z]+);(CALL|PUT);(\d+)", re.IGNORECASE)
    signals = []
    for line in text.splitlines():
        match = pattern.search(line.strip())
        if not match:
            continue
        t, asset, direction, expiry = match.groups()
        hh, mm = map(int, t.split(":"))
        sched_time = datetime.combine(date.today(), datetime.min.time()).replace(hour=hh, minute=mm)
        signals.append({
            "time": sched_time,
            "asset": asset,
            "direction": direction.lower(),
            "expiry": int(expiry),
            "line": line.strip()
        })
    return sorted(signals, key=lambda x: x["time"])
