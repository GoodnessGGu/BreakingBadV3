# signal_parser.py
import re
import logging

logger = logging.getLogger(__name__)

# --- Core Signal Parsing Logic ---
def clean_signal_line(line: str) -> str:
    """
    Cleans up a raw signal line.
    Handles space-separated or semicolon-separated inputs.
    Preserves pair names like EURUSD-OTC.
    """
    line = line.strip()
    line = line.replace("i", ";")
    line = re.sub(r"[Oo](\d)", r"0\1", line)   # O1:05 -> 01:05
    line = re.sub(r"[Oo]\s*:", "0:", line)     # O :05 -> 0:05
    
    # If no semicolons, assume whitespace is the separator
    if ";" not in line:
        # replace one or more spaces with a semicolon
        line = re.sub(r"\s+", ";", line)
    
    # Now remove all spaces (since we have semicolons or it's already dense)
    line = line.replace(" ", "")

    # If it still looks malformed (e.g. mashed like 12:00EURUSDCALL5), try regex recovery
    # But only if we don't have enough parts
    if line.count(";") < 3:
         # Improved regex to capture OTC pairs and non-standard lengths
         # 1: Time, 2: Pair (letters, nums, -, :), 3: Direction, 4: Expiry, 5: Martingale (optional)
         parts = re.findall(r"(\d{1,2}:\d{2})|([A-Z0-9:/-]{3,15})|(CALL|PUT)|(\d{1,2})", line, re.IGNORECASE)
         
         # findall returns list of tuples [('12:00', '', '', ''), ('', 'EURUSD', '', '')...]
         # We need to flatten and filter
         flat_parts = []
         for p in parts:
             for item in p:
                 if item: flat_parts.append(item)
         
         if len(flat_parts) >= 4:
             line = ";".join(flat_parts)

    return line


def parse_signal(line: str):
    """
    Parses a single cleaned signal line into a dictionary.
    Expected format: HH:MM;PAIR;DIRECTION;TIMEFRAME
    Example: 03:40;EURAUD;CALL;5
    """
    try:
        cleaned = clean_signal_line(line)
        parts = cleaned.split(";")

        # Basic validation: ensure we have at least time, pair, dir, expiry
        if len(parts) < 4:
            logger.warning(f"Skipping invalid signal format: {line}")
            return None

        time_str = parts[0].strip()
        pair = parts[1].upper().replace("/", "")
        direction = parts[2].upper()
        
        # Extract expiry - handle cases like '5m' or 'M5'
        expiry_raw = parts[3]
        expiry = int(re.sub(r"\D", "", expiry_raw)) if re.search(r"\d", expiry_raw) else 5 # default to 5 if missing? No, failing better.
        if not re.search(r"\d", expiry_raw):
             logger.warning(f"Invalid expiry (no number found): {expiry_raw}")
             return None

        # Validate structure
        if not re.match(r"^\d{1,2}:\d{2}$", time_str):
            logger.warning(f"Invalid time format: {time_str}")
            return None

        if direction not in ["CALL", "PUT"]:
            logger.warning(f"Invalid direction: {direction}")
            return None

        return {
            "time": time_str,
            "pair": pair,
            "direction": direction,
            "expiry": expiry
        }
    except Exception as e:
        logger.error(f"❌ Failed to parse line '{line}': {e}")
        return None


# --- Parse Signals from Text ---
def parse_signals_from_text(text: str):
    """
    Parses multiple signals from a text string.
    Supports both compact format (01:00;EURUSD;CALL;5) and block format with emojis.
    """
    signals = []
    
    # 1. Try format "🔔 NEW SIGNAL!" (Block format)
    # This format usually comes as one message per signal, or multiple blocks.
    block_pattern = re.compile(
        r"Trade:\s*(?:.*\s+)?([A-Z]{3}/[A-Z]{3})\s*(?:.*?)?(\(OTC\))?.*?Timer:\s*(\d+)\s*minutes.*?Entry:\s*(\d{1,2}:\d{2})\s*(AM|PM)?.*?Direction:\s*(SELL|BUY)", 
        re.DOTALL | re.IGNORECASE
    )
    
    matches = block_pattern.findall(text)
    for m in matches:
        # m = ('EUR/GBP', '(OTC)', '5', '2:36', 'AM', 'SELL')
        try:
            pair = m[0].replace("/", "") 
            if m[1]: # OTC flag
                 if "OTC" not in pair: pair += "-OTC"
                 
            timer = int(m[2])
            
            # Time 12h -> 24h
            raw_time = m[3]
            meridiem = m[4].upper() if m[4] else ""
            
            if meridiem:
                # Convert 12h to 24h
                # Example: 2:36 AM -> 02:36, 2:36 PM -> 14:36
                import datetime
                dt = datetime.datetime.strptime(f"{raw_time} {meridiem}", "%I:%M %p")
                time_str = dt.strftime("%H:%M")
            else:
                time_str = raw_time # assume 24h if no AM/PM, or fix later
            
            direction = "PUT" if "SELL" in m[5].upper() else "CALL"
            
            sig = {
                "time": time_str,
                "pair": pair,
                "direction": direction,
                "expiry": timer
            }
            signals.append(sig)
        except Exception as e:
            logger.error(f"Error parsing block signal: {e}")

    # If we found block signals, return them
    if signals:
        logger.info(f"✅ Parsed {len(signals)} signals (Block Format).")
        return signals

    # 2. Fallback to Compact Format (legacy)
    # Split the text by what looks like a time pattern, but keep the delimiter
    parts = re.split(r'(\d{1,2}:\d{2})', text)
    
    # The first part is usually empty or garbage, so skip it.
    # Then, we have pairs of [time, rest_of_signal]
    for i in range(1, len(parts), 2):
        time_str = parts[i]
        signal_body = parts[i+1]
        
        # Check if this chunk is actually part of a block format we missed?
        # Assuming compact format doesn't have "Trade:" keywords etc.
        if "Trade:" in signal_body: 
             continue # skip, handled above? (Actually findall handles it)

        # Reconstruct the signal line
        line = time_str + signal_body
        
        # Now parse the single line
        sig = parse_signal(line)
        if sig:
            signals.append(sig)

    if signals:
        logger.info(f"✅ Parsed {len(signals)} signals (Compact Format).")
    
    return signals


# --- Parse Signals from File ---
def parse_signals_from_file(filepath: str):
    """
    Reads a signal file (usually .txt) and parses all valid signals.
    """
    signals = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            sig = parse_signal(line)
            if sig:
                signals.append(sig)
        logger.info(f"✅ Parsed {len(signals)} signals from {filepath}.")
    except Exception as e:
        logger.error(f"❌ Failed to parse signal file: {e}")
    return signals