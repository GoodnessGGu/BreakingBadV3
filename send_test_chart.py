import os
import sys
import httpx
from dotenv import load_dotenv
import pandas as pd
from clients.forex_mcp_client import IQForexMCPClient
from strategies.ict_engine import ICTStrategyEngine
from utils.chart_generator import generate_ict_setup_chart

load_dotenv()
token = os.getenv("TELEGRAM_TOKEN")
admin_id = os.getenv("ADMIN_ID", "6420777416")

print("1. Initializing Forex MCP...")
fx = IQForexMCPClient()
fx.initialize()

print("2. Fetching candles...")
engine = ICTStrategyEngine(mcp_client=fx)
df = engine.fetch_recent_candles("XAUUSD", count=40)

if df is not None:
    print(f"3. Got {len(df)} candles. Generating chart...")
    last_p = float(df['Close'].iloc[-1])
    sweep_l = round(float(df['Low'].tail(15).min()), 2)
    cisd_l = round(float(df['Open'].iloc[-6]), 2)
    fvg_l = round(last_p - 4.5, 2)
    fvg_h = round(last_p - 1.0, 2)
    sl = round(sweep_l - 3.0, 2)
    tp = round(fvg_l + (abs(fvg_l - sl) * 2.2), 2)

    img_bytes = generate_ict_setup_chart(
        df=df,
        symbol="XAUUSD",
        side="BUY",
        sweep_level=sweep_l,
        cisd_level=cisd_l,
        fvg_low=fvg_l,
        fvg_high=fvg_h,
        sl=sl,
        tp=tp,
        timeframe="15M",
        num_candles=35
    )

    caption = (
        "🔥 [TEST ICT SETUP VISUALIZATION — XAUUSD BUY]\n"
        f"Sweep Trough: {sweep_l}\n"
        f"CISD Shift  : Broken above {cisd_l}\n"
        f"FVG Zone    : {fvg_l} – {fvg_h}\n"
        f"Stop Loss   : {sl}\n"
        f"Target TP   : {tp} (1:2.2 RR)\n"
        "⏳ Waiting for FVG retest..."
    )

    print("4. Sending photo to Telegram...")
    with httpx.Client(timeout=25.0) as client:
        files = {"photo": ("chart.png", img_bytes, "image/png")}
        data = {"chat_id": str(admin_id), "caption": caption}
        res = client.post(f"https://api.telegram.org/bot{token}/sendPhoto", data=data, files=files)
        print(f"5. Sent! Status: {res.status_code}, Response: {res.json().get('ok')}")
else:
    print("Failed to fetch candles from broker.")

os._exit(0)
