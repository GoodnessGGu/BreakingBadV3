import sys, asyncio, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\GushEx\Documents\IQOPTIONS BOT\BreakingBadV3')
from dotenv import load_dotenv
load_dotenv()
from telethon import TelegramClient
from datetime import datetime, timezone
import importlib.util
spec = importlib.util.spec_from_file_location('czc', r'C:\Users\GushEx\Documents\IQOPTIONS BOT\BreakingBadV3\callisto_zone_copier.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
parser = mod.CallistoZoneParser
API_ID   = os.getenv('TELEGRAM_API_ID')
API_HASH = os.getenv('TELEGRAM_API_HASH')
CALLISTO = -1002848189989
START    = datetime(2026, 9, 7, tzinfo=timezone.utc)

async def main():
    client = TelegramClient('user_desktop_session', API_ID, API_HASH)
    await client.connect()
    entity = await client.get_entity(CALLISTO)
    dfmt = '%a %d %b %Y'
    print('Channel : ' + entity.title)
    print('Period  : ' + START.strftime(dfmt) + ' -> Now')
    print('=' * 80)
    messages  = await client.get_messages(entity, limit=500)
    zones_all = []
    for m in reversed(messages):
        text = m.message
        if not text or m.date < START:
            continue
        if not parser.is_zone_message(text):
            continue
        evts = parser.parse_all(text)
        if not evts:
            continue
        ufmt = '%Y-%m-%d %H:%M UTC'
        lfmt = '%Y-%m-%d %H:%M'
        dt_utc   = m.date.strftime(ufmt)
        dt_local = m.date.astimezone().strftime(lfmt) + ' WAT'
        for ev in evts:
            ev['dt_utc']   = dt_utc
            ev['dt_local'] = dt_local
            ev['msg_id']   = m.id
            zones_all.append(ev)
    zone_num = 0
    for ev in zones_all:
        side = ev['side']
        dtu  = ev['dt_utc']
        dtl  = ev['dt_local']
        if ev['type'] == 'INVALIDATE':
            print('   !! INVALIDATED ' + side + ' ZONE')
            print('      Posted : ' + dtu + '  /  ' + dtl)
            print('')
            continue
        zone_num += 1
        zlo = ev['zone_low']
        zhi = ev['zone_high']
        tgt = ev.get('target')
        ts  = str(round(tgt, 2)) if tgt else '???'
        lbl = 'BUY ' if side == 'BUY' else 'SELL'
        print('#' + str(zone_num) + ' [' + lbl + ']  Zone: ' + str(round(zlo,2)) + ' - ' + str(round(zhi,2)) + '  |  Target: ' + ts)
        print('         Posted : ' + dtu + '  /  ' + dtl)
        print('')
    print('Total zones: ' + str(zone_num))
    await client.disconnect()

asyncio.run(main())
