"""
Chat History Exporter for BreakingBadV3 Project
Parses full conversation transcript logs and outputs both Markdown and HTML exports.
"""

import os
import json
import re
import html
import argparse
from datetime import datetime

DEFAULT_TRANSCRIPT = "/root/.gemini/antigravity-cli/brain/adf0bd6b-4a7c-49e6-aadd-c263becd35bb/.system_generated/logs/transcript_full.jsonl"
DEFAULT_OUT_MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CHAT_HISTORY_EXPORT_PART2.md")
DEFAULT_OUT_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CHAT_HISTORY_EXPORT_PART2.html")

ENV_SECRET_KEYS = (
    "TELEGRAM_TOKEN", "TELEGRAM_BOT_TOKEN", "TELEGRAM_STRING_SESSION",
    "TELEGRAM_API_ID", "TELEGRAM_API_HASH", "API_ID", "API_HASH",
    "GEMINI_API_KEY", "OPENAI_API_KEY", "GOOGLE_SHEET_ID",
    "GOOGLE_SERVICE_ACCOUNT_JSON", "IQ_AI_TOKEN", "IQ_PASSWORD",
    "RAILWAY_TOKEN", "STRING_SESSION", "PHONE_NUMBER",
)


def redact_secrets(text: str) -> str:
    """Scrub API keys, tokens, session strings, and .env secrets before repo export."""
    if not text:
        return text
    out = text
    # Redact KEY=VALUE lines for sensitive environment variables
    key_alt = "|".join(ENV_SECRET_KEYS)
    out = re.sub(
        rf"(?im)^(\s*(?:export\s+)?(?:{key_alt})\s*[:=]\s*)(.+)$",
        r"\1[REDACTED]",
        out,
    )
    # Redact inline bot tokens, Google keys, Bearer tokens, and long Telethon sessions
    out = re.sub(r"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b", "[REDACTED_BOT_TOKEN]", out)
    out = re.sub(r"\bAIza[A-Za-z0-9_-]{30,}\b", "[REDACTED_API_KEY]", out)
    out = re.sub(r"(Bearer\s+)[A-Za-z0-9_.-]{20,}", r"\1[REDACTED_TOKEN]", out)
    out = re.sub(r"\bghp_[A-Za-z0-9]{20,}\b", "[REDACTED_TOKEN]", out)
    out = re.sub(r"\bsk-[A-Za-z0-9_-]{20,}\b", "[REDACTED_TOKEN]", out)
    out = re.sub(r"\b1vYkKqMX[A-Za-z0-9_-]{20,}\b", "[REDACTED_SHEET_ID]", out)
    out = re.sub(r"\b3ULv3En8[A-Za-z0-9_-]{20,}\b", "[REDACTED_TOKEN]", out)
    out = re.sub(r"\b1BVts[A-Za-z0-9+/=_-]{60,}\b", "[REDACTED_SESSION]", out)
    out = re.sub(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----", "[REDACTED_PRIVATE_KEY]", out)
    return out


def parse_transcript(file_path):
    turns = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            t_type = obj.get("type")
            source = obj.get("source")
            created_at = obj.get("created_at", "")
            content = obj.get("content", "")

            if t_type == "USER_INPUT" and source == "USER_EXPLICIT":
                if "<USER_REQUEST>" in content:
                    m = re.search(r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>", content, re.DOTALL)
                    user_text = m.group(1).strip() if m else content.strip()
                else:
                    user_text = re.sub(r"<ADDITIONAL_METADATA>.*?</ADDITIONAL_METADATA>", "", content, flags=re.DOTALL).strip()

                user_text = redact_secrets(user_text)
                if not user_text:
                    continue

                turns.append({
                    "role": "user",
                    "time": created_at,
                    "text": user_text
                })

            elif t_type == "PLANNER_RESPONSE" and source == "MODEL":
                if content and content.strip():
                    cleaned = redact_secrets(content.strip())
                    if turns and turns[-1]["role"] == "assistant":
                        if cleaned != turns[-1]["text"]:
                            turns[-1]["text"] += "\n\n" + cleaned
                    else:
                        turns.append({
                            "role": "assistant",
                            "time": created_at,
                            "text": cleaned
                        })
    return turns

def generate_markdown(turns, out_path):
    lines = [
        "# 📜 BreakingBad V3 — Full Project Chat & Development History",
        "",
        f"> **Generated On**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> **Total Messages**: {len(turns)} ({len([t for t in turns if t['role'] == 'user'])} User Prompts, {len([t for t in turns if t['role'] == 'assistant'])} Assistant Responses)",
        f"> **Project**: BreakingBad V3 Algorithmic Trading & Telegram Ecosystem",
        "",
        "---",
        "",
        "## 📑 Table of Contents (Chronological Overview)",
        ""
    ]

    # Generate TOC
    user_idx = 0
    for i, t in enumerate(turns):
        if t["role"] == "user":
            user_idx += 1
            t_title = t["text"].split("\n")[0][:80].replace("[", "").replace("]", "").replace("`", "")
            t_time = t["time"][:16].replace("T", " ") if t["time"] else ""
            anchor = f"prompt-{user_idx}"
            lines.append(f"{user_idx}. [{t_time}] [{t_title}](#{anchor})")

    lines.append("")
    lines.append("---")
    lines.append("")

    user_idx = 0
    for t in turns:
        t_time = t["time"][:19].replace("T", " ") if t["time"] else "N/A"
        if t["role"] == "user":
            user_idx += 1
            lines.append(f'<a id="prompt-{user_idx}"></a>')
            lines.append(f"### 👤 User Prompt #{user_idx} — *{t_time}*")
            lines.append("")
            lines.append(t["text"])
            lines.append("")
            lines.append("---")
            lines.append("")
        else:
            lines.append(f"#### 🤖 Assistant (Antigravity) — *{t_time}*")
            lines.append("")
            lines.append(t["text"])
            lines.append("")
            lines.append("---")
            lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Markdown export saved to: {out_path}")

def generate_html(turns, out_path):
    user_turns_count = len([t for t in turns if t["role"] == "user"])
    asst_turns_count = len([t for t in turns if t["role"] == "assistant"])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BreakingBad V3 — Chat & Engineering History</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-user: #1f2937;
            --bg-assistant: #111827;
            --border-color: #30363d;
            --text-primary: #c9d1d9;
            --text-heading: #58a6ff;
            --accent-green: #238636;
            --accent-blue: #388bfd;
            --code-bg: #0b0e14;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        header {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        header h1 {{
            color: #ffffff;
            margin-bottom: 8px;
            font-size: 24px;
        }}
        header .meta {{
            color: #8b949e;
            font-size: 14px;
        }}
        .chat-container {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .message-card {{
            border-radius: 8px;
            border: 1px solid var(--border-color);
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .message-card.user {{
            background-color: var(--bg-user);
            border-left: 4px solid var(--accent-blue);
        }}
        .message-card.assistant {{
            background-color: var(--bg-assistant);
            border-left: 4px solid var(--accent-green);
        }}
        .message-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            padding-bottom: 8px;
        }}
        .sender-badge {{
            font-weight: 600;
            font-size: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .message-card.user .sender-badge {{ color: #79c0ff; }}
        .message-card.assistant .sender-badge {{ color: #7ee787; }}
        .timestamp {{
            font-size: 12px;
            color: #8b949e;
        }}
        .message-body {{
            font-size: 14.5px;
            overflow-x: auto;
        }}
        .message-body p {{
            margin-bottom: 12px;
        }}
        .message-body pre {{
            background-color: var(--code-bg);
            padding: 14px;
            border-radius: 6px;
            border: 1px solid #21262d;
            overflow-x: auto;
            margin: 14px 0;
        }}
        .message-body code {{
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
            font-size: 13px;
        }}
        .message-body table {{
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
        }}
        .message-body th, .message-body td {{
            border: 1px solid var(--border-color);
            padding: 8px 12px;
            text-align: left;
        }}
        .message-body th {{
            background-color: #1f242c;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📜 BreakingBad V3 — Project Chat & Engineering Transcript</h1>
            <div class="meta">
                <span><b>Export Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span> • 
                <span><b>User Prompts:</b> {user_turns_count}</span> • 
                <span><b>Assistant Turns:</b> {asst_turns_count}</span>
            </div>
        </header>

        <div class="chat-container" id="chat-stream"></div>
    </div>

    <script>
        const chatData = {json.dumps(turns, ensure_ascii=False).replace("</", "<\\/")};
        const container = document.getElementById('chat-stream');

        chatData.forEach((turn, idx) => {{
            const card = document.createElement('div');
            card.className = `message-card ${{turn.role}}`;

            const header = document.createElement('div');
            header.className = 'message-header';

            const sender = document.createElement('div');
            sender.className = 'sender-badge';
            sender.innerHTML = turn.role === 'user' ? `👤 User #${{idx + 1}}` : '🤖 Antigravity Assistant';

            const time = document.createElement('div');
            time.className = 'timestamp';
            time.textContent = turn.time ? turn.time.replace('T', ' ').substring(0, 19) : '';

            header.appendChild(sender);
            header.appendChild(time);

            const body = document.createElement('div');
            body.className = 'message-body';
            body.innerHTML = marked.parse(turn.text);

            card.appendChild(header);
            card.appendChild(body);
            container.appendChild(card);
        }});
    </script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[OK] HTML export saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export redacted Antigravity conversation history.")
    parser.add_argument("--transcript", default=DEFAULT_TRANSCRIPT)
    parser.add_argument("--out-md", default=DEFAULT_OUT_MD)
    parser.add_argument("--out-html", default=DEFAULT_OUT_HTML)
    args = parser.parse_args()

    turns = parse_transcript(args.transcript)
    generate_markdown(turns, args.out_md)
    generate_html(turns, args.out_html)
