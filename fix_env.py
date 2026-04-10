"""
fix_env.py — Fixes the .env encoding error and recreates the file cleanly.
Run:  python fix_env.py
"""
import os, secrets
from pathlib import Path

env_path = Path(".env")

# Try to read existing file in any encoding to salvage content
existing_keys = {}
for enc in ["utf-16", "utf-16-le", "utf-16-be", "latin-1", "cp1252", "utf-8-sig"]:
    try:
        content = env_path.read_text(encoding=enc)
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                existing_keys[key.strip()] = val.strip()
        print(f"Read existing .env using {enc} encoding")
        break
    except Exception:
        continue

# Use existing SECRET_KEY if it was valid, otherwise generate a new one
secret = existing_keys.get("SECRET_KEY", "")
if not secret or secret.startswith("your_"):
    secret = secrets.token_hex(32)
    print(f"Generated new SECRET_KEY")
else:
    print(f"Keeping existing SECRET_KEY: {secret[:12]}***")

sp_id     = existing_keys.get("SPOTIFY_CLIENT_ID",     "your_spotify_client_id_here")
sp_secret = existing_keys.get("SPOTIFY_CLIENT_SECRET", "your_spotify_client_secret_here")
yt_key    = existing_keys.get("YOUTUBE_API_KEY",        "your_youtube_api_key_here")

# Write back as clean UTF-8 (no BOM)
new_content = (
    "# MoodTunes API Keys\n"
    "# Encoding: UTF-8\n\n"
    f"SECRET_KEY={secret}\n"
    f"SPOTIFY_CLIENT_ID={sp_id}\n"
    f"SPOTIFY_CLIENT_SECRET={sp_secret}\n"
    f"YOUTUBE_API_KEY={yt_key}\n"
)

env_path.write_text(new_content, encoding="utf-8")
print(f"\n✅ .env rewritten as clean UTF-8 at: {env_path.absolute()}")
print("\nKeys set:")
print(f"  SECRET_KEY            = {secret[:12]}***")
print(f"  SPOTIFY_CLIENT_ID     = {sp_id[:20] if len(sp_id)>4 else sp_id}")
print(f"  SPOTIFY_CLIENT_SECRET = {'***set***' if not sp_secret.startswith('your_') else 'not set'}")
print(f"  YOUTUBE_API_KEY       = {'***set***' if not yt_key.startswith('your_') else 'not set'}")
print("\nNow run:  streamlit run app.py")
