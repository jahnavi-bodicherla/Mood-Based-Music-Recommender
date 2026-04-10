"""
spotify.py — Spotify Web API Client (OAuth 2.0 Client Credentials)
──────────────────────────────────────────────────────────────────
Security:
  • Keys loaded from .env only (never hardcoded)
  • HTTPS-only requests
  • Token cached in memory with expiry tracking
  • Auto-retry with back-off on rate limits (429)
  • Input sanitisation before building query strings
  • Sensitive data masked in all log output
"""

import os
import re
import time
import logging
from typing import Optional

import requests
from dotenv import load_dotenv

from security import validate_api_key, mask_key, sanitize_text

load_dotenv()
logger = logging.getLogger(__name__)

# ── API endpoints (HTTPS only) ─────────────────────────────────────────────────
_TOKEN_URL  = "https://accounts.spotify.com/api/token"   # OAuth 2.0 token
_SEARCH_URL = "https://api.spotify.com/v1/search"        # track search
_RECS_URL   = "https://api.spotify.com/v1/recommendations"  # audio-feature recs

# Mood → Spotify genre seeds + target audio features
_MOOD_SEEDS = {
    "happy":      {"genres":["pop","dance","happy"],       "target_valence":0.9,  "target_energy":0.8},
    "sad":        {"genres":["sad","acoustic","piano"],    "target_valence":0.2,  "target_energy":0.2},
    "energetic":  {"genres":["work-out","metal","rock"],   "target_valence":0.7,  "target_energy":0.95},
    "relaxed":    {"genres":["chill","ambient","sleep"],   "target_valence":0.6,  "target_energy":0.3},
    "romantic":   {"genres":["romance","soul","r-n-b"],    "target_valence":0.75, "target_energy":0.5},
    "angry":      {"genres":["metal","hard-rock","punk"],  "target_valence":0.3,  "target_energy":0.9},
    "chill":      {"genres":["chill","indie","lo-fi"],     "target_valence":0.65, "target_energy":0.4},
    "focus":      {"genres":["classical","study","piano"], "target_valence":0.5,  "target_energy":0.3},
    "party":      {"genres":["party","dance","edm"],       "target_valence":0.9,  "target_energy":0.95},
    "devotional": {"genres":["gospel","new-age","folk"],   "target_valence":0.7,  "target_energy":0.2},
}

_LANG_MARKETS = {
    "English":"US", "Hindi":"IN", "Telugu":"IN", "Tamil":"IN", "Any":"US",
}
_LANG_QUERY_TAGS = {
    "Hindi":"bollywood hindi", "Telugu":"telugu", "Tamil":"tamil",
    "English":"", "Any":"",
}


# ── Token store ────────────────────────────────────────────────────────────────
class _TokenStore:
    token:      Optional[str] = None
    expires_at: float         = 0.0

    @classmethod
    def valid(cls) -> bool:
        return bool(cls.token) and time.time() < cls.expires_at - 30


def _get_credentials() -> tuple[str, str]:
    cid    = os.environ.get("SPOTIFY_CLIENT_ID",     "").strip()
    secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip()
    ok_c,  msg_c  = validate_api_key(cid,    "SPOTIFY_CLIENT_ID")
    ok_s,  msg_s  = validate_api_key(secret, "SPOTIFY_CLIENT_SECRET")
    if not ok_c: raise ValueError(msg_c)
    if not ok_s: raise ValueError(msg_s)
    return cid, secret


def _fetch_token() -> str:
    """Exchange client credentials for a Bearer token (OAuth 2.0)."""
    if _TokenStore.valid():
        return _TokenStore.token  # type: ignore

    cid, secret = _get_credentials()
    resp = requests.post(
        _TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(cid, secret),
        timeout=10,
    )
    if resp.status_code == 401:
        raise RuntimeError("Spotify: invalid credentials. Check SPOTIFY_CLIENT_ID / SECRET.")
    if resp.status_code == 429:
        raise RuntimeError("Spotify: rate-limited on token endpoint.")
    resp.raise_for_status()

    d = resp.json()
    _TokenStore.token      = d["access_token"]
    _TokenStore.expires_at = time.time() + d.get("expires_in", 3600)
    logger.info("Spotify token refreshed (expires in %ds)", d.get("expires_in"))
    return _TokenStore.token  # type: ignore


def _get(url: str, params: dict, retries: int = 3) -> Optional[dict]:
    """Authenticated HTTPS GET with automatic token refresh and rate-limit retry."""
    for attempt in range(retries):
        try:
            token = _fetch_token()
            resp  = requests.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                params=params,
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 401:
                _TokenStore.token = None        # force refresh
                continue
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 2 ** attempt))
                logger.warning("Spotify 429 — waiting %ds", min(wait, 10))
                time.sleep(min(wait, 10))
                continue
            logger.error("Spotify HTTP %d on %s", resp.status_code, url)
            return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.warning("Spotify connection error (attempt %d): %s", attempt+1, e)
            if attempt < retries - 1:
                time.sleep(1 + attempt)
        except Exception as e:
            logger.error("Spotify unexpected error: %s", e)
            return None
    return None


# ── Public API ─────────────────────────────────────────────────────────────────
def is_configured() -> bool:
    """True if both Spotify env vars are set and non-placeholder."""
    cid    = os.environ.get("SPOTIFY_CLIENT_ID",     "")
    secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
    return (bool(cid and secret)
            and not cid.strip().lower().startswith("your_")
            and not secret.strip().lower().startswith("your_"))


def check_credentials() -> tuple[bool, str]:
    """Validate credentials against the real token endpoint."""
    try:
        _fetch_token()
        return True, "Spotify ✅ connected"
    except ValueError  as e: return False, str(e)
    except RuntimeError as e: return False, str(e)
    except Exception as e:   return False, f"Unexpected: {e}"


def get_recommendations(mood: str, language: str = "Any",
                        limit: int = 5) -> list[dict]:
    """
    Fetch personalised tracks from /v1/recommendations using
    audio-feature targets derived from the mood profile.
    Falls back to search_songs() if the endpoint returns nothing.
    Returns [] on any error.
    """
    mood  = sanitize_text(mood.lower())
    limit = max(1, min(limit, 20))
    seeds = _MOOD_SEEDS.get(mood, {"genres":["pop"],"target_valence":0.5,"target_energy":0.5})

    data = _get(_RECS_URL, {
        "seed_genres":    ",".join(seeds["genres"][:3]),
        "limit":          limit,
        "market":         _LANG_MARKETS.get(language, "US"),
        "target_valence": seeds["target_valence"],
        "target_energy":  seeds["target_energy"],
    })
    if not data or not data.get("tracks"):
        logger.info("Recommendations empty — falling back to search")
        return search_songs(mood, language, limit)
    return _parse_tracks(data.get("tracks", []))


def search_songs(mood: str, language: str = "Any",
                 limit: int = 5) -> list[dict]:
    """
    Search Spotify via /v1/search using mood keywords + language tag.
    Returns [] on any error.
    """
    mood   = sanitize_text(mood.lower())
    lang_q = _LANG_QUERY_TAGS.get(language, "")
    query  = f"{mood} music {lang_q}".strip()
    limit  = max(1, min(limit, 20))

    data = _get(_SEARCH_URL, {
        "q":      query,
        "type":   "track",
        "limit":  limit,
        "market": _LANG_MARKETS.get(language, "US"),
    })
    if not data:
        return []
    return _parse_tracks(data.get("tracks", {}).get("items", []))


def _parse_tracks(items: list) -> list[dict]:
    """Convert raw Spotify track objects to our standard dict format."""
    results = []
    for t in items:
        try:
            imgs = t.get("album", {}).get("images", [])
            results.append({
                "song":        t["name"],
                "artist":      ", ".join(a["name"] for a in t["artists"]),
                "album":       t["album"]["name"],
                "album_image": imgs[0]["url"] if imgs else None,
                "preview_url": t.get("preview_url"),   # 30s MP3; can be None
                "spotify_url": t["external_urls"]["spotify"],
                "popularity":  t.get("popularity", 0),
                "source":      "spotify",
            })
        except (KeyError, IndexError):
            continue
    return results
