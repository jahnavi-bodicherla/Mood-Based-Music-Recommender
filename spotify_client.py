"""
spotify_client.py — Spotify OAuth 2.0 + API Client
────────────────────────────────────────────────────
Implements Authorization Code Flow for:
  - user-read-private
  - user-top-read
  - playlist-modify-public

All credentials loaded from environment / .env.
Token stored in session_state (no disk persistence needed).
"""

import os, time, base64, hashlib, secrets, logging
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Credentials (from .env only) ────────────────────────────────────────────
CLIENT_ID     = os.environ.get("SPOTIFY_CLIENT_ID",     "").strip()
CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip()
REDIRECT_URI  = os.environ.get("SPOTIFY_REDIRECT_URI",  "http://localhost:8501/callback").strip()

# Spotify endpoints
_AUTH_URL    = "https://accounts.spotify.com/authorize"
_TOKEN_URL   = "https://accounts.spotify.com/api/token"
_API_BASE    = "https://api.spotify.com/v1"
_SCOPES      = "user-read-private user-top-read playlist-modify-public"

# Mood → genre seeds + audio feature targets
from recommender import MOOD_PROFILES
_MOOD_SEEDS: dict = {}
try:
    import yaml
    from pathlib import Path
    with open(Path(__file__).parent / "config.yaml", encoding="utf-8") as f:
        _MOOD_SEEDS = yaml.safe_load(f).get("spotify_seeds", {})
except Exception:
    _MOOD_SEEDS = {
        "happy":["pop","dance","happy"], "sad":["sad","acoustic","piano"],
        "energetic":["work-out","metal","rock"], "relaxed":["chill","ambient","sleep"],
        "romantic":["romance","soul","r-n-b"], "angry":["metal","hard-rock","punk"],
        "chill":["chill","indie","lo-fi"], "focus":["classical","study","piano"],
        "party":["party","dance","edm"], "devotional":["gospel","new-age","folk"],
    }

# ── Helpers ──────────────────────────────────────────────────────────────────
def is_configured() -> bool:
    """True if both Spotify env vars are set and non-placeholder."""
    return (bool(CLIENT_ID and CLIENT_SECRET)
            and not CLIENT_ID.startswith("your_")
            and not CLIENT_SECRET.startswith("your_"))


def _mask(val: str, show: int = 8) -> str:
    return val[:show] + "***" if len(val) > show else "***"


# ── OAuth 2.0 Authorization Code Flow ────────────────────────────────────────
def get_auth_url(state: Optional[str] = None) -> tuple[str, str]:
    """
    Build the Spotify authorization URL.
    Returns (auth_url, state) — store state in session for CSRF protection.
    """
    if not is_configured():
        raise ValueError("Spotify credentials not configured. Add to .env")
    state = state or secrets.token_hex(16)
    params = {
        "response_type": "code",
        "client_id":     CLIENT_ID,
        "scope":         _SCOPES,
        "redirect_uri":  REDIRECT_URI,
        "state":         state,
    }
    req = requests.Request("GET", _AUTH_URL, params=params).prepare()
    return req.url, state  # type: ignore


def exchange_code(code: str) -> Optional[dict]:
    """
    Exchange the authorization code for access + refresh tokens.
    Returns token dict with: access_token, refresh_token, expires_in, expires_at
    """
    creds = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    try:
        resp = requests.post(
            _TOKEN_URL,
            data={
                "grant_type":   "authorization_code",
                "code":         code,
                "redirect_uri": REDIRECT_URI,
            },
            headers={"Authorization": f"Basic {creds}",
                     "Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        resp.raise_for_status()
        token = resp.json()
        token["expires_at"] = time.time() + token.get("expires_in", 3600) - 60
        logger.info("Spotify token exchanged for user session")
        return token
    except Exception as e:
        logger.error("Token exchange failed: %s", e)
        return None


def refresh_token(token: dict) -> Optional[dict]:
    """
    Use the refresh_token to get a new access_token.
    Returns updated token dict or None on failure.
    """
    rt = token.get("refresh_token","")
    if not rt:
        return None
    creds = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    try:
        resp = requests.post(
            _TOKEN_URL,
            data={"grant_type":"refresh_token","refresh_token":rt},
            headers={"Authorization": f"Basic {creds}",
                     "Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        resp.raise_for_status()
        new = resp.json()
        new["refresh_token"]  = rt  # reuse existing refresh token
        new["expires_at"]     = time.time() + new.get("expires_in", 3600) - 60
        logger.info("Spotify token refreshed")
        return new
    except Exception as e:
        logger.error("Token refresh failed: %s", e)
        return None


def is_token_valid(token: Optional[dict]) -> bool:
    if not token or not token.get("access_token"):
        return False
    return time.time() < token.get("expires_at", 0)


def ensure_valid_token(token: Optional[dict]) -> Optional[dict]:
    """Auto-refresh if expired. Returns valid token or None."""
    if is_token_valid(token):
        return token
    if token and token.get("refresh_token"):
        return refresh_token(token)
    return None


# ── Client-Credentials flow (no user needed — for basic recommendations) ────
_cc_token:      Optional[str]  = None
_cc_expires_at: float          = 0.0

def _get_cc_token() -> Optional[str]:
    """Client Credentials token for endpoints that don't need user login."""
    global _cc_token, _cc_expires_at
    if _cc_token and time.time() < _cc_expires_at - 30:
        return _cc_token
    if not is_configured():
        return None
    creds = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    try:
        resp = requests.post(
            _TOKEN_URL,
            data={"grant_type":"client_credentials"},
            headers={"Authorization": f"Basic {creds}"},
            timeout=10,
        )
        resp.raise_for_status()
        d = resp.json()
        _cc_token      = d["access_token"]
        _cc_expires_at = time.time() + d.get("expires_in", 3600) - 60
        return _cc_token
    except Exception as e:
        logger.error("CC token failed: %s", e)
        return None


# ── API request helper ───────────────────────────────────────────────────────
def _api(endpoint: str, params: dict = None,
         user_token: Optional[dict] = None,
         retries: int = 3) -> Optional[dict]:
    """
    Make an authenticated GET request to Spotify API.
    Prefers user token if provided (richer data), falls back to CC token.
    """
    for attempt in range(retries):
        # Choose token
        bearer = None
        if user_token:
            valid = ensure_valid_token(user_token)
            if valid:
                bearer = valid["access_token"]
        if not bearer:
            bearer = _get_cc_token()
        if not bearer:
            logger.warning("No valid Spotify token available")
            return None

        url = f"{_API_BASE}/{endpoint.lstrip('/')}"
        try:
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {bearer}"},
                params=params or {},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 401:
                # Force token refresh on next attempt
                if user_token:
                    user_token["expires_at"] = 0
                _cc_expires_at = 0
                continue
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 2 ** attempt))
                logger.warning("Spotify 429 — waiting %ds", wait)
                time.sleep(min(wait, 10)); continue
            logger.error("Spotify %d on %s", resp.status_code, endpoint)
            return None
        except requests.exceptions.Timeout:
            logger.warning("Timeout on attempt %d", attempt + 1)
            time.sleep(1)
        except Exception as e:
            logger.error("Spotify error: %s", e); return None
    return None


# ── Public API ───────────────────────────────────────────────────────────────
def get_recommendations(mood: str, language: str = "Any",
                         limit: int = 5,
                         user_token: Optional[dict] = None) -> list:
    """
    Fetch track recommendations using Spotify's /recommendations endpoint.
    Seeded with mood-specific genres + audio feature targets.
    Returns list of track dicts. Falls back to search on failure.
    """
    mood     = mood.lower().strip()
    limit    = max(1, min(limit, 20))
    seeds    = _MOOD_SEEDS.get(mood, ["pop"])
    profile  = MOOD_PROFILES.get(mood, {"energy":0.5,"valence":0.5,"tempo":100})
    market   = {"Hindi":"IN","Telugu":"IN","Tamil":"IN"}.get(language, "US")

    data = _api("recommendations", {
        "seed_genres":    ",".join(seeds[:3]),
        "limit":          limit,
        "market":         market,
        "target_energy":  profile["energy"],
        "target_valence": profile["valence"],
        "target_tempo":   profile["tempo"],
    }, user_token)

    if not data or not data.get("tracks"):
        logger.info("Recommendations empty, falling back to search")
        return search_tracks(mood, language, limit, user_token)

    return _parse_tracks(data.get("tracks", []))


def search_tracks(mood: str, language: str = "Any",
                   limit: int = 5,
                   user_token: Optional[dict] = None) -> list:
    """Search Spotify for tracks matching mood + language keywords."""
    lang_tag = {"Hindi":"bollywood hindi","Telugu":"telugu","Tamil":"tamil"}.get(language, "")
    query    = f"{mood} music {lang_tag}".strip()
    market   = {"Hindi":"IN","Telugu":"IN","Tamil":"IN"}.get(language, "US")
    data     = _api("search", {"q":query,"type":"track","limit":limit,"market":market},
                    user_token)
    if not data:
        return []
    return _parse_tracks(data.get("tracks",{}).get("items",[]))


def get_user_profile(user_token: dict) -> Optional[dict]:
    """Fetch the logged-in user's Spotify profile."""
    return _api("me", user_token=user_token)


def get_user_top_tracks(user_token: dict, limit: int = 10) -> list:
    """Fetch user's top tracks (requires user-top-read scope)."""
    data = _api("me/top/tracks", {"limit": limit, "time_range": "medium_term"},
                user_token)
    return _parse_tracks(data.get("items", [])) if data else []


def create_playlist(user_token: dict, user_id: str,
                     name: str, track_uris: list) -> Optional[str]:
    """Create a public playlist and add tracks. Returns playlist URL."""
    if not is_token_valid(user_token):
        return None
    bearer = user_token["access_token"]
    # Create
    try:
        r = requests.post(
            f"{_API_BASE}/users/{user_id}/playlists",
            json={"name": name, "public": True,
                  "description": "Created by MoodTunes"},
            headers={"Authorization": f"Bearer {bearer}",
                     "Content-Type": "application/json"},
            timeout=10)
        r.raise_for_status()
        pl_id  = r.json()["id"]
        pl_url = r.json()["external_urls"]["spotify"]
        # Add tracks
        requests.post(
            f"{_API_BASE}/playlists/{pl_id}/tracks",
            json={"uris": track_uris[:100]},
            headers={"Authorization": f"Bearer {bearer}",
                     "Content-Type": "application/json"},
            timeout=10)
        return pl_url
    except Exception as e:
        logger.error("Playlist creation failed: %s", e)
        return None


def get_audio_features(track_id: str,
                        user_token: Optional[dict] = None) -> Optional[dict]:
    """Fetch audio features for a track (energy, valence, tempo, etc.)."""
    data = _api(f"audio-features/{track_id}", user_token=user_token)
    if not data:
        return None
    return {k: data.get(k) for k in
            ["energy","valence","tempo","danceability","acousticness","speechiness"]}


def _parse_tracks(items: list) -> list:
    """Convert raw Spotify track objects to our standard format."""
    out = []
    for t in items:
        try:
            imgs = t.get("album",{}).get("images",[])
            out.append({
                "song":        t["name"],
                "artist":      ", ".join(a["name"] for a in t.get("artists",[])),
                "album":       t.get("album",{}).get("name",""),
                "album_image": imgs[0]["url"] if imgs else None,
                "preview_url": t.get("preview_url"),
                "spotify_url": t.get("external_urls",{}).get("spotify",""),
                "track_uri":   t.get("uri",""),
                "popularity":  t.get("popularity", 0),
                "source":      "spotify",
            })
        except (KeyError, TypeError):
            continue
    return out
