"""
security.py — Centralised Security Module for MoodTunes
─────────────────────────────────────────────────────────
Covers:
  • SHA-256 password hashing (with per-user salt)
  • HMAC-based session tokens (signed, time-limited)
  • Input sanitisation and validation
  • Brute-force protection (in-memory attempt counter)
  • Sensitive data masking for logs
  • Environment variable validation

All API keys must live in .env — never hardcoded here.
"""

import hashlib
import hmac
import os
import re
import time
import secrets
import base64
import logging
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Load security config ────────────────────────────────────────────────────
def _load_sec_cfg() -> dict:
    p = Path(__file__).parent / "config.yaml"
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("security", {})
        except Exception:
            pass
    return {}

_SEC = _load_sec_cfg()
PASSWORD_MIN_LEN   = _SEC.get("password_min_length", 6)
MAX_LOGIN_ATTEMPTS = _SEC.get("max_login_attempts", 5)
INPUT_MAX_LEN      = _SEC.get("input_max_length", 500)

# ── In-memory brute-force tracker {username: [timestamp, ...]} ─────────────
_attempt_log: dict[str, list[float]] = {}
_WINDOW_SECONDS = 300   # 5-minute window


# ════════════════════════════════════════════════════════════════
# PASSWORD HASHING  (SHA-256 + per-user random salt)
# ════════════════════════════════════════════════════════════════

def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """
    Hash a plain-text password with SHA-256 + a random salt.

    Returns
    -------
    (hashed_password, salt)  — both are hex strings; store both in the DB.
    """
    if salt is None:
        salt = secrets.token_hex(16)   # 128-bit random salt
    combined = f"{salt}{password}".encode("utf-8")
    hashed   = hashlib.sha256(combined).hexdigest()
    return hashed, salt


def verify_password(plain: str, stored_hash: str, salt: str) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    candidate, _ = hash_password(plain, salt)
    return hmac.compare_digest(candidate, stored_hash)


# ════════════════════════════════════════════════════════════════
# SESSION TOKENS  (HMAC-SHA256, time-limited)
# ════════════════════════════════════════════════════════════════

# App-level secret — generated once per process; for production,
# load from environment: SECRET_KEY=<random 32-byte hex>
_APP_SECRET = os.environ.get("SECRET_KEY") or secrets.token_hex(32)


def generate_token(username: str, expiry_hours: float = 24.0) -> str:
    """
    Create a signed, time-limited session token.
    Format: base64( username + "|" + expiry_unix + "|" + hmac_sig )
    """
    expiry    = int(time.time() + expiry_hours * 3600)
    payload   = f"{username}|{expiry}"
    sig       = hmac.new(
        _APP_SECRET.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    raw       = f"{payload}|{sig}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def verify_token(token: str) -> Optional[str]:
    """
    Validate a session token.
    Returns the username on success, None on failure / expiry.
    """
    try:
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        parts   = decoded.rsplit("|", 2)
        if len(parts) != 3:
            return None
        username, expiry_str, sig = parts
        expiry = int(expiry_str)
        if time.time() > expiry:
            logger.debug("Token expired for %s", mask(username))
            return None
        payload   = f"{username}|{expiry_str}"
        expected  = hmac.new(
            _APP_SECRET.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()
        if hmac.compare_digest(sig, expected):
            return username
    except Exception as e:
        logger.debug("Token verification error: %s", e)
    return None


# ════════════════════════════════════════════════════════════════
# BRUTE-FORCE PROTECTION
# ════════════════════════════════════════════════════════════════

def record_failed_attempt(username: str):
    """Record a failed login attempt for rate-limiting."""
    now   = time.time()
    _attempt_log.setdefault(username, [])
    _attempt_log[username].append(now)
    # Prune old attempts outside the window
    _attempt_log[username] = [
        t for t in _attempt_log[username] if now - t < _WINDOW_SECONDS
    ]


def is_locked_out(username: str) -> bool:
    """Return True if the account has too many recent failed attempts."""
    now = time.time()
    recent = [t for t in _attempt_log.get(username, [])
              if now - t < _WINDOW_SECONDS]
    return len(recent) >= MAX_LOGIN_ATTEMPTS


def clear_attempts(username: str):
    """Clear the attempt counter on successful login."""
    _attempt_log.pop(username, None)


def lockout_remaining(username: str) -> int:
    """Return seconds until the earliest attempt expires."""
    now    = time.time()
    recent = [t for t in _attempt_log.get(username, [])
              if now - t < _WINDOW_SECONDS]
    if not recent:
        return 0
    return max(0, int(_WINDOW_SECONDS - (now - min(recent))))


# ════════════════════════════════════════════════════════════════
# INPUT VALIDATION & SANITISATION
# ════════════════════════════════════════════════════════════════

# Characters that could cause XSS / injection in HTML output
_DANGEROUS = re.compile(r'[<>"\';|&\\{}]')
# Allowed username pattern
_USERNAME_RE = re.compile(r'^[a-zA-Z0-9_]{3,30}$')


def sanitize_text(text: str, max_len: int = INPUT_MAX_LEN) -> str:
    """
    Strip dangerous characters and truncate.
    Safe to use for free-text mood inputs, song names, etc.
    """
    if not isinstance(text, str):
        text = str(text)
    clean = _DANGEROUS.sub(" ", text.strip())
    return clean[:max_len]


def validate_username(username: str) -> tuple[bool, str]:
    """Validate username format. Returns (ok, error_message)."""
    username = username.strip()
    if not _USERNAME_RE.match(username):
        return (False,
                "Username must be 3–30 characters: letters, numbers, underscores only.")
    return True, ""


def validate_password(password: str) -> tuple[bool, str]:
    """Enforce minimum password requirements."""
    if len(password) < PASSWORD_MIN_LEN:
        return False, f"Password must be at least {PASSWORD_MIN_LEN} characters."
    return True, ""


def validate_api_key(key: str, name: str = "API key") -> tuple[bool, str]:
    """Check that an API key looks plausible (not a placeholder)."""
    if not key or key.strip().lower().startswith("your_"):
        return False, f"{name} is not configured. Add it to your .env file."
    if len(key.strip()) < 10:
        return False, f"{name} appears too short — check your .env file."
    return True, ""


# ════════════════════════════════════════════════════════════════
# SENSITIVE DATA MASKING  (for logs / UI display)
# ════════════════════════════════════════════════════════════════

def mask(value: str, show: int = 4) -> str:
    """
    Mask a sensitive string for safe logging.
    e.g. mask("abc123xyz") → "abc1***"
    """
    if not value:
        return "***"
    visible = min(show, len(value) // 2)
    return value[:visible] + "***"


def mask_key(key: str) -> str:
    """Show only first 8 chars of an API key."""
    return mask(key, show=8) if key else "NOT SET"


# ════════════════════════════════════════════════════════════════
# SHA-256 GENERIC HASHING  (for IDs, fingerprinting)
# ════════════════════════════════════════════════════════════════

def sha256_hex(data: str) -> str:
    """Return the SHA-256 hex digest of any string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ════════════════════════════════════════════════════════════════
# ENVIRONMENT VALIDATION  (startup check)
# ════════════════════════════════════════════════════════════════

def check_env() -> dict[str, bool]:
    """
    Return a dict of {env_var_name: is_set} for all expected keys.
    Used on the Home page to show API status.
    """
    keys = [
        "SPOTIFY_CLIENT_ID",
        "SPOTIFY_CLIENT_SECRET",
        "YOUTUBE_API_KEY",
        "SECRET_KEY",
    ]
    return {
        k: bool(os.environ.get(k, "").strip()
                and not os.environ.get(k, "").strip().lower().startswith("your_"))
        for k in keys
    }
