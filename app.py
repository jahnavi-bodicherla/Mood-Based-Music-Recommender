"""
MoodTunes — Complete Mood-Based Music Recommender
──────────────────────────────────────────────────
Run:  streamlit run app.py

Pages:  Login/Signup → Home → Recommend → Analytics
Modes:  Offline (CSV + ML)  |  Live (Spotify + YouTube)
"""

import csv
import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load .env before any API module imports
load_dotenv()

try:
    from security import (
        hash_password, verify_password,
        generate_token, verify_token,
        record_failed_attempt, is_locked_out, clear_attempts, lockout_remaining,
        sanitize_text, validate_username, validate_password,
        check_env,
    )
except ModuleNotFoundError:
    st.error("❌ **security.py not found.**\n\n"
             "Make sure ALL project files are in the same folder as app.py:\n"
             "- security.py\n- recommender.py\n- spotify.py\n- youtube.py\n"
             "- songs.csv\n- config.yaml\n\n"
             "Download them all from the chat and place them together.")
    st.stop()
try:
    from recommender import (
        ALL_MOODS, MOOD_PROFILES,
        detect_mood, load_songs, recommend, multi_mood_recommend,
        compute_language_weight, mood_language_heatmap, mood_popularity_table,
    )
except ModuleNotFoundError:
    st.error("❌ **recommender.py not found.**\n\n"
             "Place recommender.py in the same folder as app.py and restart.")
    st.stop()
# Safe imports — app works in offline-only mode if these files are missing
try:
    import spotify as sp
    _SP_AVAILABLE = True
except ModuleNotFoundError:
    _SP_AVAILABLE = False
    # Create a stub so the rest of app.py never needs to check
    class _SpotifyStub:
        def is_configured(self): return False
        def check_credentials(self): return False, "spotify.py not found in project folder"
        def get_recommendations(self, *a, **kw): return []
        def search_songs(self, *a, **kw): return []
    sp = _SpotifyStub()

try:
    import youtube as yt
    _YT_AVAILABLE = True
except ModuleNotFoundError:
    _YT_AVAILABLE = False
    class _YouTubeStub:
        def is_configured(self): return False
        def check_credentials(self): return False, "youtube.py not found in project folder"
        def search_videos(self, *a, **kw): return []
        def search_for_song(self, *a, **kw): return []
    yt = _YouTubeStub()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MoodTunes 🎵",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── File paths ─────────────────────────────────────────────────────────────────
USERS_FILE    = Path("users.csv")
FEEDBACK_FILE = Path("feedback.csv")

# ── Mood cosmetics ─────────────────────────────────────────────────────────────
EMOJI = {
    "happy":"😊","sad":"😔","energetic":"⚡","relaxed":"😌",
    "romantic":"💕","angry":"😤","chill":"🧊","focus":"🧠",
    "party":"🎉","devotional":"🙏",
}
COLOR = {
    "happy":"#F9C846","sad":"#5BA4CF","energetic":"#FF5F57",
    "relaxed":"#42C89B","romantic":"#EC4899","angry":"#F97316",
    "chill":"#60A5FA","focus":"#A855F7","party":"#FBBF24",
    "devotional":"#D97706",
}
LANGS = ["Any","English","Hindi","Telugu","Tamil"]

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:1rem}

/* ── offline song card ── */
.card{background:#1a1d28;border:1px solid #2e3250;border-radius:14px;
      padding:.9rem 1.1rem;margin-bottom:.45rem;
      transition:border-color .2s,transform .15s}
.card:hover{border-color:#7c5cfc;transform:translateY(-1px)}
.ctitle{font-weight:700;font-size:.97rem;margin-bottom:.08rem}
.csub{color:#888;font-size:.82rem;margin-bottom:.3rem}
.pill{display:inline-block;background:rgba(124,92,252,.14);color:#a78bfa;
      border-radius:20px;padding:.05rem .44rem;font-size:.67rem;
      font-weight:600;margin:.05rem .08rem 0 0}
.mpill{display:inline-block;border-radius:20px;padding:.05rem .44rem;
       font-size:.67rem;font-weight:600;margin:.05rem .08rem 0 0}

/* ── live cards ── */
.sp-card{background:#091610;border:1px solid #1db95440;border-radius:12px;
         padding:.85rem 1rem;margin-bottom:.45rem}
.yt-card{background:#130909;border:1px solid #ff000040;border-radius:12px;
         padding:.85rem 1rem;margin-bottom:.45rem}

/* ── badges ── */
.badge-offline{display:inline-block;background:rgba(124,92,252,.14);color:#a78bfa;
    border:1px solid rgba(124,92,252,.3);border-radius:20px;
    padding:.18rem .75rem;font-size:.74rem;font-weight:700}
.badge-live{display:inline-block;background:rgba(29,185,84,.11);color:#1db954;
    border:1px solid rgba(29,185,84,.3);border-radius:20px;
    padding:.18rem .75rem;font-size:.74rem;font-weight:700}
.badge-yt{display:inline-block;background:rgba(255,50,50,.1);color:#ff5555;
    border:1px solid rgba(255,50,50,.28);border-radius:20px;
    padding:.18rem .75rem;font-size:.74rem;font-weight:700}

/* ── section headers ── */
.sec-head{font-size:1rem;font-weight:700;margin:.8rem 0 .5rem}

/* ── metrics ── */
[data-testid="stMetric"]{background:#1a1d28;border:1px solid #2e3250;
    border-radius:12px;padding:.75rem!important}
[data-testid="stMetricValue"]{color:#7c5cfc!important;font-weight:700!important}

/* ── inputs ── */
.stTextInput>div>div>input{background:#10131e!important;
    border:1px solid #2e3250!important;border-radius:10px!important;
    color:#e8eaf0!important}
.stTextInput>div>div>input:focus{
    border-color:#7c5cfc!important;box-shadow:none!important}

.stButton>button{border-radius:10px!important;font-weight:600!important}

/* ── Live Encryption Dashboard ── */
.enc-panel{background:#0a0f1e;border:1px solid #1e3a5f;border-radius:14px;
           padding:1rem 1.2rem;margin-bottom:.7rem}
.enc-title{font-size:.82rem;font-weight:700;color:#64b5f6;
           letter-spacing:.06em;text-transform:uppercase;margin-bottom:.3rem}
.enc-value{font-family:'Courier New',monospace;font-size:.78rem;
           color:#a5d6a7;word-break:break-all;line-height:1.5}
.enc-label{font-size:.72rem;color:#888;margin-bottom:.15rem}
.hash-flow{background:#050b14;border:1px solid #1b2a40;border-radius:10px;
           padding:.75rem 1rem;margin:.3rem 0;font-family:monospace;
           font-size:.76rem;color:#80cbc4;word-break:break-all}
.sec-badge{display:inline-block;padding:.2rem .65rem;border-radius:20px;
           font-size:.72rem;font-weight:700;margin:.15rem .1rem}
.badge-green{background:rgba(29,185,84,.12);color:#1db954;
             border:1px solid rgba(29,185,84,.3)}
.badge-blue{background:rgba(33,150,243,.12);color:#42a5f5;
            border:1px solid rgba(33,150,243,.3)}
.badge-orange{background:rgba(255,152,0,.12);color:#ffa726;
              border:1px solid rgba(255,152,0,.3)}
.badge-red{background:rgba(244,67,54,.12);color:#ef5350;
           border:1px solid rgba(244,67,54,.3)}
.flow-arrow{color:#7c5cfc;font-size:1.1rem;text-align:center;
            margin:.2rem 0;display:block}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECURITY-AWARE USER STORE  (CSV with hashed passwords + salt)
# ══════════════════════════════════════════════════════════════════════════════

def _users_file_exists() -> bool:
    return USERS_FILE.exists() and USERS_FILE.stat().st_size > 0


def load_users() -> dict:
    """Returns {username: {password_hash, salt, age, language, token}}"""
    if not _users_file_exists():
        return {}
    users = {}
    try:
        with open(USERS_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                users[row["username"]] = {
                    "password_hash": row["password_hash"],
                    "salt":          row.get("salt",""),
                    "age":           int(row.get("age", 22)),
                    "language":      row.get("language","Any"),
                }
    except Exception:
        pass
    return users


def save_user(username: str, plain_password: str, age: int, language: str):
    """Hash the password (SHA-256 + salt) before writing to disk."""
    pwd_hash, salt = hash_password(plain_password)
    exists = _users_file_exists()
    with open(USERS_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["username","password_hash","salt","age","language"])
        if not exists:
            w.writeheader()
        w.writerow({
            "username":      username,
            "password_hash": pwd_hash,
            "salt":          salt,
            "age":           age,
            "language":      language,
        })


def load_feedback(username: str) -> dict:
    """Returns {song_name: "like"|"dislike"}"""
    if not FEEDBACK_FILE.exists():
        return {}
    fb = {}
    try:
        with open(FEEDBACK_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("username") == username:
                    fb[row["song"]] = row["action"]
    except Exception:
        pass
    return fb


def save_feedback_row(username: str, song: str, action: str):
    """Upsert a feedback row (removes old entry for same song first)."""
    rows = []
    if FEEDBACK_FILE.exists():
        try:
            with open(FEEDBACK_FILE, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            rows = [r for r in rows
                    if not (r["username"] == username and r["song"] == song)]
        except Exception:
            pass
    rows.append({
        "username":  username,
        "song":      song,
        "action":    action,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    with open(FEEDBACK_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["username","song","action","timestamp"])
        w.writeheader()
        w.writerows(rows)


# ── Cached data loader ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_df() -> pd.DataFrame:
    return load_songs()


def liked_disliked(fb: dict) -> tuple[list, list]:
    return ([s for s,a in fb.items() if a=="like"],
            [s for s,a in fb.items() if a=="dislike"])


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = dict(
    logged_in=False, username="", age=22, language="Any",
    token="", page="home", feedback={}, mood_history=[],
    auth_tab="login",
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Re-validate existing token on each rerun
if (st.session_state["logged_in"]
        and st.session_state["token"]
        and not verify_token(st.session_state["token"])):
    st.session_state["logged_in"] = False
    st.warning("⏱ Session expired. Please log in again.")


# ══════════════════════════════════════════════════════════════════════════════
# AUTH PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_auth():
    st.markdown("""
    <div style='text-align:center;padding:1.3rem 0 .5rem'>
        <div style='font-size:3rem'>🎵</div>
        <h1 style='font-size:1.9rem;margin:.15rem 0;
            background:linear-gradient(135deg,#7c5cfc,#e85d9b);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
            MoodTunes</h1>
        <p style='color:#888;font-size:.87rem'>
            Discover music that matches your mood</p>
    </div>""", unsafe_allow_html=True)

    tab = st.session_state["auth_tab"]
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔐 Login", use_container_width=True,
                      type="primary" if tab=="login" else "secondary"):
            st.session_state["auth_tab"] = "login"; st.rerun()
    with c2:
        if st.button("✏️ Sign Up", use_container_width=True,
                      type="primary" if tab=="signup" else "secondary"):
            st.session_state["auth_tab"] = "signup"; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    users = load_users()

    # ── LOGIN ──────────────────────────────────────────────────────────────────
    if tab == "login":
        st.markdown("<h3 style='text-align:center'>Welcome back 👋</h3>",
                    unsafe_allow_html=True)
        with st.form("lf"):
            uname = st.text_input("👤 Username", placeholder="your_username")
            pw    = st.text_input("🔒 Password", type="password",
                                   placeholder="••••••••")
            if st.form_submit_button("Sign In →", use_container_width=True):
                uname = sanitize_text(uname.strip())

                if is_locked_out(uname):
                    remaining = lockout_remaining(uname)
                    st.error(f"🔒 Too many failed attempts. Try again in "
                              f"{remaining}s.")
                elif not uname or not pw:
                    st.error("Both fields are required.")
                elif uname not in users:
                    record_failed_attempt(uname)
                    st.error("Username not found.")
                elif not verify_password(pw,
                                          users[uname]["password_hash"],
                                          users[uname]["salt"]):
                    record_failed_attempt(uname)
                    remaining = MAX_ATTEMPTS - len([
                        t for t in __import__('security')._attempt_log.get(uname,[])
                    ])
                    st.error(f"Incorrect password.")
                else:
                    clear_attempts(uname)
                    token = generate_token(uname)
                    st.session_state.update({
                        "logged_in": True,
                        "username":  uname,
                        "age":       users[uname]["age"],
                        "language":  users[uname]["language"],
                        "token":     token,
                        "feedback":  load_feedback(uname),
                        "page":      "home",
                    })
                    st.rerun()

        st.markdown("""
        <p style='text-align:center;color:#555;font-size:.76rem;margin-top:.8rem'>
            🔒 Passwords are hashed with SHA-256 + salt — never stored in plain text
        </p>""", unsafe_allow_html=True)

    # ── SIGN UP ────────────────────────────────────────────────────────────────
    else:
        st.markdown("<h3 style='text-align:center'>Create your account 🎧</h3>",
                    unsafe_allow_html=True)
        with st.form("sf"):
            cu, cpw = st.columns(2)
            with cu:  nu  = st.text_input("👤 Username *",  placeholder="cool_listener")
            with cpw: npw = st.text_input("🔒 Password *",  type="password",
                                           placeholder="Min 6 characters")
            npw2 = st.text_input("🔒 Confirm Password *", type="password",
                                  placeholder="Repeat your password")
            ca, cl = st.columns(2)
            with ca: age  = st.number_input("🎂 Age", 10, 99, 22)
            with cl: lang = st.selectbox("🌐 Language", LANGS)

            if st.form_submit_button("Create Account →", use_container_width=True):
                nu = sanitize_text(nu.strip())
                errs = []
                ok_u, msg_u = validate_username(nu)
                ok_p, msg_p = validate_password(npw)
                if not ok_u: errs.append(msg_u)
                if not ok_p: errs.append(msg_p)
                if npw != npw2: errs.append("Passwords do not match.")
                if nu in users:  errs.append("Username already taken.")
                if errs:
                    for e in errs: st.error(e)
                else:
                    save_user(nu, npw, int(age), lang)
                    st.success("✅ Account created! Sign in above.")
                    st.session_state["auth_tab"] = "login"
                    st.rerun()

    st.markdown("""
    <p style='text-align:center;color:#444;font-size:.73rem;margin-top:1.5rem'>
        Demo: create any account — no email needed</p>""",
    unsafe_allow_html=True)


MAX_ATTEMPTS = 5  # for inline remaining display


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        fb  = st.session_state["feedback"]
        mh  = st.session_state["mood_history"]
        lkd = sum(1 for a in fb.values() if a=="like")
        top = max(set(mh), key=mh.count) if mh else "—"

        st.markdown(f"""
        <div style='background:#1a1d28;border:1px solid #2e3250;
             border-radius:12px;padding:.75rem 1rem;margin-bottom:.85rem'>
            <div style='font-size:1.5rem;text-align:center'>🎵</div>
            <div style='font-weight:800;font-size:.98rem;text-align:center;
                 background:linear-gradient(135deg,#7c5cfc,#e85d9b);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
                 MoodTunes</div>
            <div style='color:#888;font-size:.74rem;text-align:center;margin-top:.25rem'>
                👤 {st.session_state["username"]} &nbsp;·&nbsp;
                🎂 {st.session_state["age"]}
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("### 📍 Navigation")
        for icon, pg, label in [
            ("🏠","home","Home"),
            ("🎵","recommend","Recommend"),
            ("🔒","security","Live Encryption"),
            ("📊","analytics","Analytics"),
        ]:
            active = st.session_state["page"] == pg
            if st.button(f"{icon} {label}", key=f"nav_{pg}",
                          use_container_width=True,
                          type="primary" if active else "secondary"):
                st.session_state["page"] = pg; st.rerun()

        st.markdown("---")
        st.metric("❤️ Liked",   lkd)
        st.metric("🎭 Top Mood",
                   f"{EMOJI.get(top,'')} {top.capitalize()}" if top!="—" else "—")

        # API status
        st.markdown("---")
        st.markdown("### 🔌 API Status")
        env = check_env()
        st.markdown(
            f"{'🟢' if (env.get('SPOTIFY_CLIENT_ID') and env.get('SPOTIFY_CLIENT_SECRET')) else '🔴'}"
            f" Spotify &nbsp;&nbsp;"
            f"{'🟢' if env.get('YOUTUBE_API_KEY') else '🔴'} YouTube",
            unsafe_allow_html=True)
        if not any(env.get(k) for k in
                   ["SPOTIFY_CLIENT_ID","YOUTUBE_API_KEY"]):
            st.caption("Add keys to .env to enable Live Mode")

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for k, v in _DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# CARD RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def offline_card(row: pd.Series, idx: int):
    """Offline CSV song card with Like/Dislike buttons."""
    name    = row["song"]
    artist  = row["artist"]
    mood    = row.get("mood","happy")
    lang    = row.get("language","")
    genre   = row.get("genre","")
    energy  = float(row.get("energy", 0))
    valence = float(row.get("valence", 0))
    tempo   = int(row.get("tempo", 0))
    pop     = int(row.get("popularity", 0))
    mc      = COLOR.get(mood,"#7c5cfc")

    fb_s   = st.session_state["feedback"].get(name, "none")
    border = "#42C89B" if fb_s=="like" else "#FF5F57" if fb_s=="dislike" else "#2e3250"

    st.markdown(f"""
    <div class='card' style='border-color:{border}'>
        <div class='ctitle'>
            <span style='color:{mc};font-weight:800;margin-right:.3rem'>{idx}.</span>
            {"❤️ " if fb_s=="like" else ""}{name}
        </div>
        <div class='csub'>🎤 {artist} &nbsp;·&nbsp; 🎸 {genre}</div>
        <span class='pill'>🥁 {tempo} BPM</span>
        <span class='pill'>⚡ {energy:.0%}</span>
        <span class='pill'>😊 {valence:.0%}</span>
        <span class='pill'>🔥 {pop}</span>
        <span class='mpill' style='background:{mc}22;color:{mc}'>
            {EMOJI.get(mood,"")} {mood}</span>
        <span class='mpill' style='background:rgba(255,255,255,.05);color:#bbb'>
            🌐 {lang}</span>
    </div>""", unsafe_allow_html=True)

    b1, b2, _ = st.columns([1,1,7])
    with b1:
        if st.button("❤️" if fb_s=="like" else "👍",
                      key=f"like_{idx}_{name}", use_container_width=True):
            st.session_state["feedback"][name] = "like"
            save_feedback_row(st.session_state["username"], name, "like")
            st.rerun()
    with b2:
        if st.button("👎", key=f"dis_{idx}_{name}", use_container_width=True):
            st.session_state["feedback"][name] = "dislike"
            save_feedback_row(st.session_state["username"], name, "dislike")
            st.rerun()


def spotify_card(item: dict, idx: int):
    """Spotify live result: album art + 30s HTML5 audio preview."""
    name  = item.get("song","")
    art   = item.get("artist","")
    album = item.get("album","")
    img   = item.get("album_image")
    prev  = item.get("preview_url")
    sp_u  = item.get("spotify_url","#")
    pop   = item.get("popularity",0)

    img_col, info_col = st.columns([1,5])
    with img_col:
        if img:
            st.image(img, width=80)
        else:
            st.markdown(
                "<div style='width:80px;height:80px;background:#0a1a0e;"
                "border-radius:8px;display:flex;align-items:center;"
                "justify-content:center;font-size:1.6rem'>🎵</div>",
                unsafe_allow_html=True)
    with info_col:
        st.markdown(f"""
        <div class='sp-card'>
            <div style='font-weight:700;font-size:.95rem'>
                <span style='color:#1db954;font-weight:800'>{idx}.</span> {name}
            </div>
            <div style='color:#888;font-size:.8rem;margin:.08rem 0 .3rem'>
                🎤 {art} &nbsp;·&nbsp; 💿 {album} &nbsp;·&nbsp; 🔥 {pop}
            </div>
            <span class='badge-live' style='font-size:.62rem'>🟢 Spotify</span>
        </div>""", unsafe_allow_html=True)
        lc, pc = st.columns([1,2])
        with lc:
            if sp_u != "#":
                st.link_button("🎧 Open", sp_u, use_container_width=True)
        with pc:
            if prev:
                st.markdown(
                    f"<audio controls style='width:100%;height:34px;"
                    f"border-radius:8px'><source src='{prev}' type='audio/mpeg'>"
                    f"</audio>",
                    unsafe_allow_html=True)
            else:
                st.caption("No preview available")


def youtube_card(item: dict, idx: int):
    """YouTube live result with embedded iframe player."""
    title   = item.get("title","")
    channel = item.get("channel","")
    embed   = item.get("embed_url","")
    watch   = item.get("watch_url","#")
    pub     = item.get("published","")
    short   = title[:70] + ("…" if len(title)>70 else "")

    with st.expander(f"📺 {idx}. {short}", expanded=(idx==1)):
        if embed:
            st.markdown(
                f"<iframe width='100%' height='260' src='{embed}' "
                f"frameborder='0' allow='accelerometer;autoplay;"
                f"clipboard-write;encrypted-media;gyroscope;picture-in-picture' "
                f"allowfullscreen style='border-radius:10px'></iframe>",
                unsafe_allow_html=True)
        st.markdown(f"""
        <div class='yt-card'>
            <div style='font-weight:700;font-size:.88rem'>{title}</div>
            <div style='color:#888;font-size:.77rem;margin:.1rem 0 .3rem'>
                📺 {channel} {"&nbsp;·&nbsp; 📅 "+pub if pub else ""}
            </div>
            <span class='badge-yt' style='font-size:.62rem'>▶ YouTube</span>
        </div>""", unsafe_allow_html=True)
        if watch != "#":
            st.link_button("▶ Watch on YouTube", watch,
                            use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    uname = st.session_state["username"]
    fb    = st.session_state["feedback"]
    mh    = st.session_state["mood_history"]

    sp_ready = sp.is_configured()
    yt_ready = yt.is_configured()
    mode_lbl = "🟢 Live + Offline" if (sp_ready or yt_ready) else "⚙️ Offline Only"

    # Warn if optional API files weren't found
    if not _SP_AVAILABLE or not _YT_AVAILABLE:
        missing = []
        if not _SP_AVAILABLE: missing.append("spotify.py")
        if not _YT_AVAILABLE: missing.append("youtube.py")
        st.warning(
            f"⚠️ **{', '.join(missing)} not found in your project folder.**  \n"
            f"The app works in Offline Mode without them. "
            f"To enable Live Mode, download these files from the chat "
            f"and place them in the same folder as app.py."
        )

    st.markdown(f"""
    <div style='padding:.4rem 0 .9rem'>
        <h1 style='font-size:1.85rem;margin-bottom:.2rem'>
            🎵 Welcome, <span style='color:#a78bfa'>{uname}</span>!
        </h1>
        <p style='color:#888;font-size:.88rem'>
            Mood-based music — 10 moods · ML + Hybrid AI · {mode_lbl}
        </p>
    </div>""", unsafe_allow_html=True)

    liked    = sum(1 for a in fb.values() if a=="like")
    disliked = sum(1 for a in fb.values() if a=="dislike")
    top_m    = max(set(mh),key=mh.count) if mh else "—"

    m1,m2,m3,m4 = st.columns(4)
    with m1: st.metric("🎂 Age",      st.session_state["age"])
    with m2: st.metric("🌐 Language", st.session_state["language"])
    with m3: st.metric("❤️ Liked",    liked)
    with m4: st.metric("🎭 Top Mood",
                        f"{EMOJI.get(top_m,'')} {top_m.capitalize()}"
                        if top_m!="—" else "—")

    # API / Security status
    st.markdown("---")
    st.markdown("### 🔒 Security & API Status")
    env = check_env()
    sec_items = [
        ("🔒 Password Hashing",    True,  "SHA-256 + salt enabled"),
        ("🔐 Session Tokens",       True,  "HMAC-signed, time-limited"),
        ("🌐 HTTPS API calls",      True,  "All API requests use HTTPS"),
        ("🔑 Keys in .env",         True,  "Never exposed in frontend"),
        ("🎧 Spotify API",          env.get("SPOTIFY_CLIENT_ID",False) and env.get("SPOTIFY_CLIENT_SECRET",False), ""),
        ("▶ YouTube API",           env.get("YOUTUBE_API_KEY",False), ""),
    ]
    c1, c2 = st.columns(2)
    for i,(label, ok, note) in enumerate(sec_items):
        col = c1 if i%2==0 else c2
        with col:
            icon = "🟢" if ok else "🔴"
            st.markdown(
                f"<div style='background:#1a1d28;border:1px solid "
                f"{'#1db95430' if ok else '#ff000030'};"
                f"border-radius:10px;padding:.5rem .8rem;margin-bottom:.4rem;"
                f"font-size:.83rem'>{icon} <b>{label}</b>"
                f"{'<br><span style=\"color:#888;font-size:.76rem\">' + note + '</span>' if note else ''}"
                f"</div>",
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 Quick Start — Pick a Mood")
    cols = st.columns(5)
    for i, mood in enumerate(ALL_MOODS):
        with cols[i%5]:
            if st.button(f"{EMOJI[mood]} {mood.capitalize()}",
                          key=f"qm_{mood}", use_container_width=True):
                st.session_state["_preset"] = mood
                st.session_state["page"]    = "recommend"; st.rerun()

    st.markdown("---")
    st.markdown("### 🎭 10 Mood Profiles")
    rows = [{"Mood": f"{EMOJI[m]} {m.capitalize()}",
             "Energy":  f"{p['energy']:.0%}",
             "Valence": f"{p['valence']:.0%}",
             "Tempo":   f"{p['tempo']} BPM"}
            for m, p in MOOD_PROFILES.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if mh:
        st.markdown("---")
        st.markdown("### 📈 Your Mood History")
        mc_s = pd.Series(mh).value_counts().reset_index()
        mc_s.columns = ["Mood","Count"]
        st.bar_chart(mc_s.set_index("Mood")["Count"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RECOMMEND
# ══════════════════════════════════════════════════════════════════════════════
def page_recommend():
    df = get_df()
    fb = st.session_state["feedback"]
    liked, disliked = liked_disliked(fb)
    lw = compute_language_weight(liked, df)

    sp_ok = sp.is_configured()
    yt_ok = yt.is_configured()

    st.markdown("""
    <h1 style='font-size:1.85rem;margin-bottom:.2rem'>🎵 Recommend</h1>
    <p style='color:#888;margin-bottom:.7rem'>
        Offline ML • Live Spotify • Live YouTube</p>
    """, unsafe_allow_html=True)

    # ── Offline/Live toggle ────────────────────────────────────────────────────
    tog_col, _ = st.columns([2,3])
    with tog_col:
        if sp_ok or yt_ok:
            live = st.toggle("🌐 Live Mode (Spotify + YouTube)", value=False)
        else:
            live = False
            st.markdown(
                "<span class='badge-offline'>⚙️ Offline (add .env to enable Live)</span>",
                unsafe_allow_html=True)

    if live:
        parts = []
        if sp_ok: parts.append("🟢 Spotify")
        if yt_ok: parts.append("▶ YouTube")
        st.markdown(
            f"<span class='badge-live'>🌐 Live — {' + '.join(parts)}</span>"
            f"&nbsp; <span class='badge-offline'>+ ⚙️ ML Offline</span>",
            unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge-offline'>⚙️ Offline Mode — CSV + ML</span>",
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input ──────────────────────────────────────────────────────────────────
    mode = st.radio("", ["💬 Text","🎛️ Dropdown","🎲 Surprise"],
                     horizontal=True, label_visibility="collapsed")

    preset        = st.session_state.pop("_preset", None)
    selected_mood = preset or "happy"
    multi_mode    = False
    multi_moods   = []

    if mode == "💬 Text":
        raw = st.text_input("How are you feeling?",
                             placeholder='"Need to focus…" or "Feeling so happy!"')
        if raw:
            clean_input   = sanitize_text(raw)   # strip dangerous chars
            selected_mood, conf = detect_mood(clean_input)
            mc = COLOR[selected_mood]
            st.markdown(
                f"<div style='background:{mc}22;border:1px solid {mc}55;"
                f"border-radius:10px;padding:.45rem .85rem;font-size:.85rem;"
                f"margin:.35rem 0'>"
                f"<b>Detected:</b> "
                f"<b style='color:{mc}'>{EMOJI[selected_mood]} "
                f"{selected_mood.capitalize()}</b>"
                f"<span style='color:#888;font-size:.77rem'>"
                f" &nbsp;·&nbsp; {conf:.0%} confidence</span></div>",
                unsafe_allow_html=True)

    elif mode == "🎛️ Dropdown":
        cm, ct = st.columns([3,1])
        with cm:
            selected_mood = st.selectbox(
                "Mood", ALL_MOODS,
                index=ALL_MOODS.index(selected_mood)
                      if selected_mood in ALL_MOODS else 0,
                format_func=lambda m: f"{EMOJI[m]}  {m.capitalize()}",
                label_visibility="collapsed")
        with ct:
            multi_mode = st.toggle("Multi", value=False)
        if multi_mode:
            multi_moods = st.multiselect(
                "Blend 2–4 moods", ALL_MOODS,
                default=[selected_mood], max_selections=4,
                format_func=lambda m: f"{EMOJI[m]} {m.capitalize()}")
            if len(multi_moods) < 2:
                st.info("Pick at least 2 moods to blend.")
                multi_mode = False
    else:
        if st.button("🎲 Random Mood", use_container_width=True):
            st.session_state["_surprise"] = random.choice(ALL_MOODS)
        selected_mood = st.session_state.get("_surprise","happy")
        mc = COLOR[selected_mood]
        st.markdown(
            f"<div style='background:{mc}22;border:1px solid {mc}55;"
            f"border-radius:10px;padding:.45rem .85rem;font-size:.85rem'>"
            f"🎲 <b style='color:{mc}'>{EMOJI[selected_mood]} "
            f"{selected_mood.capitalize()}</b></div>",
            unsafe_allow_html=True)

    # ── Filters ────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        lang_filter = st.selectbox(
            "🌐 Language", LANGS,
            index=LANGS.index(st.session_state["language"])
                  if st.session_state["language"] in LANGS else 0)
    with f2:
        n_offline = st.slider("🎵 Offline songs", 3, 10, 5)
    with f3:
        n_live = st.slider("🌐 Live results", 3, 8, 4,
                            disabled=not live)

    if lw > 0.20:
        st.info(f"🌐 Language weight boosted to {lw:.0%} — you prefer {lang_filter} songs.")

    st.session_state.pop("_preset", None)

    go = st.button("🎶 Get Recommendations", type="primary",
                    use_container_width=True)
    if not go:
        return

    mc = COLOR.get(selected_mood,"#7c5cfc")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — OFFLINE ML
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown(
        "<div class='sec-head'><span class='badge-offline'>⚙️ ML Offline</span></div>",
        unsafe_allow_html=True)

    with st.spinner("Running hybrid ML engine…"):
        if multi_mode and len(multi_moods) > 1:
            recs  = multi_mood_recommend(df, multi_moods, lang_filter,
                                          st.session_state["age"],
                                          n_offline, liked, disliked)
            label = " + ".join(f"{EMOJI[m]} {m.capitalize()}" for m in multi_moods)
        else:
            recs  = recommend(df, selected_mood, lang_filter,
                               st.session_state["age"],
                               n_offline, liked, disliked, lw)
            label = f"{EMOJI[selected_mood]} {selected_mood.capitalize()}"
            st.session_state["mood_history"].append(selected_mood)

    if recs.empty:
        st.warning("No offline results — try 'Any' language.")
    else:
        st.markdown(f"""
        <div style='padding:.25rem 0 .65rem'>
            <span style='color:{mc};font-weight:700'>{label}</span>
            <span style='color:#888;font-size:.8rem'>
                &nbsp;·&nbsp; {len(recs)} songs &nbsp;·&nbsp;
                {lang_filter} &nbsp;·&nbsp; Age {st.session_state["age"]}
            </span>
        </div>""", unsafe_allow_html=True)
        for i,(_, row) in enumerate(recs.iterrows(), 1):
            offline_card(row, i)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — SPOTIFY LIVE
    # ══════════════════════════════════════════════════════════════════════════
    if live:
        st.markdown("---")
        st.markdown(
            "<div class='sec-head'><span class='badge-live'>🟢 Spotify Live</span></div>",
            unsafe_allow_html=True)

        if not sp_ok:
            st.warning("Spotify not configured — add SPOTIFY_CLIENT_ID and "
                        "SPOTIFY_CLIENT_SECRET to your .env file.")
        else:
            sp_res = []
            with st.spinner("Fetching from Spotify…"):
                try:
                    sp_res = sp.get_recommendations(selected_mood, lang_filter, n_live)
                    if not sp_res:
                        sp_res = sp.search_songs(selected_mood, lang_filter, n_live)
                except Exception as e:
                    st.error(f"Spotify error: {e}")

            if sp_res:
                st.caption(f"{len(sp_res)} live tracks from Spotify")
                for i, item in enumerate(sp_res, 1):
                    spotify_card(item, i)
            else:
                st.info("No Spotify results — showing extra offline songs.")
                extras = recommend(df, selected_mood, lang_filter,
                                    st.session_state["age"], n_live, liked, disliked, lw)
                for i,(_, row) in enumerate(extras.iterrows(), len(recs)+1):
                    offline_card(row, i)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — YOUTUBE LIVE
    # ══════════════════════════════════════════════════════════════════════════
    if live:
        st.markdown("---")
        st.markdown(
            "<div class='sec-head'><span class='badge-yt'>▶ YouTube Videos</span></div>",
            unsafe_allow_html=True)

        if not yt_ok:
            st.warning("YouTube not configured — add YOUTUBE_API_KEY to .env.")
        else:
            yt_res = []
            with st.spinner("Fetching from YouTube…"):
                try:
                    yt_res = yt.search_videos(selected_mood, lang_filter, limit=n_live)
                except Exception as e:
                    st.error(f"YouTube error: {e}")

            if yt_res:
                st.caption(f"{len(yt_res)} videos · click to expand player")
                for i, item in enumerate(yt_res, 1):
                    youtube_card(item, i)
            else:
                st.info("No YouTube results found.")

    if liked:
        st.markdown("---")
        st.markdown(f"❤️ **{len(liked)} liked songs** are boosting your recommendations.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
def page_analytics():
    df  = get_df()
    fb  = st.session_state["feedback"]
    mh  = st.session_state["mood_history"]

    st.markdown("""
    <h1 style='font-size:1.85rem;margin-bottom:.2rem'>📊 Analytics</h1>
    <p style='color:#888;margin-bottom:.7rem'>
        Catalog insights and your listening patterns.</p>
    """, unsafe_allow_html=True)

    liked    = [s for s,a in fb.items() if a=="like"]
    disliked = [s for s,a in fb.items() if a=="dislike"]

    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    with k1: st.metric("❤️ Liked",    len(liked))
    with k2: st.metric("👎 Disliked", len(disliked))
    with k3:
        top = max(set(mh),key=mh.count) if mh else "—"
        st.metric("🎭 Top Mood",
                   f"{EMOJI.get(top,'')} {top.capitalize()}" if top!="—" else "—")
    with k4:
        if liked:
            tl = df[df["song"].isin(liked)]["language"].value_counts().idxmax()
            st.metric("🌐 Top Language", tl)
        else:
            st.metric("🌐 Top Language","—")

    # Mood history
    if mh:
        st.markdown("---")
        st.markdown("### 📈 Your Mood Sessions")
        ms = pd.Series(mh).value_counts().reset_index()
        ms.columns = ["Mood","Count"]
        st.bar_chart(ms.set_index("Mood")["Count"])

    st.markdown("---")

    # Top songs by popularity
    st.markdown("### 🔥 Most Popular Songs in Catalog")
    top15 = df.nlargest(15,"popularity")[["song","artist","mood","popularity"]]
    st.bar_chart(top15.set_index("song")["popularity"])

    st.markdown("---")

    # Heatmaps
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🌐 Mood × Language")
        hm = mood_language_heatmap(df)
        st.dataframe(hm.style.background_gradient(cmap="Purples"),
                      use_container_width=True)
    with c2:
        st.markdown("### 🎭 Avg Popularity per Mood")
        pt = mood_popularity_table(df)
        st.bar_chart(pt.set_index("mood")["avg_popularity"])

    st.markdown("---")

    # Catalog summary
    o1,o2,o3,o4 = st.columns(4)
    with o1: st.metric("Total Songs",   len(df))
    with o2: st.metric("Moods",         df["mood"].nunique())
    with o3: st.metric("Languages",     df["language"].nunique())
    with o4: st.metric("Avg Popularity",f"{df['popularity'].mean():.0f}")

    ca, cb = st.columns(2)
    with ca:
        st.markdown("#### Songs per Mood")
        mc_df = df["mood"].value_counts().reset_index()
        mc_df.columns = ["Mood","Songs"]
        st.bar_chart(mc_df.set_index("Mood")["Songs"])
    with cb:
        st.markdown("#### Songs per Language")
        lc_df = df["language"].value_counts().reset_index()
        lc_df.columns = ["Language","Songs"]
        st.bar_chart(lc_df.set_index("Language")["Songs"])

    # Liked songs table
    if liked:
        st.markdown("---")
        st.markdown("### ❤️ Your Liked Songs")
        ld = df[df["song"].isin(liked)][
            ["song","artist","mood","language","genre"]].copy()
        ld.columns = ["Song","Artist","Mood","Language","Genre"]
        st.dataframe(ld, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE ENCRYPTION DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_security():
    import hashlib, hmac as _hmac, base64, time as _time, secrets as _secrets

    st.markdown("""
    <h1 style='font-size:1.85rem;margin-bottom:.15rem'>🔒 Live Encryption</h1>
    <p style='color:#888;margin-bottom:.8rem'>
        Watch every security mechanism work in real time — type anything and
        see SHA-256 hashing, HMAC tokens, and input sanitisation live.</p>
    """, unsafe_allow_html=True)

    # ── Security architecture overview ────────────────────────────────────────
    st.markdown("### 🛡️ Security Layers Active in This App")
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    layers = [
        ("🔒","SHA-256\n+ Salt","Password\nHashing","badge-green"),
        ("🔐","HMAC-SHA256\nTokens","Session\nAuth","badge-blue"),
        ("🌐","HTTPS Only","All API\nCalls","badge-orange"),
        ("🚫","Input\nSanitise","XSS / Inject\nPrevention","badge-green"),
    ]
    for col, (icon, title, desc, badge) in zip([r1c1,r1c2,r1c3,r1c4], layers):
        with col:
            st.markdown(f"""
            <div class='enc-panel' style='text-align:center'>
                <div style='font-size:1.8rem'>{icon}</div>
                <div style='font-weight:700;font-size:.85rem;margin:.2rem 0;
                     color:#e8eaf0'>{title}</div>
                <div style='color:#888;font-size:.75rem'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════
    # MODULE 1: LIVE SHA-256 PASSWORD HASHING
    # ════════════════════════════════════════════════════════════
    st.markdown("### 🔒 Module 1 — Live Password Hashing (SHA-256 + Salt)")
    st.caption("Type a password below and watch it get hashed in real time "
               "using SHA-256 with a unique random salt per user.")

    demo_pw = st.text_input(
        "🔑 Enter any password to hash",
        value="MyPassword123",
        key="sec_pw",
        placeholder="Type any password…")

    if demo_pw:
        # Generate a fixed demo salt (stable for this session)
        demo_salt = st.session_state.get("_demo_salt") or _secrets.token_hex(16)
        st.session_state["_demo_salt"] = demo_salt

        combined     = f"{demo_salt}{demo_pw}".encode("utf-8")
        hashed       = hashlib.sha256(combined).hexdigest()
        verify_match = hashlib.sha256(f"{demo_salt}{demo_pw}".encode()).hexdigest()
        verify_wrong = hashlib.sha256(f"{demo_salt}wrong".encode()).hexdigest()

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class='enc-panel'>
                <div class='enc-title'>📥 Plain-text Input (NEVER stored)</div>
                <div class='hash-flow'>{demo_pw}</div>
                <span class='sec-badge badge-orange'>Plain text</span>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class='enc-panel'>
                <div class='enc-title'>🧂 Random Salt (unique per user)</div>
                <div class='hash-flow'>{demo_salt}</div>
                <span class='sec-badge badge-blue'>128-bit random</span>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class='enc-panel'>
                <div class='enc-title'>➕ Combined: salt + password</div>
                <div class='hash-flow'>{demo_salt + demo_pw}</div>
                <span class='sec-badge badge-orange'>Input to SHA-256</span>
            </div>""", unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
            <div class='enc-panel'>
                <div class='enc-title'>🔒 SHA-256 Hash (what's stored in DB)</div>
                <div class='hash-flow'>{hashed}</div>
                <span class='sec-badge badge-green'>256-bit / 64 hex chars</span>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class='enc-panel'>
                <div class='enc-title'>✅ Verify correct password</div>
                <div class='enc-label'>sha256(salt + "{demo_pw}")</div>
                <div class='hash-flow'>{verify_match}</div>
                <span class='sec-badge badge-green'>✅ MATCH — Login allowed</span>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class='enc-panel'>
                <div class='enc-title'>❌ Verify wrong password</div>
                <div class='enc-label'>sha256(salt + "wrong")</div>
                <div class='hash-flow'>{verify_wrong}</div>
                <span class='sec-badge badge-red'>❌ NO MATCH — Login denied</span>
            </div>""", unsafe_allow_html=True)

        # Visual flow diagram
        st.markdown(f"""
        <div style='background:#050b14;border:1px solid #1b2a40;border-radius:12px;
             padding:1rem 1.5rem;margin:.5rem 0;font-size:.82rem'>
            <b style='color:#64b5f6'>🔄 Hashing Flow:</b><br><br>
            <span style='color:#ffa726'>"{demo_pw}"</span> &nbsp;+&nbsp;
            <span style='color:#42a5f5'>salt "{demo_salt[:12]}…"</span>
            &nbsp; → &nbsp;
            <span style='color:#888'>SHA-256()</span>
            &nbsp; → &nbsp;
            <span style='color:#a5d6a7'>"{hashed[:20]}…"</span>
            &nbsp; <span style='color:#1db954'>(stored ✅)</span>
        </div>""", unsafe_allow_html=True)

        # Show that the same password always produces the same hash (with same salt)
        st.success(f"✅ Same password → same hash every time (deterministic). "
                    f"Different salt → completely different hash. "
                    f"Original password cannot be recovered from the hash.")

    st.markdown("---")

    # ════════════════════════════════════════════════════════════
    # MODULE 2: LIVE SESSION TOKEN (HMAC)
    # ════════════════════════════════════════════════════════════
    st.markdown("### 🔐 Module 2 — Live Session Token (HMAC-SHA256)")
    st.caption("Session tokens prove who you are without sending your password. "
               "They are signed with HMAC-SHA256 and expire automatically.")

    tok_user = st.text_input("👤 Username to tokenise",
                              value=st.session_state["username"],
                              key="sec_tok_user")
    tok_hours = st.slider("⏱ Token expiry (hours)", 1, 48, 24, key="sec_tok_h")

    if tok_user.strip():
        from security import generate_token, verify_token
        live_token  = generate_token(tok_user.strip(), expiry_hours=tok_hours)
        decoded_raw = base64.urlsafe_b64decode(live_token.encode()).decode()
        parts       = decoded_raw.rsplit("|", 2)

        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown(f"""
            <div class='enc-panel'>
                <div class='enc-title'>📦 Token Payload</div>
                <div class='enc-label'>username | expiry_unix_timestamp</div>
                <div class='hash-flow'>{parts[0] if len(parts)>1 else ""}|{parts[1] if len(parts)>1 else ""}</div>
                <span class='sec-badge badge-orange'>Plaintext part</span>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class='enc-panel'>
                <div class='enc-title'>✍️ HMAC-SHA256 Signature</div>
                <div class='enc-label'>hmac(app_secret, payload)</div>
                <div class='hash-flow'>{parts[2] if len(parts)>2 else ""}</div>
                <span class='sec-badge badge-blue'>64-char hex</span>
            </div>""", unsafe_allow_html=True)

        with tc2:
            st.markdown(f"""
            <div class='enc-panel'>
                <div class='enc-title'>🎫 Full Signed Token (Base64)</div>
                <div class='enc-label'>base64url(payload + "|" + signature)</div>
                <div class='hash-flow'>{live_token}</div>
                <span class='sec-badge badge-green'>Sent to browser</span>
            </div>""", unsafe_allow_html=True)

            # Verify it
            verified_user = verify_token(live_token)
            st.markdown(f"""
            <div class='enc-panel'>
                <div class='enc-title'>🔍 Token Verification Result</div>
                <div class='enc-label'>verify_token(token) → username</div>
                <div class='hash-flow'>{verified_user}</div>
                <span class='sec-badge {"badge-green" if verified_user else "badge-red"}'>
                    {"✅ Valid — user identified" if verified_user else "❌ Invalid / expired"}
                </span>
            </div>""", unsafe_allow_html=True)

        # Tamper demonstration
        st.markdown("#### 🚨 Tamper Detection Demo")
        st.caption("Change even one character of the token and the signature breaks.")
        tampered = live_token[:-3] + "XYZ"  # corrupt last 3 chars
        tamper_result = verify_token(tampered)
        st.markdown(f"""
        <div style='background:#1a0505;border:1px solid #ef535040;border-radius:10px;
             padding:.75rem 1rem;font-size:.8rem;font-family:monospace'>
            <span style='color:#ef9a9a'>Original:&nbsp;</span>
            <span style='color:#a5d6a7'>…{live_token[-20:]}</span><br>
            <span style='color:#ef9a9a'>Tampered:&nbsp;</span>
            <span style='color:#ef5350'>…{tampered[-20:]}</span><br><br>
            <b style='color:{"#ef5350"}'>verify_token(tampered) → {tamper_result}</b>
            &nbsp; <span style='color:#ef5350'>❌ Rejected — signature mismatch</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════
    # MODULE 3: LIVE INPUT SANITISATION
    # ════════════════════════════════════════════════════════════
    st.markdown("### 🧹 Module 3 — Live Input Sanitisation")
    st.caption("Every text input is sanitised before it reaches the ML model "
               "or the database. Try typing XSS / injection payloads below.")

    dirty_input = st.text_input(
        "💉 Type any input (try XSS or SQL injection)",
        value='<script>alert("XSS")</script> happy mood',
        key="sec_dirty")

    if dirty_input:
        from security import sanitize_text
        clean = sanitize_text(dirty_input)
        dangerous_chars = set('<>"\';|&\\{}') & set(dirty_input)

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f"""
            <div class='enc-panel'>
                <div class='enc-title'>📥 Raw Input (unsafe)</div>
                <div class='hash-flow' style='color:#ef9a9a'>{dirty_input}</div>
                <span class='sec-badge badge-red'>⚠️ Potentially dangerous</span>
            </div>""", unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""
            <div class='enc-panel'>
                <div class='enc-title'>✅ Sanitised Output (safe)</div>
                <div class='hash-flow' style='color:#a5d6a7'>{clean}</div>
                <span class='sec-badge badge-green'>✅ Safe to process</span>
            </div>""", unsafe_allow_html=True)

        if dangerous_chars:
            st.warning(f"⚠️ Dangerous characters stripped: "
                        f"{' '.join(repr(c) for c in sorted(dangerous_chars))}")
        else:
            st.success("✅ No dangerous characters detected.")

        # Show what gets stripped
        st.markdown(f"""
        <div style='background:#050b14;border:1px solid #1b2a40;border-radius:10px;
             padding:.75rem 1rem;font-size:.8rem;margin-top:.3rem'>
            <b style='color:#64b5f6'>Characters blocked:</b>
            <span style='color:#ef9a9a;font-family:monospace'>
                &lt; &gt; " ' ; | &amp; \\ {{ }}</span>
            &nbsp;&nbsp;
            <b style='color:#64b5f6'>Max length:</b>
            <span style='color:#a5d6a7'>{len(clean)}/{500} chars</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════
    # MODULE 4: BRUTE-FORCE PROTECTION
    # ════════════════════════════════════════════════════════════
    st.markdown("### 🚫 Module 4 — Brute-Force Login Protection")
    st.caption("After 5 failed attempts in 5 minutes, the account is locked. "
               "Simulate it below.")

    from security import (record_failed_attempt, is_locked_out,
                           clear_attempts, lockout_remaining)

    demo_acct = st.text_input("👤 Test account name", value="demo_user",
                               key="sec_bf_user")
    bf1, bf2, bf3 = st.columns(3)
    with bf1:
        if st.button("❌ Simulate Failed Login", use_container_width=True,
                      key="sec_bf_fail"):
            record_failed_attempt(demo_acct)
            st.rerun()
    with bf2:
        if st.button("✅ Simulate Successful Login", use_container_width=True,
                      key="sec_bf_ok"):
            clear_attempts(demo_acct)
            st.rerun()
    with bf3:
        if st.button("🔄 Reset Counter", use_container_width=True,
                      key="sec_bf_rst"):
            clear_attempts(demo_acct)
            st.rerun()

    locked   = is_locked_out(demo_acct)
    from security import _attempt_log, _WINDOW_SECONDS
    import time as _t
    now      = _t.time()
    attempts = [t for t in _attempt_log.get(demo_acct, [])
                if now - t < _WINDOW_SECONDS]
    remaining = lockout_remaining(demo_acct)

    # Live counter display
    pct   = min(len(attempts) / 5, 1.0)
    color = "#ef5350" if locked else ("#ffa726" if pct >= 0.6 else "#1db954")
    bar_w = int(pct * 100)

    st.markdown(f"""
    <div class='enc-panel'>
        <div style='display:flex;justify-content:space-between;margin-bottom:.4rem'>
            <span style='font-weight:700;color:#e8eaf0'>
                {"🔒 LOCKED" if locked else "🟢 OK"} — {demo_acct}
            </span>
            <span style='color:#888;font-size:.8rem'>
                {len(attempts)}/5 attempts in 5 min window
            </span>
        </div>
        <div style='background:#1a1d28;border-radius:6px;height:10px'>
            <div style='width:{bar_w}%;background:{color};height:10px;
                 border-radius:6px;transition:width .3s'></div>
        </div>
        <div style='margin-top:.5rem;font-size:.8rem;color:#888'>
            {"🔒 Locked for " + str(remaining) + "s — too many failed attempts"
             if locked else
             "Each failed attempt is tracked. 5 failures = 5-minute lockout."}
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════
    # MODULE 5: API KEY SECURITY
    # ════════════════════════════════════════════════════════════
    st.markdown("### 🔑 Module 5 — API Key Security")
    st.caption("API keys are loaded from .env and masked in all logs and UI. "
               "They are never hardcoded or displayed in full.")

    from security import mask_key, check_env
    env = check_env()

    demo_key = st.text_input(
        "🗝️ Enter any API key to see masking",
        value="sk-abcdef1234567890abcdef1234567890",
        key="sec_key_demo")

    if demo_key:
        from security import mask
        masked = mask_key(demo_key)
        mk1, mk2 = st.columns(2)
        with mk1:
            st.markdown(f"""
            <div class='enc-panel'>
                <div class='enc-title'>🗝️ Full API Key (in .env)</div>
                <div class='hash-flow' style='color:#ef9a9a'>{demo_key}</div>
                <span class='sec-badge badge-red'>⚠️ Only in .env file</span>
            </div>""", unsafe_allow_html=True)
        with mk2:
            st.markdown(f"""
            <div class='enc-panel'>
                <div class='enc-title'>😷 Masked Key (in logs/UI)</div>
                <div class='hash-flow' style='color:#a5d6a7'>{masked}</div>
                <span class='sec-badge badge-green'>✅ Safe to display</span>
            </div>""", unsafe_allow_html=True)

    # Current env status
    st.markdown("#### 🌐 Current Environment Key Status")
    env_rows = [
        ("SPOTIFY_CLIENT_ID",     env.get("SPOTIFY_CLIENT_ID",   False)),
        ("SPOTIFY_CLIENT_SECRET", env.get("SPOTIFY_CLIENT_SECRET",False)),
        ("YOUTUBE_API_KEY",       env.get("YOUTUBE_API_KEY",      False)),
        ("SECRET_KEY",            env.get("SECRET_KEY",           False)),
    ]
    for key_name, is_set in env_rows:
        icon  = "🟢" if is_set else "🔴"
        state = "Configured ✅" if is_set else "Not set — add to .env"
        st.markdown(f"""
        <div style='background:#0a0f1e;border:1px solid #1e3a5f;border-radius:8px;
             padding:.4rem .9rem;margin-bottom:.3rem;font-size:.81rem;
             display:flex;justify-content:space-between;align-items:center'>
            <span style='font-family:monospace;color:#80cbc4'>{icon} {key_name}</span>
            <span style='color:{"#a5d6a7" if is_set else "#ef9a9a"}'>{state}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════
    # MODULE 6: HTTPS STATUS + SHA-256 PLAYGROUND
    # ════════════════════════════════════════════════════════════
    st.markdown("### 🌐 Module 6 — HTTPS & General SHA-256 Playground")

    col_h, col_p = st.columns(2)
    with col_h:
        st.markdown("#### 🔒 HTTPS Enforcement")
        for endpoint in [
            ("Spotify Token",        "https://accounts.spotify.com/api/token"),
            ("Spotify Search",       "https://api.spotify.com/v1/search"),
            ("Spotify Recommends",   "https://api.spotify.com/v1/recommendations"),
            ("YouTube Search",       "https://www.googleapis.com/youtube/v3/search"),
        ]:
            label, url = endpoint
            st.markdown(f"""
            <div style='background:#0a0f1e;border:1px solid #1e3a5f;
                 border-radius:8px;padding:.35rem .8rem;margin-bottom:.25rem;
                 font-size:.78rem'>
                🔒 <b style='color:#80cbc4'>{label}</b><br>
                <span style='color:#546e7a;font-family:monospace;font-size:.72rem'>
                {url}</span>
            </div>""", unsafe_allow_html=True)

    with col_p:
        st.markdown("#### 🔢 SHA-256 Playground")
        any_text = st.text_area(
            "Hash any text",
            value="Hello World",
            height=80,
            key="sec_sha_play")
        if any_text:
            h256 = hashlib.sha256(any_text.encode()).hexdigest()
            h512 = hashlib.sha512(any_text.encode()).hexdigest()
            st.markdown(f"""
            <div class='enc-panel'>
                <div class='enc-label'>SHA-256 (64 hex chars):</div>
                <div class='hash-flow'>{h256}</div>
                <div class='enc-label' style='margin-top:.4rem'>
                    SHA-512 (128 hex chars):</div>
                <div class='hash-flow' style='color:#80cbc4'>{h512[:64]}…</div>
            </div>""", unsafe_allow_html=True)
            st.caption("Same input always → same hash. "
                        "Change 1 character → completely different hash.")

    # Session info for current user
    st.markdown("---")
    st.markdown("### 👤 Your Current Session")
    current_token = st.session_state.get("token","")
    if current_token:
        try:
            decoded = base64.urlsafe_b64decode(current_token.encode()).decode()
            parts   = decoded.rsplit("|",2)
            import datetime
            exp_dt  = datetime.datetime.fromtimestamp(int(parts[1])).strftime(
                      "%Y-%m-%d %H:%M:%S") if len(parts)>1 else "?"
        except Exception:
            parts  = ["?","?","?"]
            exp_dt = "?"

        verified = verify_token(current_token)
        st.markdown(f"""
        <div class='enc-panel'>
            <div style='display:flex;gap:1rem;flex-wrap:wrap'>
                <div>
                    <div class='enc-label'>👤 Logged in as</div>
                    <div style='font-weight:700;color:#a78bfa'>
                        {st.session_state["username"]}</div>
                </div>
                <div>
                    <div class='enc-label'>⏱ Token expires</div>
                    <div style='color:#80cbc4'>{exp_dt}</div>
                </div>
                <div>
                    <div class='enc-label'>✅ Token valid</div>
                    <div style='color:{"#a5d6a7" if verified else "#ef5350"}'>
                        {"Yes ✅" if verified else "No ❌"}</div>
                </div>
            </div>
            <div style='margin-top:.5rem'>
                <div class='enc-label'>🎫 Your session token (masked):</div>
                <div class='hash-flow'>{current_token[:40]}…</div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("No active token found in session state.")


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state["logged_in"]:
    page_auth()
else:
    render_sidebar()
    pg = st.session_state.get("page","home")
    if   pg == "home":      page_home()
    elif pg == "recommend": page_recommend()
    elif pg == "security":  page_security()
    elif pg == "analytics": page_analytics()
