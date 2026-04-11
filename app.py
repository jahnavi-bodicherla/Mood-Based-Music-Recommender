"""
MoodTunes v2 — Production-Ready Mood-Based Music Recommender
────────────────────────────────────────────────────────────
Run:  streamlit run app.py
"""

import csv, hashlib, random, re, time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(encoding="utf-8")   # load .env (handles encoding gracefully)

from recommender import (
    ALL_MOODS, MOOD_PROFILES, detect_mood, load_songs,
    recommend, multi_mood_recommend, compute_language_weight,
    mood_lang_pivot, mood_age_pivot,
)
import spotify_client as sp

# ── Page setup ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="MoodTunes 🎵", page_icon="🎵",
                   layout="wide", initial_sidebar_state="expanded")

# ── Paths ────────────────────────────────────────────────────────────────────
USERS_CSV    = Path("users.csv")
FEEDBACK_CSV = Path("feedback.csv")

# ── Mood metadata ─────────────────────────────────────────────────────────────
EMOJI = {"happy":"😊","sad":"😔","energetic":"⚡","relaxed":"😌",
         "romantic":"💕","angry":"😤","chill":"🧊","focus":"🧠",
         "party":"🎉","devotional":"🙏"}
COLOR = {"happy":"#F9C846","sad":"#5BA4CF","energetic":"#FF5F57",
         "relaxed":"#42C89B","romantic":"#EC4899","angry":"#F97316",
         "chill":"#60A5FA","focus":"#A855F7","party":"#FBBF24","devotional":"#D97706"}
LANGS = ["Any","English","Hindi","Telugu","Tamil"]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:.8rem}

.card{background:#1a1d28;border:1px solid #2e3250;border-radius:14px;
      padding:.9rem 1.1rem;margin-bottom:.45rem;transition:border-color .2s,transform .15s}
.card:hover{border-color:#7c5cfc;transform:translateY(-1px)}
.ctitle{font-weight:700;font-size:.97rem;margin-bottom:.08rem}
.csub{color:#888;font-size:.8rem;margin-bottom:.3rem}
.pill{display:inline-block;background:rgba(124,92,252,.14);color:#a78bfa;
      border-radius:20px;padding:.05rem .44rem;font-size:.67rem;font-weight:600;margin:.05rem .08rem 0 0}
.mpill{display:inline-block;border-radius:20px;padding:.05rem .44rem;
       font-size:.67rem;font-weight:600;margin:.05rem .08rem 0 0}

.sp-card{background:#091610;border:1px solid #1db95440;border-radius:12px;
         padding:.85rem 1rem;margin-bottom:.45rem}
.sp-title{font-weight:700;font-size:.95rem;margin-bottom:.08rem}
.sp-sub{color:#888;font-size:.8rem;margin-bottom:.3rem}

.badge-green{display:inline-block;background:rgba(29,185,84,.11);color:#1db954;
  border:1px solid rgba(29,185,84,.3);border-radius:20px;padding:.15rem .65rem;
  font-size:.72rem;font-weight:700}
.badge-purple{display:inline-block;background:rgba(124,92,252,.14);color:#a78bfa;
  border:1px solid rgba(124,92,252,.3);border-radius:20px;padding:.15rem .65rem;
  font-size:.72rem;font-weight:700}
.badge-orange{display:inline-block;background:rgba(255,152,0,.12);color:#ffa726;
  border:1px solid rgba(255,152,0,.3);border-radius:20px;padding:.15rem .65rem;
  font-size:.72rem;font-weight:700}

[data-testid="stMetric"]{background:#1a1d28;border:1px solid #2e3250;
    border-radius:12px;padding:.75rem!important}
[data-testid="stMetricValue"]{color:#7c5cfc!important;font-weight:700!important}
.stTextInput>div>div>input{background:#10131e!important;border:1px solid #2e3250!important;
    border-radius:10px!important;color:#e8eaf0!important}
.stTextInput>div>div>input:focus{border-color:#7c5cfc!important;box-shadow:none!important}
.stButton>button{border-radius:10px!important;font-weight:600!important}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# AUTH UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def _sanitize(text: str, max_len: int = 300) -> str:
    return re.sub(r'[<>"\';|&\\{}]', " ", str(text).strip())[:max_len]

def load_users() -> dict:
    if not USERS_CSV.exists(): return {}
    users = {}
    try:
        with open(USERS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                users[row["username"]] = {
                    "password": row["password"],
                    "age":      int(row.get("age", 22)),
                    "language": row.get("preferred_language","Any"),
                }
    except Exception: pass
    return users

def save_user(username, password, age, language):
    exists = USERS_CSV.exists()
    with open(USERS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["username","password","age","preferred_language"])
        if not exists: w.writeheader()
        w.writerow({"username":username,"password":_hash(password),
                    "age":age,"preferred_language":language})

def load_feedback(username: str) -> dict:
    if not FEEDBACK_CSV.exists(): return {}
    fb = {}
    try:
        with open(FEEDBACK_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("username") == username:
                    fb[row["song"]] = row["action"]
    except Exception: pass
    return fb

def save_feedback_row(username, song, action):
    rows = []
    if FEEDBACK_CSV.exists():
        try:
            with open(FEEDBACK_CSV, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            rows = [r for r in rows if not (r["username"]==username and r["song"]==song)]
        except Exception: pass
    rows.append({"username":username,"song":song,"action":action,
                 "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M")})
    with open(FEEDBACK_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["username","song","action","timestamp"])
        w.writeheader(); w.writerows(rows)

@st.cache_data(show_spinner=False)
def get_df(): return load_songs()

def liked_disliked(fb):
    return ([s for s,a in fb.items() if a=="like"],
            [s for s,a in fb.items() if a=="dislike"])


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = dict(logged_in=False, username="", age=22, language="Any",
                 page="home", feedback={}, mood_history=[],
                 auth_tab="login", spotify_token=None,
                 sp_oauth_state=None, _attempts={})
for k,v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Handle Spotify OAuth callback in URL ──────────────────────────────────────
_qp = st.query_params
if "code" in _qp and st.session_state.get("logged_in"):
    code  = _qp.get("code","")
    state = _qp.get("state","")
    if code and state == st.session_state.get("sp_oauth_state"):
        with st.spinner("Connecting to Spotify…"):
            token = sp.exchange_code(code)
            if token:
                st.session_state["spotify_token"] = token
                st.query_params.clear()
                st.rerun()
            else:
                st.error("Spotify connection failed. Please try again.")
                st.query_params.clear()


# ══════════════════════════════════════════════════════════════════════════════
# AUTH PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_auth():
    st.markdown("""
    <div style='text-align:center;padding:1.2rem 0 .5rem'>
        <div style='font-size:3rem'>🎵</div>
        <h1 style='font-size:1.9rem;margin:.15rem 0;
            background:linear-gradient(135deg,#7c5cfc,#e85d9b);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
            MoodTunes</h1>
        <p style='color:#888;font-size:.87rem'>
            Mood-based music · ML + Spotify · 10 moods</p>
    </div>""", unsafe_allow_html=True)

    tab = st.session_state["auth_tab"]
    c1,c2 = st.columns(2)
    with c1:
        if st.button("🔐 Login", use_container_width=True,
                      type="primary" if tab=="login" else "secondary"):
            st.session_state["auth_tab"]="login"; st.rerun()
    with c2:
        if st.button("✏️ Sign Up", use_container_width=True,
                      type="primary" if tab=="signup" else "secondary"):
            st.session_state["auth_tab"]="signup"; st.rerun()

    st.markdown("<br>",unsafe_allow_html=True)
    users = load_users()

    if tab == "login":
        st.markdown("<h3 style='text-align:center'>Welcome back 👋</h3>",
                    unsafe_allow_html=True)
        with st.form("lf"):
            uname = st.text_input("👤 Username", placeholder="your_username")
            pw    = st.text_input("🔒 Password", type="password", placeholder="••••••••")
            if st.form_submit_button("Sign In →", use_container_width=True):
                uname = _sanitize(uname)
                # Brute-force check
                attempts = st.session_state["_attempts"]
                info = attempts.get(uname, {"count":0,"since":0})
                if info["count"] >= 5 and time.time()-info["since"] < 300:
                    st.error(f"🔒 Too many attempts. Wait {int(300-(time.time()-info['since']))}s.")
                elif not uname or not pw:
                    st.error("Both fields are required.")
                elif uname not in users:
                    st.error("Username not found.")
                elif users[uname]["password"] != _hash(pw):
                    info = {"count": info["count"]+1, "since": info.get("since") or time.time()}
                    attempts[uname] = info
                    remaining = 5 - info["count"]
                    st.error(f"Incorrect password. {remaining} attempt(s) left.")
                else:
                    attempts.pop(uname, None)
                    st.session_state.update({
                        "logged_in":True,"username":uname,
                        "age":users[uname]["age"],
                        "language":users[uname]["language"],
                        "feedback":load_feedback(uname),
                        "page":"home",
                    })
                    st.rerun()

    else:  # signup
        st.markdown("<h3 style='text-align:center'>Create Account 🎧</h3>",
                    unsafe_allow_html=True)
        with st.form("sf"):
            cu,cpw = st.columns(2)
            with cu:  nu  = st.text_input("👤 Username *", placeholder="cool_listener")
            with cpw: npw = st.text_input("🔒 Password *", type="password",
                                           placeholder="Min 6 chars")
            npw2 = st.text_input("🔒 Confirm *", type="password", placeholder="Repeat password")
            ca,cl = st.columns(2)
            with ca: age  = st.number_input("🎂 Age", 10, 99, 22)
            with cl: lang = st.selectbox("🌐 Language", LANGS)
            if st.form_submit_button("Create Account →", use_container_width=True):
                nu = _sanitize(nu)
                errs = []
                if not re.match(r'^[a-zA-Z0-9_]{3,30}$', nu):
                    errs.append("Username: 3-30 chars, letters/numbers/underscore only.")
                if len(npw) < 6: errs.append("Password must be at least 6 characters.")
                if npw != npw2:  errs.append("Passwords do not match.")
                if nu in users:  errs.append("Username already taken.")
                if errs:
                    for e in errs: st.error(e)
                else:
                    save_user(nu, npw, int(age), lang)
                    st.success("✅ Account created! Sign in above.")
                    st.session_state["auth_tab"]="login"; st.rerun()

    st.markdown("""
    <p style='text-align:center;color:#444;font-size:.73rem;margin-top:1.4rem'>
        🔒 Passwords are SHA-256 hashed — never stored in plain text</p>""",
    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        fb  = st.session_state["feedback"]
        mh  = st.session_state["mood_history"]
        lkd = sum(1 for a in fb.values() if a=="like")
        top = max(set(mh), key=mh.count) if mh else "—"

        # Spotify connect button
        sp_token = st.session_state.get("spotify_token")
        sp_valid = sp.is_token_valid(sp_token)

        st.markdown(f"""
        <div style='background:#1a1d28;border:1px solid #2e3250;border-radius:12px;
             padding:.75rem 1rem;margin-bottom:.85rem'>
            <div style='font-size:1.5rem;text-align:center'>🎵</div>
            <div style='font-weight:800;font-size:.98rem;text-align:center;
                 background:linear-gradient(135deg,#7c5cfc,#e85d9b);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
                 MoodTunes</div>
            <div style='color:#888;font-size:.74rem;text-align:center;margin-top:.25rem'>
                👤 {st.session_state["username"]} &nbsp;·&nbsp;
                🎂 {st.session_state["age"]}</div>
        </div>""", unsafe_allow_html=True)

        # ── Spotify connect ────────────────────────────────────────────────
        st.markdown("### 🎧 Spotify")
        if sp_valid:
            sp_user = sp_token.get("user", {})
            st.markdown(
                f"<span class='badge-green'>🟢 Connected</span>"
                f"<span style='color:#888;font-size:.76rem;margin-left:.5rem'>"
                f"{sp_user.get('display_name','User')}</span>",
                unsafe_allow_html=True)
            if st.button("Disconnect Spotify", use_container_width=True, key="sp_dc"):
                st.session_state["spotify_token"] = None; st.rerun()
        elif sp.is_configured():
            if st.button("🎧 Connect Spotify Account", use_container_width=True,
                          key="sp_conn"):
                url, state = sp.get_auth_url()
                st.session_state["sp_oauth_state"] = state
                st.markdown(
                    f"<meta http-equiv='refresh' content='0;url={url}'>",
                    unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge-orange'>⚙️ Not configured</span>",
                        unsafe_allow_html=True)
            st.caption("Add SPOTIFY_CLIENT_ID + SECRET to .env")

        st.markdown("---")
        st.markdown("### 📍 Navigation")
        for icon,pg,label in [("🏠","home","Home"),("🎵","recommend","Recommend"),
                               ("📊","analytics","Analytics")]:
            active = st.session_state["page"] == pg
            if st.button(f"{icon} {label}", key=f"nav_{pg}",
                          use_container_width=True,
                          type="primary" if active else "secondary"):
                st.session_state["page"]=pg; st.rerun()

        st.markdown("---")
        st.metric("❤️ Liked",   lkd)
        st.metric("🎭 Top Mood",
                   f"{EMOJI.get(top,'')} {top.capitalize()}" if top!="—" else "—")
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for k,v in _DEFAULTS.items(): st.session_state[k]=v
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# CARD RENDERERS
# ══════════════════════════════════════════════════════════════════════════════
def offline_card(row, idx):
    name   = row["song"]; artist = row["artist"]; mood = row.get("mood","happy")
    lang   = row.get("language",""); genre = row.get("genre","")
    energy = float(row.get("energy",0)); valence = float(row.get("valence",0))
    tempo  = int(row.get("tempo",0)); pop = int(row.get("popularity",0))
    mc     = COLOR.get(mood,"#7c5cfc")
    fb_s   = st.session_state["feedback"].get(name,"none")
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
        <span class='mpill' style='background:{mc}22;color:{mc}'>{EMOJI.get(mood,"")} {mood}</span>
        <span class='mpill' style='background:rgba(255,255,255,.05);color:#bbb'>🌐 {lang}</span>
    </div>""", unsafe_allow_html=True)

    b1,b2,_ = st.columns([1,1,7])
    with b1:
        if st.button("❤️" if fb_s=="like" else "👍",
                      key=f"like_{idx}_{name}", use_container_width=True):
            st.session_state["feedback"][name]="like"
            save_feedback_row(st.session_state["username"],name,"like"); st.rerun()
    with b2:
        if st.button("👎", key=f"dis_{idx}_{name}", use_container_width=True):
            st.session_state["feedback"][name]="dislike"
            save_feedback_row(st.session_state["username"],name,"dislike"); st.rerun()


def spotify_card(item, idx):
    name  = item.get("song",""); artist = item.get("artist","")
    album = item.get("album",""); img   = item.get("album_image")
    prev  = item.get("preview_url"); sp_url = item.get("spotify_url","#")
    pop   = item.get("popularity",0)

    ic,info = st.columns([1,5])
    with ic:
        if img:
            st.image(img, width=78)
        else:
            st.markdown("<div style='width:78px;height:78px;background:#0a1a0e;"
                        "border-radius:8px;display:flex;align-items:center;"
                        "justify-content:center;font-size:1.5rem'>🎵</div>",
                        unsafe_allow_html=True)
    with info:
        st.markdown(f"""
        <div class='sp-card'>
            <div class='sp-title'>
                <span style='color:#1db954;font-weight:800;margin-right:.3rem'>{idx}.</span>{name}
            </div>
            <div class='sp-sub'>
                🎤 {artist} &nbsp;·&nbsp; 💿 {album} &nbsp;·&nbsp; 🔥 {pop}
            </div>
            <span class='badge-green' style='font-size:.62rem'>🟢 Spotify Live</span>
        </div>""", unsafe_allow_html=True)
        lc,pc = st.columns([1,2])
        with lc:
            if sp_url and sp_url != "#":
                st.link_button("🎧 Open", sp_url, use_container_width=True)
        with pc:
            if prev:
                st.markdown(
                    f"<audio controls style='width:100%;height:34px;border-radius:8px'>"
                    f"<source src='{prev}' type='audio/mpeg'></audio>",
                    unsafe_allow_html=True)
            else:
                st.caption("No preview available")

    # Fetch and show audio features
    if item.get("track_uri","").startswith("spotify:track:"):
        track_id = item["track_uri"].split(":")[-1]
        tok      = st.session_state.get("spotify_token")
        feats    = sp.get_audio_features(track_id, tok)
        if feats:
            f1,f2,f3 = st.columns(3)
            with f1: st.caption(f"⚡ Energy: {feats.get('energy',0):.0%}")
            with f2: st.caption(f"😊 Valence: {feats.get('valence',0):.0%}")
            with f3: st.caption(f"🥁 Tempo: {feats.get('tempo',0):.0f} BPM")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    uname = st.session_state["username"]
    fb    = st.session_state["feedback"]
    mh    = st.session_state["mood_history"]
    sp_ok = sp.is_token_valid(st.session_state.get("spotify_token"))

    st.markdown(f"""
    <div style='padding:.3rem 0 .9rem'>
        <h1 style='font-size:1.85rem;margin-bottom:.2rem'>
            🎵 Welcome, <span style='color:#a78bfa'>{uname}</span>!</h1>
        <p style='color:#888;font-size:.87rem'>
            Mood-based music · TF-IDF ML · Hybrid scoring
            {'&nbsp;·&nbsp; <span style="color:#1db954">🟢 Spotify connected</span>'
             if sp_ok else ''}</p>
    </div>""", unsafe_allow_html=True)

    liked = sum(1 for a in fb.values() if a=="like")
    top_m = max(set(mh),key=mh.count) if mh else "—"

    m1,m2,m3,m4 = st.columns(4)
    with m1: st.metric("🎂 Age",      st.session_state["age"])
    with m2: st.metric("🌐 Language", st.session_state["language"])
    with m3: st.metric("❤️ Liked",    liked)
    with m4: st.metric("🎭 Top Mood",
                        f"{EMOJI.get(top_m,'')} {top_m.capitalize()}" if top_m!="—" else "—")

    # Spotify top tracks (if connected)
    if sp_ok:
        st.markdown("---")
        st.markdown("### 🎧 Your Spotify Top Tracks")
        tok = st.session_state["spotify_token"]
        with st.spinner("Loading your top tracks…"):
            top_tracks = sp.get_user_top_tracks(tok, limit=5)
        if top_tracks:
            for i,t in enumerate(top_tracks,1):
                st.markdown(
                    f"<div style='background:#091610;border:1px solid #1db95430;"
                    f"border-radius:10px;padding:.5rem .9rem;margin-bottom:.3rem;font-size:.84rem'>"
                    f"<b style='color:#1db954'>{i}.</b> {t['song']}"
                    f" <span style='color:#888'>— {t['artist']}</span></div>",
                    unsafe_allow_html=True)
        else:
            st.info("No top tracks found. Listen more on Spotify!")

    st.markdown("---")
    st.markdown("### 🚀 Quick Start")
    cols = st.columns(5)
    for i,mood in enumerate(ALL_MOODS):
        with cols[i%5]:
            if st.button(f"{EMOJI[mood]} {mood.capitalize()}",
                          key=f"qm_{mood}", use_container_width=True):
                st.session_state["_preset"]=mood
                st.session_state["page"]="recommend"; st.rerun()

    st.markdown("---")
    st.markdown("### 🎭 Mood Profiles")
    rows = [{"Mood":f"{EMOJI[m]} {m.capitalize()}","Energy":f"{p['energy']:.0%}",
             "Valence":f"{p['valence']:.0%}","Tempo":f"{p['tempo']} BPM"}
            for m,p in MOOD_PROFILES.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if mh:
        st.markdown("---")
        st.markdown("### 📈 Mood History")
        mc_s = pd.Series(mh).value_counts().reset_index()
        mc_s.columns=["Mood","Count"]
        st.bar_chart(mc_s.set_index("Mood")["Count"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RECOMMEND
# ══════════════════════════════════════════════════════════════════════════════
def page_recommend():
    df = get_df()
    fb = st.session_state["feedback"]
    liked, disliked = liked_disliked(fb)
    lw  = compute_language_weight(liked, df)
    tok = st.session_state.get("spotify_token")
    sp_valid = sp.is_token_valid(tok)

    st.markdown("""
    <h1 style='font-size:1.85rem;margin-bottom:.2rem'>🎵 Recommend</h1>
    <p style='color:#888;margin-bottom:.7rem'>
        ML mood detection · Local dataset · Live Spotify</p>
    """, unsafe_allow_html=True)

    # ── Dataset toggle ─────────────────────────────────────────────────────
    tog_col,_ = st.columns([2,3])
    with tog_col:
        if sp_valid or sp.is_configured():
            use_spotify = st.toggle(
                "🎧 Spotify Live Mode",
                value=sp_valid,
                help="ON = Spotify API  |  OFF = Local CSV dataset")
        else:
            use_spotify = False
            st.markdown("<span class='badge-purple'>⚙️ Local Dataset</span>",
                        unsafe_allow_html=True)

    if use_spotify and not sp_valid:
        st.warning("Connect your Spotify account in the sidebar first.")
        use_spotify = False

    mode_badge = ("<span class='badge-green'>🟢 Spotify Live</span>"
                  if use_spotify else
                  "<span class='badge-purple'>⚙️ Local Dataset</span>")
    st.markdown(mode_badge, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input mode ─────────────────────────────────────────────────────────
    mode = st.radio("", ["💬 Text","🎛️ Dropdown","🎲 Surprise"],
                     horizontal=True, label_visibility="collapsed")
    preset        = st.session_state.pop("_preset", None)
    selected_mood = preset or "happy"
    multi_mode    = False; multi_moods = []

    if mode == "💬 Text":
        raw = st.text_input("How are you feeling?",
                             placeholder='"Need to focus on my project…"')
        if raw:
            clean_txt     = _sanitize(raw)
            selected_mood, conf = detect_mood(clean_txt)
            mc = COLOR[selected_mood]
            st.markdown(
                f"<div style='background:{mc}22;border:1px solid {mc}55;"
                f"border-radius:10px;padding:.45rem .85rem;font-size:.85rem;margin:.35rem 0'>"
                f"<b>Detected:</b> <b style='color:{mc}'>"
                f"{EMOJI[selected_mood]} {selected_mood.capitalize()}</b>"
                f"<span style='color:#888;font-size:.77rem'>"
                f" &nbsp;·&nbsp; {conf:.0%} confidence</span></div>",
                unsafe_allow_html=True)

    elif mode == "🎛️ Dropdown":
        cm,ct = st.columns([3,1])
        with cm:
            selected_mood = st.selectbox(
                "Mood", ALL_MOODS,
                index=ALL_MOODS.index(selected_mood) if selected_mood in ALL_MOODS else 0,
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
                st.info("Pick at least 2 moods.")
                multi_mode = False
    else:
        if st.button("🎲 Random Mood", use_container_width=True):
            st.session_state["_surprise"] = random.choice(ALL_MOODS)
        selected_mood = st.session_state.get("_surprise","happy")
        mc = COLOR[selected_mood]
        st.markdown(
            f"<div style='background:{mc}22;border:1px solid {mc}55;"
            f"border-radius:10px;padding:.45rem .85rem;font-size:.85rem'>"
            f"🎲 <b style='color:{mc}'>{EMOJI[selected_mood]} {selected_mood.capitalize()}</b></div>",
            unsafe_allow_html=True)

    # ── Filters ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    f1,f2 = st.columns(2)
    with f1:
        lang_filter = st.selectbox(
            "🌐 Language", LANGS,
            index=LANGS.index(st.session_state["language"])
                  if st.session_state["language"] in LANGS else 0)
    with f2:
        n_songs = st.slider("🎵 Number of songs", 3, 10, 5)

    if lw > 0.15:
        st.info(f"🌐 Language weight boosted to {lw:.0%} — you prefer {lang_filter} songs.")

    st.session_state.pop("_preset", None)
    go = st.button("🎶 Get Recommendations", type="primary", use_container_width=True)
    if not go: return

    mc = COLOR.get(selected_mood,"#7c5cfc")

    # ── LOCAL RECOMMENDATIONS ──────────────────────────────────────────────
    if not use_spotify:
        st.markdown("---")
        st.markdown("<span class='badge-purple'>⚙️ Local ML Recommendations</span>",
                    unsafe_allow_html=True)
        with st.spinner("Running hybrid ML engine…"):
            if multi_mode and len(multi_moods) > 1:
                recs  = multi_mood_recommend(df, multi_moods, lang_filter,
                                              st.session_state["age"],
                                              n_songs, liked, disliked)
                label = " + ".join(f"{EMOJI[m]} {m.capitalize()}" for m in multi_moods)
            else:
                recs  = recommend(df, selected_mood, lang_filter,
                                   st.session_state["age"],
                                   n_songs, liked, disliked, lw)
                label = f"{EMOJI[selected_mood]} {selected_mood.capitalize()}"
                st.session_state["mood_history"].append(selected_mood)

        if recs.empty:
            st.warning("No results. Try 'Any' language.")
            return
        st.markdown(f"""
        <div style='padding:.25rem 0 .65rem'>
            <span style='color:{mc};font-weight:700'>{label}</span>
            <span style='color:#888;font-size:.8rem'>
                &nbsp;·&nbsp; {len(recs)} songs &nbsp;·&nbsp; {lang_filter}</span>
        </div>""", unsafe_allow_html=True)
        for i,(_, row) in enumerate(recs.iterrows(), 1):
            offline_card(row, i)

    # ── SPOTIFY LIVE RECOMMENDATIONS ───────────────────────────────────────
    else:
        st.markdown("---")
        st.markdown("<span class='badge-green'>🟢 Spotify Live Recommendations</span>",
                    unsafe_allow_html=True)
        sp_results = []
        with st.spinner("Fetching from Spotify…"):
            try:
                sp_results = sp.get_recommendations(
                    selected_mood, lang_filter, n_songs, tok)
                if not sp_results:
                    sp_results = sp.search_tracks(
                        selected_mood, lang_filter, n_songs, tok)
            except Exception as e:
                st.error(f"Spotify error: {e}")

        st.session_state["mood_history"].append(selected_mood)

        if sp_results:
            st.caption(f"{len(sp_results)} live tracks from Spotify")
            # Offer to create playlist
            if st.button("➕ Save as Spotify Playlist", key="save_pl"):
                tok2 = st.session_state.get("spotify_token")
                prof = sp.get_user_profile(tok2) if tok2 else None
                if prof:
                    uris    = [t.get("track_uri","") for t in sp_results
                               if t.get("track_uri","").startswith("spotify:track:")]
                    pl_name = f"MoodTunes — {selected_mood.capitalize()} {datetime.now().strftime('%d %b')}"
                    url     = sp.create_playlist(tok2, prof["id"], pl_name, uris)
                    if url:
                        st.success(f"✅ Playlist created! [Open on Spotify]({url})")
                    else:
                        st.error("Playlist creation failed.")
                else:
                    st.warning("Connect Spotify account to save playlists.")

            for i,item in enumerate(sp_results, 1):
                spotify_card(item, i)
        else:
            st.warning("No Spotify results — showing local recommendations.")
            recs = recommend(df, selected_mood, lang_filter,
                              st.session_state["age"], n_songs, liked, disliked, lw)
            for i,(_, row) in enumerate(recs.iterrows(), 1):
                offline_card(row, i)

    if liked:
        st.markdown("---")
        st.markdown(f"❤️ **{len(liked)} liked songs** are influencing your recommendations.")


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
        else: st.metric("🌐 Top Language","—")

    if mh:
        st.markdown("---")
        st.markdown("### 📈 Mood Sessions")
        ms = pd.Series(mh).value_counts().reset_index()
        ms.columns=["Mood","Count"]
        st.bar_chart(ms.set_index("Mood")["Count"])

    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("### 🌐 Mood × Language")
        st.dataframe(mood_lang_pivot(df).style.background_gradient(cmap="Purples"),
                      use_container_width=True)
    with c2:
        st.markdown("### 👤 Mood × Age Group")
        st.dataframe(mood_age_pivot(df).style.background_gradient(cmap="Blues"),
                      use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔥 Most Popular Songs")
    top15 = df.nlargest(15,"popularity")[["song","artist","mood","popularity"]]
    st.bar_chart(top15.set_index("song")["popularity"])

    o1,o2,o3,o4 = st.columns(4)
    with o1: st.metric("Total Songs",   len(df))
    with o2: st.metric("Moods",         df["mood"].nunique())
    with o3: st.metric("Languages",     df["language"].nunique())
    with o4: st.metric("Avg Popularity",f"{df['popularity'].mean():.0f}")

    ca,cb = st.columns(2)
    with ca:
        st.markdown("#### Songs per Mood")
        mc_df = df["mood"].value_counts().reset_index()
        mc_df.columns=["Mood","Songs"]
        st.bar_chart(mc_df.set_index("Mood")["Songs"])
    with cb:
        st.markdown("#### Songs per Language")
        lc_df = df["language"].value_counts().reset_index()
        lc_df.columns=["Language","Songs"]
        st.bar_chart(lc_df.set_index("Language")["Songs"])

    if liked:
        st.markdown("---")
        st.markdown("### ❤️ Your Liked Songs")
        ld = df[df["song"].isin(liked)][["song","artist","mood","language","genre"]].copy()
        ld.columns=["Song","Artist","Mood","Language","Genre"]
        st.dataframe(ld, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state["logged_in"]:
    page_auth()
else:
    render_sidebar()
    pg = st.session_state.get("page","home")
    if   pg == "home":      page_home()
    elif pg == "recommend": page_recommend()
    elif pg == "analytics": page_analytics()
