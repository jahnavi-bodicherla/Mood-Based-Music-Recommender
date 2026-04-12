"""
╔══════════════════════════════════════════════════════════════╗
║  Multilingual Mood-Based Music Recommender                  ║
║  ──────────────────────────────────────────────────────     ║
║  ML        : TF-IDF + Cosine Similarity                     ║
║  Security  : SHA-256 hashing + Fernet AES encryption        ║
║  Dataset   : 325 songs · 5 languages · 5 moods              ║
║  UI        : Streamlit                                       ║
╚══════════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import csv, hashlib, os, random, re
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
from cryptography.fernet import Fernet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ══════════════════════════════════════════════════════════════════════════════
# FILE PATHS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
KEY_FILE      = Path("secret.key")
USERS_FILE    = Path("users.csv")
SONGS_FILE    = Path("songs.csv")
HISTORY_FILE  = Path("history.csv")
FAVS_FILE     = Path("favorites.csv")

MOODS     = ["Happy", "Sad", "Energetic", "Calm", "Romantic"]
LANGUAGES = ["Any", "Telugu", "Hindi", "Tamil", "English", "Korean"]
GENRES    = ["Any", "Pop", "Rock", "Melody", "Classical", "Indie",
             "Folk", "K-pop", "EDM", "Ambient", "Hip-Hop", "R&B"]

MOOD_EMOJI  = {"Happy":"😊","Sad":"😔","Energetic":"⚡","Calm":"😌","Romantic":"💕"}
MOOD_COLOR  = {"Happy":"#F9C846","Sad":"#5BA4CF","Energetic":"#FF5F57",
               "Calm":"#42C89B","Romantic":"#EC4899"}
LANG_FLAG   = {"Telugu":"🇮🇳","Hindi":"🇮🇳","Tamil":"🇮🇳",
               "English":"🇬🇧","Korean":"🇰🇷","Any":"🌐"}

# ══════════════════════════════════════════════════════════════════════════════
# 1 ── ENCRYPTION  (Fernet / AES-128)
# ══════════════════════════════════════════════════════════════════════════════
def load_key() -> Fernet:
    """Load existing secret.key or generate + save a new one."""
    if KEY_FILE.exists():
        return Fernet(KEY_FILE.read_bytes())
    key = Fernet.generate_key()
    KEY_FILE.write_bytes(key)
    return Fernet(key)

_CIPHER: Fernet = load_key()

def encrypt_data(text: str) -> str:
    """Encrypt plain text → Fernet base64 token."""
    return _CIPHER.encrypt(text.encode()).decode()

def decrypt_data(token: str) -> str:
    """Decrypt Fernet token → plain text. Returns '' on failure."""
    try:
        return _CIPHER.decrypt(token.encode()).decode()
    except Exception:
        return ""

# ══════════════════════════════════════════════════════════════════════════════
# 2 ── AUTHENTICATION  (SHA-256 hashed passwords)
# ══════════════════════════════════════════════════════════════════════════════
def hash_password(pw: str) -> str:
    """SHA-256 hex digest. NEVER store plain text."""
    return hashlib.sha256(pw.encode()).hexdigest()

def _load_users() -> dict:
    """Return {encrypted_username: password_hash} from users.csv."""
    if not USERS_FILE.exists() or USERS_FILE.stat().st_size == 0:
        return {}
    out = {}
    with open(USERS_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["username_enc"]] = row["password_hash"]
    return out

def signup(username: str, password: str) -> tuple[bool, str]:
    """
    Create a new account.
    Encrypts username with Fernet, hashes password with SHA-256.
    """
    username = username.strip()
    if not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
        return False, "Username must be 3–20 chars (letters, numbers, _)."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    users = _load_users()
    for enc in users:
        if decrypt_data(enc) == username:
            return False, "Username already taken. Choose another."
    exists = USERS_FILE.exists() and USERS_FILE.stat().st_size > 0
    with open(USERS_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["username_enc", "password_hash"])
        if not exists:
            w.writeheader()
        w.writerow({
            "username_enc": encrypt_data(username),    # ← Fernet encrypted
            "password_hash": hash_password(password),  # ← SHA-256 hashed
        })
    return True, "Account created!"

def login(username: str, password: str) -> bool:
    """Verify credentials. Decrypt stored usernames, compare hashed passwords."""
    users = _load_users()
    for enc, pwd_hash in users.items():
        if decrypt_data(enc) == username:
            return hash_password(password) == pwd_hash
    return False

# ══════════════════════════════════════════════════════════════════════════════
# 3 ── DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_songs() -> pd.DataFrame:
    """Load songs.csv and build TF-IDF feature string."""
    df = pd.read_csv(SONGS_FILE)
    df["mood"]     = df["mood"].fillna("Happy").str.strip()
    df["language"] = df["language"].fillna("English").str.strip()
    df["genre"]    = df["genre"].fillna("Pop").str.strip()
    # Feature = "mood language genre"  (used for cosine similarity)
    df["_feat"] = (df["mood"].str.lower() + " " +
                   df["language"].str.lower() + " " +
                   df["genre"].str.lower())
    return df

@st.cache_resource(show_spinner=False)
def build_tfidf(df: pd.DataFrame):
    """Fit TF-IDF vectoriser on the corpus (cached as resource)."""
    vec    = TfidfVectorizer(ngram_range=(1, 2))
    matrix = vec.fit_transform(df["_feat"])
    return vec, matrix

# ══════════════════════════════════════════════════════════════════════════════
# 4 ── RECOMMENDATION ENGINE  (TF-IDF + Cosine Similarity)
# ══════════════════════════════════════════════════════════════════════════════
def recommend(
    mood: str,
    language: str = "Any",
    genre:    str = "Any",
    n:        int = 8,
    exclude:  list = None,
) -> pd.DataFrame:
    """
    TF-IDF + cosine similarity recommender.

    Steps:
      1. Build query: 'mood [language] [genre]'
      2. Score all songs by cosine similarity
      3. Hard-filter pool to the selected mood
      4. Apply optional language / genre filters (relax if < 3 results)
      5. Return top-n by score
    """
    df       = load_songs()
    vec, mat = build_tfidf(df)
    exclude  = exclude or []

    pool = df[~df["song_name"].isin(exclude)].copy()

    # Build query string
    parts = [mood.lower()]
    if language != "Any": parts.append(language.lower())
    if genre    != "Any": parts.append(genre.lower())
    q_vec = vec.transform([" ".join(parts)])
    pool["_score"] = cosine_similarity(q_vec, mat[pool.index]).flatten()

    # Hard-filter: must match requested mood
    mood_pool = pool[pool["mood"].str.lower() == mood.lower()].copy()

    # Soft-filter: language
    if language != "Any":
        sub = mood_pool[mood_pool["language"] == language]
        mood_pool = sub if len(sub) >= 3 else mood_pool

    # Soft-filter: genre
    if genre != "Any":
        sub = mood_pool[mood_pool["genre"] == genre]
        mood_pool = sub if len(sub) >= 3 else mood_pool

    # Fallback to full pool if very few results
    if len(mood_pool) < n:
        mood_pool = pool.copy()

    cols = ["song_name", "artist", "language", "mood", "genre", "youtube_url"]
    return mood_pool.nlargest(n, "_score")[cols].reset_index(drop=True)


def mix_moods(mood1: str, mood2: str,
              language: str = "Any", genre: str = "Any",
              n: int = 8) -> pd.DataFrame:
    """
    Blend two moods: interleave results 50/50, deduplicate.
    """
    half  = max(n // 2, 2)
    r1    = recommend(mood1, language, genre, half + 2)
    r2    = recommend(mood2, language, genre, half + 2,
                      exclude=r1["song_name"].tolist())
    return pd.concat([r1.head(half), r2.head(half)],
                     ignore_index=True).drop_duplicates("song_name").head(n)


def trending_in_language(language: str, n: int = 5) -> pd.DataFrame:
    """Return top n popular songs for a given language (random from top pool)."""
    df   = load_songs()
    pool = df[df["language"] == language] if language != "Any" else df
    return pool.sample(min(n, len(pool)))[
        ["song_name","artist","language","mood","genre","youtube_url"]
    ].reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# 5 ── HISTORY & FAVORITES  (encrypted username)
# ══════════════════════════════════════════════════════════════════════════════
def save_history(username: str, mood: str, language: str, song: str):
    """Append session to history. Username is Fernet encrypted."""
    exists = HISTORY_FILE.exists() and HISTORY_FILE.stat().st_size > 0
    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["user_enc","mood","language","song","ts"])
        if not exists:
            w.writeheader()
        w.writerow({
            "user_enc": encrypt_data(username),  # ← encrypted
            "mood": mood, "language": language, "song": song,
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

def load_history(username: str) -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    rows = []
    with open(HISTORY_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if decrypt_data(row.get("user_enc","")) == username:
                rows.append({k: row[k] for k in ["mood","language","song","ts"]})
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def toggle_favorite(username: str, song_name: str, artist: str) -> bool:
    """Add or remove a favorite. Returns True if added, False if removed."""
    rows = []
    if FAVS_FILE.exists() and FAVS_FILE.stat().st_size > 0:
        with open(FAVS_FILE, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if decrypt_data(r.get("user_enc","")) == username and r["song"] == song_name:
                rows = [x for x in rows
                        if not (decrypt_data(x.get("user_enc","")) == username
                                and x["song"] == song_name)]
                with open(FAVS_FILE, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=["user_enc","song","artist"])
                    w.writeheader(); w.writerows(rows)
                return False  # removed
    rows.append({"user_enc": encrypt_data(username), "song": song_name, "artist": artist})
    with open(FAVS_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["user_enc","song","artist"])
        w.writeheader(); w.writerows(rows)
    return True  # added

def load_favorites(username: str) -> list:
    if not FAVS_FILE.exists(): return []
    favs = []
    with open(FAVS_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if decrypt_data(row.get("user_enc","")) == username:
                favs.append(row["song"])
    return favs

# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MoodTunes 🎵",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:.7rem}

.card{background:#1a1d28;border:1px solid #2e3250;border-radius:14px;
      padding:.85rem 1.05rem;margin-bottom:.42rem;
      transition:border-color .2s,transform .14s}
.card:hover{border-color:#7c5cfc;transform:translateY(-1px)}
.c-title{font-weight:700;font-size:.96rem;margin-bottom:.06rem}
.c-sub{color:#888;font-size:.79rem;margin-bottom:.28rem}
.pill{display:inline-block;background:rgba(124,92,252,.14);color:#a78bfa;
      border-radius:20px;padding:.04rem .42rem;font-size:.67rem;font-weight:600;margin-right:.18rem}
.mpill{display:inline-block;border-radius:20px;
       padding:.04rem .42rem;font-size:.67rem;font-weight:600;margin-right:.18rem}
.lpill{display:inline-block;background:rgba(255,255,255,.06);color:#ccc;
       border-radius:20px;padding:.04rem .42rem;font-size:.67rem;margin-right:.18rem}

[data-testid="stMetric"]{background:#1a1d28;border:1px solid #2e3250;
    border-radius:12px;padding:.7rem!important}
[data-testid="stMetricValue"]{color:#7c5cfc!important;font-weight:700!important}
.stButton>button{border-radius:10px!important;font-weight:600!important}
.stTextInput>div>div>input{background:#10131e!important;border:1px solid #2e3250!important;
    border-radius:10px!important;color:#e8eaf0!important}
.stTextInput>div>div>input:focus{border-color:#7c5cfc!important;box-shadow:none!important}
.stTabs [data-baseweb="tab-list"]{background:#12151e;border-radius:10px;padding:.18rem}
.stTabs [data-baseweb="tab"]{border-radius:8px;font-weight:600}
</style>
""", unsafe_allow_html=True)

# ── Session defaults ──────────────────────────────────────────────────────────
for k, v in dict(logged_in=False, username="", page="login",
                 last_recs=None).items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# AUTH PAGES  (Login + Signup as tabs)
# ══════════════════════════════════════════════════════════════════════════════
def show_auth():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("""
        <div style='text-align:center;padding:1.5rem 0 .6rem'>
            <div style='font-size:3rem'>🎵</div>
            <h1 style='font-size:2rem;margin:.15rem 0;
                background:linear-gradient(135deg,#7c5cfc,#e85d9b);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
                MoodTunes</h1>
            <p style='color:#888;font-size:.87rem'>
                Multilingual Mood-Based Music Recommender</p>
        </div>""", unsafe_allow_html=True)

        tab_l, tab_s = st.tabs(["🔐  Login", "✏️  Sign Up"])

        # ── Login ──────────────────────────────────────────────────────────
        with tab_l:
            with st.form("lf"):
                u = st.text_input("Username", placeholder="your_username")
                p = st.text_input("Password", type="password",
                                   placeholder="Enter password")
                if st.form_submit_button("Sign In →", use_container_width=True,
                                          type="primary"):
                    if not u.strip() or not p:
                        st.error("Both fields are required.")
                    elif login(u.strip(), p):
                        st.session_state.update(
                            logged_in=True, username=u.strip(), page="app")
                        st.rerun()
                    else:
                        st.error("❌ Incorrect username or password.")

        # ── Sign Up ────────────────────────────────────────────────────────
        with tab_s:
            with st.form("sf"):
                nu  = st.text_input("Choose Username",
                                     placeholder="3–20 chars, a-z 0-9 _")
                np1 = st.text_input("Choose Password",
                                     type="password", placeholder="Min 6 chars")
                np2 = st.text_input("Confirm Password",
                                     type="password", placeholder="Repeat password")
                if st.form_submit_button("Create Account →",
                                          use_container_width=True, type="primary"):
                    if np1 != np2:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = signup(nu.strip(), np1)
                        if ok:
                            st.success(f"✅ {msg} — Sign in above.")
                        else:
                            st.error(f"❌ {msg}")
            st.markdown("""
            <div style='background:#0e1621;border:1px solid #1b3a5c;border-radius:10px;
                 padding:.6rem .9rem;font-size:.78rem;color:#90a4ae;margin-top:.4rem'>
                🔒 Password → <b style='color:#64b5f6'>SHA-256 hashed</b>
                &nbsp;·&nbsp;
                Username → <b style='color:#64b5f6'>Fernet AES encrypted</b><br>
                Plain-text credentials are <b>never stored on disk</b>.
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SONG CARD RENDERER
# ══════════════════════════════════════════════════════════════════════════════
def render_card(row: dict, idx: int, username: str, favs: list):
    """Render a single song card with mood colour, tags, fav button, YouTube."""
    name      = row.get("song_name", "")
    artist    = row.get("artist", "")
    lang      = row.get("language", "")
    mood      = row.get("mood", "")
    genre     = row.get("genre", "")
    yt_url    = row.get("youtube_url", "")
    mc        = MOOD_COLOR.get(mood, "#7c5cfc")
    is_fav    = name in favs
    border    = "#FBBF24" if is_fav else "#2e3250"
    fav_star  = "\u2b50 " if is_fav else ""
    mood_icon = MOOD_EMOJI.get(mood, "")
    lang_icon = LANG_FLAG.get(lang, "\U0001f310")

    # Build HTML as plain string concatenation — avoids raw HTML leaking
    html = (
        "<div class='card' style='border-color:" + border + "'>"
        "<div class='c-title'>"
        + fav_star
        + "<span style='color:" + mc + ";margin-right:.3rem'>\U0001f3b5</span>"
        + name + "</div>"
        "<div class='c-sub'>\U0001f3a4 " + artist + "</div>"
        "<span class='mpill' style='background:" + mc + "22;color:" + mc + "'>"
        + mood_icon + " " + mood + "</span>"
        "<span class='lpill'>" + lang_icon + " " + lang + "</span>"
        "<span class='pill'>" + genre + "</span>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)

    b1, b2 = st.columns([1, 1])
    sfx = f"{name[:12]}_{idx}"
    with b1:
        lbl = "⭐ Saved" if is_fav else "☆ Favorite"
        if st.button(lbl, key=f"fav_{sfx}", use_container_width=True):
            toggle_favorite(username, name, artist)
            st.rerun()
    with b2:
        if yt_url:
            st.link_button("▶ YouTube", yt_url, use_container_width=True)

    # Embedded video player inside an expander
    if yt_url:
        with st.expander(f"▶ Play: {name}"):
            st.video(yt_url)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def show_app():
    user  = st.session_state["username"]
    df    = load_songs()
    hist  = load_history(user)
    favs  = load_favorites(user)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style='background:#1a1d28;border:1px solid #2e3250;border-radius:12px;
             padding:.75rem 1rem;text-align:center;margin-bottom:.9rem'>
            <div style='font-size:1.4rem'>🎵</div>
            <div style='font-weight:800;font-size:.96rem;
                 background:linear-gradient(135deg,#7c5cfc,#e85d9b);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
                 MoodTunes</div>
            <div style='color:#888;font-size:.73rem;margin-top:.2rem'>👤 {user}</div>
        </div>""", unsafe_allow_html=True)

        # Stats
        total_s  = len(hist)
        fav_mood = hist["mood"].value_counts().idxmax() if not hist.empty else "—"
        st.metric("🔍 Searches", total_s)
        st.metric("🏆 Fav Mood",
                   f"{MOOD_EMOJI.get(fav_mood,'')} {fav_mood}" if fav_mood!="—" else "—")
        st.metric("⭐ Favorites", len(favs))
        st.markdown("---")

        # Filters
        st.markdown("### 🎭 Mood")
        mood = st.selectbox("M", MOODS, label_visibility="collapsed",
                             format_func=lambda m: f"{MOOD_EMOJI[m]}  {m}")

        st.markdown("### 🌐 Language")
        lang = st.selectbox("L", LANGUAGES, label_visibility="collapsed",
                             format_func=lambda l: f"{LANG_FLAG.get(l,'🌐')}  {l}")

        st.markdown("### 🎸 Genre (optional)")
        genre = st.selectbox("G", GENRES, label_visibility="collapsed")

        st.markdown("### 🎵 How many songs?")
        n_songs = st.slider("N", 5, 15, 8, label_visibility="collapsed")

        # Mix mood
        st.markdown("### 🎛️ Mix My Mood")
        mix_on = st.checkbox("Blend two moods", value=False)
        mood2  = "Sad"
        if mix_on:
            mood2 = st.selectbox("Second mood", MOODS, index=1,
                                  label_visibility="collapsed",
                                  format_func=lambda m: f"{MOOD_EMOJI[m]}  {m}",
                                  key="m2")

        st.markdown("---")
        b_rec  = st.button("🎶  Recommend",    use_container_width=True, type="primary")
        b_surp = st.button("🎲  Surprise Me",  use_container_width=True)
        st.markdown("---")
        if st.button("🚪  Logout", use_container_width=True):
            st.session_state.update(logged_in=False, username="",
                                     page="login", last_recs=None)
            st.rerun()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <h1 style='font-size:1.85rem;margin-bottom:.1rem'>
        🎵 Welcome, <span style='color:#a78bfa'>{user}</span>!
    </h1>
    <p style='color:#888;font-size:.84rem;margin-bottom:.7rem'>
        325 songs · 5 languages · TF-IDF ML ·
        SHA-256 Auth · Fernet Encryption</p>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_r, tab_t, tab_s, tab_f, tab_a = st.tabs(
        ["🎶 Recommend","🔥 Trending","🔍 Search","⭐ Favorites","📊 Analytics"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — RECOMMEND
    # ════════════════════════════════════════════════════════════════════════
    with tab_r:
        results = None
        label   = ""

        if b_rec:
            with st.spinner("Finding your songs…"):
                if mix_on:
                    results = mix_moods(mood, mood2, lang, genre, n_songs)
                    label   = (f"{MOOD_EMOJI[mood]} {mood} + "
                               f"{MOOD_EMOJI[mood2]} {mood2}")
                else:
                    results = recommend(mood, lang, genre, n_songs)
                    label   = f"{MOOD_EMOJI[mood]} {mood}"
            if results is not None and not results.empty:
                save_history(user, mood, lang,
                             results.iloc[0]["song_name"])
            st.session_state["last_recs"] = (results.to_dict("records")
                                              if results is not None else [])

        elif b_surp:
            rand_mood = random.choice(MOODS)
            results   = recommend(rand_mood, "Any", "Any", n_songs)
            label     = f"🎲 Surprise: {MOOD_EMOJI[rand_mood]} {rand_mood}"
            st.session_state["last_recs"] = results.to_dict("records")

        # Show current or last results
        display = []
        if results is not None and not results.empty:
            display = results.to_dict("records")
        elif st.session_state.get("last_recs"):
            display = st.session_state["last_recs"]
            label   = "🕓 Last Session"

        if display:
            first_mood = display[0].get("mood", mood)
            mc = MOOD_COLOR.get(first_mood,"#7c5cfc")
            st.markdown(f"""
            <div style='text-align:center;padding:.3rem 0 .65rem'>
                <span style='font-weight:700;color:{mc};font-size:1.05rem'>{label}</span>
                <span style='color:#888;font-size:.8rem'>
                    &nbsp;·&nbsp; {len(display)} songs</span>
            </div>""", unsafe_allow_html=True)
            for i, row in enumerate(display):
                render_card(row, i, user, favs)
        else:
            st.markdown("""
            <div style='text-align:center;padding:3rem 0;color:#555'>
                <div style='font-size:2.8rem'>🎧</div>
                <p style='margin-top:.5rem'>Select filters and click
                    <b style='color:#ccc'>Recommend</b></p>
            </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — TRENDING IN YOUR LANGUAGE
    # ════════════════════════════════════════════════════════════════════════
    with tab_t:
        st.markdown("### 🔥 Trending in Your Language")
        t_lang = st.selectbox(
            "Pick language", LANGUAGES[1:], key="tlang",
            format_func=lambda l: f"{LANG_FLAG.get(l,'🌐')} {l}")
        trending = trending_in_language(t_lang, n=8)
        if trending.empty:
            st.info("No songs found for this language.")
        else:
            for i, (_, row) in enumerate(trending.iterrows()):
                render_card(row.to_dict(), i + 100, user, favs)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — SEARCH
    # ════════════════════════════════════════════════════════════════════════
    with tab_s:
        st.markdown("### 🔍 Search Songs")
        q = st.text_input("Search by song name or artist",
                           placeholder="e.g. Naatu Naatu or Arijit Singh",
                           key="search_q")
        if q and len(q.strip()) >= 2:
            mask = (df["song_name"].str.lower().str.contains(q.lower(), na=False) |
                    df["artist"].str.lower().str.contains(q.lower(), na=False))
            res  = df[mask][["song_name","artist","language","mood","genre","youtube_url"]]
            st.caption(f"{len(res)} result(s) for '{q}'")
            for i, (_, row) in enumerate(res.head(15).iterrows()):
                render_card(row.to_dict(), i + 200, user, favs)
        elif q:
            st.info("Type at least 2 characters to search.")
        else:
            st.info("Enter a song name or artist to search the full catalog.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — FAVORITES
    # ════════════════════════════════════════════════════════════════════════
    with tab_f:
        st.markdown("### ⭐ Your Favorite Songs")
        if not favs:
            st.info("No favorites yet. Click ☆ on any song to save it.")
        else:
            fav_df = df[df["song_name"].isin(favs)][
                ["song_name","artist","language","mood","genre","youtube_url"]]
            for i, (_, row) in enumerate(fav_df.iterrows()):
                render_card(row.to_dict(), i + 300, user, favs)
            st.markdown("---")
            st.download_button(
                "⬇️ Download Playlist (CSV)",
                data=fav_df.to_csv(index=False),
                file_name=f"{user}_playlist.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — ANALYTICS
    # ════════════════════════════════════════════════════════════════════════
    with tab_a:
        st.markdown("### 📊 Your Analytics")
        if hist.empty:
            st.info("No history yet — start recommending to see analytics!")
        else:
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Searches",   len(hist))
            with c2:
                top_m = hist["mood"].value_counts().idxmax()
                st.metric("Top Mood", f"{MOOD_EMOJI.get(top_m,'')} {top_m}")
            with c3:
                top_l = hist["language"].value_counts().idxmax() \
                        if "language" in hist.columns else "—"
                st.metric("Top Language", f"{LANG_FLAG.get(top_l,'🌐')} {top_l}")

            st.markdown("#### 🎭 Mood Usage")
            mc_c = hist["mood"].value_counts().reset_index()
            mc_c.columns = ["Mood","Count"]
            st.bar_chart(mc_c.set_index("Mood")["Count"])

            if "language" in hist.columns:
                st.markdown("#### 🌐 Language Preference")
                lc_c = hist["language"].value_counts().reset_index()
                lc_c.columns = ["Language","Count"]
                st.bar_chart(lc_c.set_index("Language")["Count"])

            with st.expander("📋 Session History"):
                st.dataframe(hist.sort_values("ts", ascending=False),
                              use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 🗂️ Dataset Overview")
        d1, d2, d3, d4 = st.columns(4)
        with d1: st.metric("Songs",     len(df))
        with d2: st.metric("Languages", df["language"].nunique())
        with d3: st.metric("Moods",     df["mood"].nunique())
        with d4: st.metric("Genres",    df["genre"].nunique())

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("#### Songs by Language")
            lc = df["language"].value_counts().reset_index()
            lc.columns = ["Language","Count"]
            st.bar_chart(lc.set_index("Language")["Count"])
        with cc2:
            st.markdown("#### Songs by Mood")
            mc = df["mood"].value_counts().reset_index()
            mc.columns = ["Mood","Count"]
            st.bar_chart(mc.set_index("Mood")["Count"])

        # Security info
        st.markdown("---")
        with st.expander("🔒 Security Details"):
            st.markdown("""
            | Layer | Method | Stored as |
            |---|---|---|
            | Password | SHA-256 (hashlib) | 64-char hex digest |
            | Username | Fernet AES-128 | Base64 encrypted token |
            | Mood history | Fernet AES-128 | Username encrypted |
            | Favorites | Fernet AES-128 | Username encrypted |
            | ML model | TF-IDF + Cosine | In-memory (no disk) |
            """)
            sa, sb = st.columns(2)
            with sa:
                st.markdown("**SHA-256 Live:**")
                st.code(
                    f'Input : "password"\n'
                    f'Hash  : {hash_password("password")[:32]}…\n'
                    f'Bits  : 256  (64 hex chars)',
                    language="text")
            with sb:
                st.markdown("**Fernet Live:**")
                enc = encrypt_data(user)
                st.code(
                    f'Input    : "{user}"\n'
                    f'Encrypted: {enc[:35]}…\n'
                    f'Decrypted: {decrypt_data(enc)}',
                    language="text")

# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state["logged_in"]:
    show_auth()
else:
    show_app()
