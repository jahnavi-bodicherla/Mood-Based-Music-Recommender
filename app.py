"""
╔══════════════════════════════════════════════════════════╗
║   Mood-Based Music Recommender                          ║
║   ─────────────────────────────────────────────────     ║
║   ML   : TF-IDF + Cosine Similarity (scikit-learn)     ║
║   Auth : SHA-256 password hashing (hashlib)            ║
║   Enc  : Fernet symmetric encryption (cryptography)    ║
║   UI   : Streamlit                                      ║
╚══════════════════════════════════════════════════════════╝
Run:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Mood Music Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Standard library ──────────────────────────────────────────────────────────
import csv, hashlib, os, random, re
from datetime import datetime
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd
import streamlit as st
from cryptography.fernet import Fernet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & FILE PATHS
# ══════════════════════════════════════════════════════════════════════════════
KEY_FILE     = Path("secret.key")
USERS_FILE   = Path("users.csv")
SONGS_FILE   = Path("songs.csv")
HISTORY_FILE = Path("history.csv")

MOODS  = ["Happy", "Sad", "Energetic", "Calm"]
GENRES = ["Any", "Pop", "Rock", "Indie", "Classical",
          "Ambient", "EDM", "Hip-Hop", "Folk", "Soul",
          "Reggae", "Country"]

MOOD_EMOJI  = {"Happy":"😊", "Sad":"😔", "Energetic":"⚡", "Calm":"😌"}
MOOD_COLOUR = {"Happy":"#F9C846", "Sad":"#5BA4CF",
               "Energetic":"#FF5F57", "Calm":"#42C89B"}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ENCRYPTION  (Fernet / AES-128 under the hood)
# ══════════════════════════════════════════════════════════════════════════════
def _load_or_create_key() -> Fernet:
    """Load key from secret.key; generate + save a new one if absent."""
    if KEY_FILE.exists():
        return Fernet(KEY_FILE.read_bytes())
    key = Fernet.generate_key()
    KEY_FILE.write_bytes(key)
    return Fernet(key)

# One cipher object reused for the whole session
_CIPHER: Fernet = _load_or_create_key()


def encrypt_data(text: str) -> str:
    """Encrypt plain text → URL-safe base64 token (str)."""
    return _CIPHER.encrypt(text.encode()).decode()


def decrypt_data(token: str) -> str:
    """Decrypt Fernet token → original plain text. Returns '' on failure."""
    try:
        return _CIPHER.decrypt(token.encode()).decode()
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — AUTHENTICATION  (SHA-256 hashed passwords)
# ══════════════════════════════════════════════════════════════════════════════
def hash_password(password: str) -> str:
    """SHA-256 hex digest — NEVER store plain text."""
    return hashlib.sha256(password.encode()).hexdigest()


def _load_users() -> dict:
    """Returns {encrypted_username: password_hash} from users.csv."""
    if not USERS_FILE.exists() or USERS_FILE.stat().st_size == 0:
        return {}
    out = {}
    with open(USERS_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["username_enc"]] = row["password_hash"]
    return out


def signup(username: str, password: str) -> tuple[bool, str]:
    """
    Register a new user.
    - username is encrypted with Fernet before writing
    - password is hashed with SHA-256 before writing
    Returns (success, message).
    """
    if not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
        return False, "Username: 3–20 chars, letters/numbers/underscore only."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    users = _load_users()
    # Check for duplicate (decrypt all stored names)
    for enc in users:
        if decrypt_data(enc) == username:
            return False, "Username already taken."

    exists = USERS_FILE.exists() and USERS_FILE.stat().st_size > 0
    with open(USERS_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["username_enc", "password_hash"])
        if not exists:
            w.writeheader()
        w.writerow({
            "username_enc":  encrypt_data(username),  # ← Fernet encrypted
            "password_hash": hash_password(password),  # ← SHA-256 hashed
        })
    return True, "Account created successfully!"


def login(username: str, password: str) -> bool:
    """Verify credentials — decrypt stored names, compare hashed passwords."""
    users = _load_users()
    for enc, pwd_hash in users.items():
        if decrypt_data(enc) == username:
            return hash_password(password) == pwd_hash
    return False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MACHINE LEARNING  (TF-IDF + Cosine Similarity)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _load_songs() -> pd.DataFrame:
    """Load songs.csv and add combined text feature for TF-IDF."""
    df = pd.read_csv(SONGS_FILE)
    # Feature = "mood genre" — used to compute cosine similarity
    df["_feat"] = df["mood"].str.lower() + " " + df["genre"].str.lower()
    return df


@st.cache_resource(show_spinner=False)
def _build_tfidf(df: pd.DataFrame):
    """Fit TF-IDF vectoriser on the feature corpus (cached as resource)."""
    vec    = TfidfVectorizer()
    matrix = vec.fit_transform(df["_feat"])
    return vec, matrix


def recommend(mood: str, genre: str = "Any", n: int = 5) -> pd.DataFrame:
    """
    TF-IDF + cosine similarity recommender.
    Query = 'mood [genre]' → rank all songs by similarity.
    Hard-filter by mood first, then by genre if requested.
    """
    df        = _load_songs()
    vec, mat  = _build_tfidf(df)

    query     = mood.lower() if genre == "Any" else f"{mood.lower()} {genre.lower()}"
    qvec      = vec.transform([query])
    scores    = cosine_similarity(qvec, mat).flatten()

    tmp       = df.copy()
    tmp["_score"] = scores

    pool = tmp[tmp["mood"].str.lower() == mood.lower()].copy()
    if genre != "Any":
        sub = pool[pool["genre"] == genre]
        pool = sub if len(sub) >= n else pool

    cols = ["song_name", "artist", "mood", "genre"]
    return pool.nlargest(n, "_score")[cols].reset_index(drop=True)


def random_picks(n: int = 5) -> pd.DataFrame:
    """Return n random songs from the full catalogue."""
    df = _load_songs()
    return df.sample(min(n, len(df)))[["song_name","artist","mood","genre"]]\
             .reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MOOD HISTORY  (per-user, username stored encrypted)
# ══════════════════════════════════════════════════════════════════════════════
def save_history(username: str, mood: str, top_song: str):
    """Append a history entry; username is encrypted for privacy."""
    exists = HISTORY_FILE.exists() and HISTORY_FILE.stat().st_size > 0
    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["user_enc","mood","top_song","timestamp"])
        if not exists:
            w.writeheader()
        w.writerow({
            "user_enc":  encrypt_data(username),
            "mood":      mood,
            "top_song":  top_song,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })


def load_history(username: str) -> pd.DataFrame:
    """Return history rows for this user (matched by decrypting stored names)."""
    if not HISTORY_FILE.exists():
        return pd.DataFrame()
    rows = []
    with open(HISTORY_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if decrypt_data(row.get("user_enc","")) == username:
                rows.append({"Mood": row["mood"],
                             "Top Song": row["top_song"],
                             "Time": row["timestamp"]})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MoodTunes 🎵",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:.8rem;max-width:760px}

/* song card */
.card{background:#1a1d28;border:1px solid #2e3250;border-radius:12px;
      padding:.75rem 1rem;margin-bottom:.4rem;transition:border-color .18s}
.card:hover{border-color:#7c5cfc}
.card-title{font-weight:700;font-size:.96rem}
.card-sub{color:#888;font-size:.8rem;margin-top:.1rem}
.pill{display:inline-block;background:rgba(124,92,252,.15);color:#a78bfa;
      border-radius:20px;padding:.04rem .42rem;font-size:.67rem;font-weight:600;
      margin-right:.2rem}

/* metrics */
[data-testid="stMetric"]{background:#1a1d28;border:1px solid #2e3250;
    border-radius:12px;padding:.7rem!important}
[data-testid="stMetricValue"]{color:#7c5cfc!important;font-weight:700!important}

/* inputs & buttons */
.stButton>button{border-radius:10px!important;font-weight:600!important}
.stTextInput>div>div>input{background:#10131e!important;
    border:1px solid #2e3250!important;border-radius:10px!important;
    color:#e8eaf0!important}
.stTextInput>div>div>input:focus{border-color:#7c5cfc!important;
    box-shadow:none!important}

/* security box */
.sec-box{background:#0e1621;border:1px solid #1b3a5c;border-radius:10px;
         padding:.7rem 1rem;font-size:.8rem;color:#90a4ae;line-height:1.7}
.sec-box b{color:#64b5f6}
</style>
""", unsafe_allow_html=True)

# ── Session defaults ───────────────────────────────────────────────────────────
for k, v in dict(logged_in=False, username="",
                 page="login", last_recs=None).items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# PAGE A — LOGIN
# ══════════════════════════════════════════════════════════════════════════════
def page_login():
    st.markdown("""
    <div style='text-align:center;padding:1.4rem 0 .5rem'>
        <div style='font-size:3rem'>🎵</div>
        <h1 style='font-size:1.95rem;margin:.15rem 0;
            background:linear-gradient(135deg,#7c5cfc,#e85d9b);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
            MoodTunes</h1>
        <p style='color:#888;font-size:.87rem'>
            Mood-Based Music Recommender</p>
    </div>""", unsafe_allow_html=True)

    with st.form("login_form"):
        st.markdown("#### 🔐 Sign In")
        uname = st.text_input("Username", placeholder="your_username")
        pw    = st.text_input("Password", type="password",
                               placeholder="Enter password")
        c1, c2 = st.columns(2)
        with c1: do_login  = st.form_submit_button("Sign In →",
                                                    use_container_width=True,
                                                    type="primary")
        with c2: go_signup = st.form_submit_button("Create Account",
                                                    use_container_width=True)

    if do_login:
        if not uname.strip() or not pw:
            st.error("Both fields are required.")
        elif login(uname.strip(), pw):
            st.session_state.update(logged_in=True,
                                     username=uname.strip(),
                                     page="app")
            st.rerun()
        else:
            st.error("❌ Incorrect username or password.")

    if go_signup:
        st.session_state["page"] = "signup"; st.rerun()

    st.markdown("""
    <p style='text-align:center;color:#555;font-size:.73rem;margin-top:1rem'>
        🔒 SHA-256 hashed passwords &nbsp;·&nbsp; Fernet encrypted usernames
    </p>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE B — SIGNUP
# ══════════════════════════════════════════════════════════════════════════════
def page_signup():
    st.markdown("""
    <div style='text-align:center;padding:1.2rem 0 .4rem'>
        <div style='font-size:2.6rem'>🎵</div>
        <h1 style='font-size:1.8rem;margin:.15rem 0;
            background:linear-gradient(135deg,#7c5cfc,#e85d9b);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
            Create Account</h1>
    </div>""", unsafe_allow_html=True)

    with st.form("signup_form"):
        st.markdown("#### ✏️ Register")
        nu   = st.text_input("Choose Username",
                              placeholder="3–20 chars, letters/numbers/_")
        np1  = st.text_input("Choose Password",   type="password",
                              placeholder="Minimum 6 characters")
        np2  = st.text_input("Confirm Password",  type="password",
                              placeholder="Repeat your password")
        c1, c2 = st.columns(2)
        with c1: do_reg = st.form_submit_button("Register →",
                                                 use_container_width=True,
                                                 type="primary")
        with c2: go_log = st.form_submit_button("← Back to Login",
                                                 use_container_width=True)

    if do_reg:
        if np1 != np2:
            st.error("Passwords do not match.")
        else:
            ok, msg = signup(nu.strip(), np1)
            if ok:
                st.success(f"✅ {msg}  — Sign in above.")
                st.session_state["page"] = "login"; st.rerun()
            else:
                st.error(f"❌ {msg}")

    if go_log:
        st.session_state["page"] = "login"; st.rerun()

    st.markdown("""
    <div class='sec-box'>
        <b>🔒 What we store:</b><br>
        &nbsp;&nbsp;• Username → <b>Fernet (AES-128)</b> encrypted before saving<br>
        &nbsp;&nbsp;• Password → <b>SHA-256</b> hashed before saving<br>
        &nbsp;&nbsp;• Plain-text credentials are <b>never written to disk</b>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE C — MAIN APP  (Recommend + History + Security panel)
# ══════════════════════════════════════════════════════════════════════════════
def page_app():
    user = st.session_state["username"]

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style='background:#1a1d28;border:1px solid #2e3250;border-radius:12px;
             padding:.75rem 1rem;text-align:center;margin-bottom:.9rem'>
            <div style='font-size:1.4rem'>🎵</div>
            <div style='font-weight:700;font-size:.95rem;
                 background:linear-gradient(135deg,#7c5cfc,#e85d9b);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
                 MoodTunes</div>
            <div style='color:#888;font-size:.73rem;margin-top:.2rem'>
                👤 {user}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("### 🎭 Select Mood")
        mood = st.selectbox("Mood", MOODS, label_visibility="collapsed",
                             format_func=lambda m: f"{MOOD_EMOJI[m]}  {m}")

        st.markdown("### 🎸 Genre Filter")
        genre = st.selectbox("Genre", GENRES, label_visibility="collapsed")

        st.markdown("### 🎵 How many songs?")
        n = st.slider("N", 3, 10, 5, label_visibility="collapsed")

        st.markdown("---")
        btn_rec = st.button("🎶  Recommend Songs",
                             use_container_width=True, type="primary")
        btn_rnd = st.button("🎲  Random Pick",
                             use_container_width=True)
        st.markdown("---")
        if st.button("🚪  Logout", use_container_width=True):
            st.session_state.update(logged_in=False, username="",
                                     page="login", last_recs=None)
            st.rerun()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <h1 style='font-size:1.8rem;margin-bottom:.1rem'>
        🎵 Welcome, <span style='color:#a78bfa'>{user}</span>!
    </h1>
    <p style='color:#888;font-size:.85rem;margin-bottom:.8rem'>
        TF-IDF · Cosine Similarity · SHA-256 Auth · Fernet Encryption</p>
    """, unsafe_allow_html=True)

    # ── Stats row ──────────────────────────────────────────────────────────────
    hist = load_history(user)
    total  = len(hist)
    fav    = hist["Mood"].value_counts().idxmax() if not hist.empty else "—"
    fav_lbl = f"{MOOD_EMOJI.get(fav,'')} {fav}" if fav != "—" else "—"

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("🎭 Mood",       f"{MOOD_EMOJI.get(mood,'')} {mood}")
    with c2: st.metric("🔍 Searches",   total)
    with c3: st.metric("🏆 Fav Mood",   fav_lbl)

    st.markdown("---")

    # ── Run recommendations ────────────────────────────────────────────────────
    results = None

    if btn_rec:
        with st.spinner("Finding your songs…"):
            results = recommend(mood, genre, n)
        save_history(user, mood,
                     results.iloc[0]["song_name"] if not results.empty else "")
        st.session_state["last_recs"] = results.to_dict("records")

    elif btn_rnd:
        results = random_picks(n)
        st.session_state["last_recs"] = results.to_dict("records")

    # ── Display results ────────────────────────────────────────────────────────
    display = results.to_dict("records") if results is not None \
              else (st.session_state.get("last_recs") or [])

    if display:
        first_mood = display[0].get("mood","Happy")
        mc = MOOD_COLOUR.get(first_mood, "#7c5cfc")
        tag = (f"{MOOD_EMOJI.get(mood,'')} {mood}" +
               (f"  ·  {genre}" if genre != "Any" else ""))

        st.markdown(f"""
        <div style='text-align:center;padding:.3rem 0 .7rem'>
            <span style='font-weight:700;color:{mc};font-size:1rem'>{tag}</span>
            <span style='color:#888;font-size:.8rem'>
                &nbsp;·&nbsp; {len(display)} songs</span>
        </div>""", unsafe_allow_html=True)

        for i, row in enumerate(display, 1):
            rc = MOOD_COLOUR.get(row["mood"], "#7c5cfc")
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>
                    <span style='color:{rc};font-weight:800;margin-right:.3rem'>
                        {i}.</span>{row['song_name']}
                </div>
                <div class='card-sub'>
                    🎤 {row['artist']}
                    &nbsp;&nbsp;
                    <span class='pill'>{row['mood']}</span>
                    <span class='pill'>{row['genre']}</span>
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:2.5rem 0;color:#555'>
            <div style='font-size:2.5rem'>🎧</div>
            <p style='margin-top:.5rem'>
                Choose a mood in the sidebar and click
                <b style='color:#ccc'>Recommend Songs</b>
            </p>
        </div>""", unsafe_allow_html=True)

    # ── Mood history ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Your Mood History")
    if hist.empty:
        st.info("No history yet — start recommending to build your profile!")
    else:
        mc_df = hist["Mood"].value_counts().reset_index()
        mc_df.columns = ["Mood","Sessions"]
        st.bar_chart(mc_df.set_index("Mood")["Sessions"])
        with st.expander("📋 View full history"):
            st.dataframe(hist, use_container_width=True, hide_index=True)

    # ── Security info panel ────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("🔒 Security & Encryption Details"):
        demo_pw  = "demo_password"
        demo_enc = encrypt_data(user)
        demo_h   = hash_password(demo_pw)

        st.markdown("""
        | Layer | Method | What's stored |
        |---|---|---|
        | Password | SHA-256 (hashlib) | 64-char hex digest |
        | Username | Fernet AES-128 | Base64 encrypted token |
        | Mood history | Fernet AES-128 | Username encrypted |
        | Secret key | Fernet keygen | `secret.key` file |
        | ML model | TF-IDF + Cosine | In-memory (no persistence) |
        """)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🔑 SHA-256 live demo:**")
            st.code(
                f'Input : "{demo_pw}"\n'
                f'Hash  : {demo_h[:32]}…\n'
                f'Length: {len(demo_h)} chars',
                language="text")
        with col_b:
            st.markdown("**🔐 Fernet live demo:**")
            st.code(
                f'Input    : "{user}"\n'
                f'Encrypted: {demo_enc[:36]}…\n'
                f'Decrypted: {decrypt_data(demo_enc)}',
                language="text")


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state["logged_in"]:
    if st.session_state["page"] == "signup":
        page_signup()
    else:
        page_login()
else:
    page_app()
