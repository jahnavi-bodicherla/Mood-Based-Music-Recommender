"""
recommender.py — TF-IDF Mood Classifier + Hybrid Recommendation Engine
"""
import re, warnings, numpy as np, pandas as pd, yaml
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

# ── Config ──────────────────────────────────────────────────────────────────
def _cfg():
    p = ROOT / "config.yaml"
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {}

_C = _cfg()
MOOD_PROFILES: dict = {k.lower(): v for k, v in
    _C.get("mood_profiles", {
        "happy":{"energy":0.80,"valence":0.90,"tempo":120},
        "sad":{"energy":0.20,"valence":0.20,"tempo":60},
        "energetic":{"energy":0.95,"valence":0.70,"tempo":140},
        "relaxed":{"energy":0.30,"valence":0.60,"tempo":70},
        "romantic":{"energy":0.50,"valence":0.75,"tempo":90},
        "angry":{"energy":0.90,"valence":0.30,"tempo":130},
        "chill":{"energy":0.40,"valence":0.65,"tempo":80},
        "focus":{"energy":0.30,"valence":0.50,"tempo":75},
        "party":{"energy":0.95,"valence":0.90,"tempo":132},
        "devotional":{"energy":0.20,"valence":0.70,"tempo":65},
    }).items()}

WEIGHTS: dict = _C.get("scoring_weights", {
    "similarity":0.30,"mood_match":0.20,"language_match":0.15,
    "age_match":0.10,"popularity":0.10,"user_feedback":0.10,"profile_distance":0.05
})
LANG_BOOST: dict = _C.get("language_boost", {"threshold":0.60,"amount":0.05})
ALL_MOODS: list  = sorted(MOOD_PROFILES.keys())

# ── Training data (120 sentences × 10 moods) ───────────────────────────────
_TRAIN = {
    "happy":["I feel so happy and excited","feeling joyful and cheerful today","I am thrilled everything is wonderful","great mood today life is beautiful","I feel amazing and full of positivity","celebrating feeling on top of the world","smiling laughing having so much fun","I am delighted and overjoyed","wonderful day feeling grateful and blessed","I feel light and full of joy","everything is great feeling fantastic","I am in a great mood right now"],
    "sad":["I feel sad and lonely","feeling down depressed and low","I am heartbroken and want to cry","missing someone feeling very empty","I feel hopeless and gloomy inside","feeling blue and melancholy all day","I am very sad nothing feels right","I feel deep pain and grief","life feels dark and pointless","I feel broken and alone everything hurts","crying and feeling miserable today","I am devastated and numb from sadness"],
    "energetic":["I feel pumped and full of energy","time for a workout feeling fired up","I am motivated and ready to conquer the day","feeling powerful and unstoppable","I feel like running and sprinting hard","gym session beast mode activated","feeling hyper and buzzing with adrenaline","I have so much energy cannot sit still","ready to crush it today hustling hard","charged up feeling alive intense and powerful","I feel explosive strength and determination","high energy and ready for anything"],
    "relaxed":["I feel calm and very peaceful","want to relax and completely unwind","feeling serene tranquil and at ease","I want to rest quietly and slow down","gentle easy peaceful quiet mood","feeling at peace still and mellow","breathe slowly and just relax completely","I need to de-stress and find calm","peaceful Sunday afternoon vibes","feeling content still and well rested","I am completely relaxed and without worry","soft quiet comfortable feeling calm"],
    "romantic":["I am in love and feeling so romantic","thinking about someone very special","I miss you and feel deep love","feeling affectionate tender and warm","I want to cuddle and be close","love is in the air tonight","feeling intimate passionate and attached","my heart is full of love and warmth","I adore my partner feeling cherished","romantic evening candles and soft music","I feel a deep connection and affection","in love and wanting to share it"],
    "angry":["I feel very angry and furious","I am so mad and deeply frustrated","feeling intense rage and irritation","I want to vent I am totally livid","so annoyed and completely pissed off","feeling explosive aggressive and angry","I am outraged and boiling inside","nobody listens I am absolutely fed up","I am seething with anger","I hate this situation so angry right now","full of rage and frustration today","I am furious and cannot calm down"],
    "chill":["I want to chill and relax at home","lazy day just vibing and hanging out","feeling laid back comfortable and easy","just chilling cozy evening at home","I want to vibe with some soft music","easy breezy relaxed and calm afternoon","coffee and music low key mood today","feeling mellow and totally easygoing","no stress just hanging and chilling","cozy rainy day indoors just vibing","I feel like taking it slow today","soft playlist and a comfortable couch"],
    "focus":["I need to concentrate and study hard","time to work and focus very deeply","I am doing deep work writing and coding","need background music to concentrate","studying for important exams need focus","working on a project need mental clarity","I want to be productive and sharp today","deep focus mode research and analysis","deadline is coming need to concentrate","blocking all distractions focusing hard","I need instrumental music to stay in the zone","brain in work mode need calm music"],
    "party":["let us party and dance all night","I want to celebrate and have great fun","going to a club it is party time","feeling festive and ready to dance","it is the weekend let us get wild","I want upbeat songs for a big party","night out with friends dancing and celebrating","celebration vibes party mode on","dance floor energy feeling fully alive","I want to groove and bounce all night","bass drops and lights I am ready to party","best night out with everyone dancing"],
    "devotional":["I want to pray and meditate peacefully","feeling very spiritual and deeply peaceful","time for worship devotion and prayer","I want to connect deeply with god","feeling divine blessed and very grateful","temple visit and devotional music mood","I want bhajans mantras and spiritual songs","spiritual journey and searching for inner peace","feeling sacred holy and completely grateful","time for meditation prayer and inner reflection","I feel a deep spiritual calm and serenity","seeking divine guidance through music"],
}

def _build_model():
    texts, labels = [], []
    for mood, sents in _TRAIN.items():
        for s in sents:
            texts.append(s.lower()); labels.append(mood)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=4000, sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced", random_state=42)),
    ])
    pipe.fit(texts, labels)
    return pipe

_MODEL = _build_model()

def detect_mood(text: str) -> tuple:
    """Returns (mood, confidence). Falls back to ('happy', 0.5) on empty input."""
    if not text or not text.strip():
        return "happy", 0.5
    clean = re.sub(r"[^\w\s]", " ", text.lower().strip())
    proba = _MODEL.predict_proba([clean])[0]
    idx   = int(np.argmax(proba))
    return _MODEL.classes_[idx], round(float(proba[idx]), 3)

# ── Data loading ────────────────────────────────────────────────────────────
def load_songs(path=None) -> pd.DataFrame:
    p  = Path(path) if path else ROOT / "songs.csv"
    df = pd.read_csv(p)
    df["mood"]     = df["mood"].str.strip().str.lower()
    df["language"] = df["language"].str.strip()
    for col in ["energy","valence","popularity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.5 if col!="popularity" else 50)
    df["tempo"] = pd.to_numeric(df.get("tempo", 100), errors="coerce").fillna(100)
    t0, t1 = df["tempo"].min(), df["tempo"].max()
    df["tempo_norm"] = (df["tempo"] - t0) / (t1 - t0 + 1e-9)
    p0, p1 = df["popularity"].min(), df["popularity"].max()
    df["pop_norm"]   = (df["popularity"] - p0) / (p1 - p0 + 1e-9)
    return df

def get_age_group(age: int) -> str:
    if age < 20: return "teen"
    if age < 31: return "young_adult"
    if age < 51: return "adult"
    return "senior"

def compute_language_weight(liked: list, df: pd.DataFrame, base: float = 0.15) -> float:
    if len(liked) < 3: return base
    sub = df[df["song"].isin(liked)]
    if sub.empty: return base
    ratio = sub["language"].value_counts(normalize=True).iloc[0]
    th, amt = LANG_BOOST.get("threshold",0.60), LANG_BOOST.get("amount",0.05)
    return min(base + amt, 0.25) if ratio > th else base

def _profile_dist(df: pd.DataFrame, mood: str) -> np.ndarray:
    p  = MOOD_PROFILES.get(mood, {"energy":0.5,"valence":0.5,"tempo":100})
    t0 = df["tempo"].min(); t1 = df["tempo"].max()
    ideal_tn = (p["tempo"] - t0) / (t1 - t0 + 1e-9)
    dist = (np.abs(df["energy"].values   - p["energy"])  +
            np.abs(df["valence"].values  - p["valence"]) +
            np.abs(df["tempo_norm"].values - ideal_tn))
    return 1.0 - dist / 3.0

# ── Hybrid recommend ────────────────────────────────────────────────────────
def recommend(df, mood, language="Any", age=25, n=5,
              liked=None, disliked=None, language_weight=None):
    liked    = liked    or []
    disliked = disliked or []
    mood     = mood.lower().strip()
    pool     = df[~df["song"].isin(disliked)].copy()

    mood_s = (pool["mood"] == mood).astype(float).values
    lang_s = ((pool["language"]==language).astype(float).values
              if language != "Any" else np.ones(len(pool)))
    pop_s  = pool["pop_norm"].values
    fb_s   = np.where(pool["song"].isin(liked), 1.0, 0.0)
    dist_s = _profile_dist(pool, mood)

    w = dict(WEIGHTS)
    if language_weight is not None:
        w["language_match"] = language_weight

    score = (w.get("mood_match",0.20)       * mood_s +
             w.get("popularity",0.10)        * pop_s  +
             w.get("language_match",0.15)    * lang_s +
             w.get("user_feedback",0.10)     * fb_s   +
             w.get("profile_distance",0.05)  * dist_s)

    pool = pool.copy(); pool["_score"] = score
    drop = [c for c in ["_score","tempo_norm","pop_norm"] if c in pool.columns]
    return pool.nlargest(n,"_score").drop(columns=drop,errors="ignore").reset_index(drop=True)

def multi_mood_recommend(df, moods, language="Any", age=25, n=10,
                          liked=None, disliked=None):
    liked    = liked    or []
    disliked = disliked or []
    lw       = compute_language_weight(liked, df)
    per      = max(2, -(-n // len(moods)))
    buckets  = [recommend(df, m, language, age, per, liked, disliked, lw) for m in moods]
    mixed, seen = [], set()
    for i in range(max(len(b) for b in buckets)):
        for bucket in buckets:
            if i < len(bucket):
                s = bucket.iloc[i]["song"]
                if s not in seen:
                    seen.add(s); mixed.append(bucket.iloc[i])
    return pd.DataFrame(mixed).head(n).reset_index(drop=True)

# Analytics
def mood_lang_pivot(df):  return df.groupby(["mood","language"]).size().unstack(fill_value=0)
def mood_age_pivot(df):   return df.groupby(["mood","age_group"]).size().unstack(fill_value=0)
