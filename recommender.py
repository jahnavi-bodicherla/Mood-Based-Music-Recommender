"""
recommender.py — ML + Hybrid Recommendation Engine
────────────────────────────────────────────────────
• TF-IDF + Logistic Regression (100+ training sentences, 10 moods)
• 5-component hybrid scoring formula
• Multi-mood round-robin mixing
• Dynamic language weight boosting
• Profile distance scoring
"""

import re
import warnings
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent

# ── Config loading (safe fallback) ─────────────────────────────────────────────
_FALLBACK_CFG = {
    "mood_profiles": {
        "happy":      {"energy": 0.80, "valence": 0.90, "tempo": 120},
        "sad":        {"energy": 0.20, "valence": 0.20, "tempo":  60},
        "energetic":  {"energy": 0.95, "valence": 0.70, "tempo": 140},
        "relaxed":    {"energy": 0.30, "valence": 0.60, "tempo":  70},
        "romantic":   {"energy": 0.50, "valence": 0.75, "tempo":  90},
        "angry":      {"energy": 0.90, "valence": 0.30, "tempo": 130},
        "chill":      {"energy": 0.40, "valence": 0.65, "tempo":  80},
        "focus":      {"energy": 0.30, "valence": 0.50, "tempo":  75},
        "party":      {"energy": 0.95, "valence": 0.90, "tempo": 132},
        "devotional": {"energy": 0.20, "valence": 0.70, "tempo":  65},
    },
    "scoring_weights": {
        "mood_match":       0.30,
        "popularity":       0.20,
        "language_match":   0.20,
        "user_feedback":    0.15,
        "profile_distance": 0.15,
    },
    "language_boost": {"threshold": 0.60, "boost": 0.05},
}

def _load_cfg() -> dict:
    p = ROOT / "config.yaml"
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data:
                return data
        except Exception:
            pass
    return _FALLBACK_CFG

_CFG          = _load_cfg()
MOOD_PROFILES = {k.lower(): v for k, v in
                 _CFG.get("mood_profiles", _FALLBACK_CFG["mood_profiles"]).items()}
WEIGHTS       = _CFG.get("scoring_weights", _FALLBACK_CFG["scoring_weights"])
LANG_BOOST    = _CFG.get("language_boost",  _FALLBACK_CFG["language_boost"])
ALL_MOODS     = sorted(MOOD_PROFILES.keys())

# ──────────────────────────────────────────────────────────────────────────────
# TRAINING DATA  —  120 sentences × 10 moods (12 per mood)
# ──────────────────────────────────────────────────────────────────────────────
_TRAIN = {
    "happy": [
        "I feel so happy and excited right now",
        "feeling joyful and cheerful today",
        "I am thrilled and upbeat everything is great",
        "great mood today life is wonderful",
        "I feel amazing fantastic and full of positivity",
        "celebrating feeling on top of the world",
        "smiling laughing having so much fun",
        "I am delighted and overjoyed today",
        "wonderful day feeling grateful blessed and happy",
        "I feel light and full of energy and joy",
        "everything is going well feeling fantastic",
        "I am in a great mood feeling cheerful",
    ],
    "sad": [
        "I feel sad and lonely right now",
        "feeling down depressed and low",
        "I am heartbroken and want to cry",
        "missing someone close feeling very empty",
        "I feel hopeless and gloomy inside",
        "feeling blue and melancholy all day long",
        "I am very sad and nothing feels right",
        "I feel deep pain and grief today",
        "life feels dark and pointless right now",
        "I feel broken and alone everything hurts",
        "crying and feeling miserable today",
        "I am devastated and numb from sadness",
    ],
    "energetic": [
        "I feel pumped and full of energy",
        "time for a workout feeling fired up",
        "I am motivated and ready to conquer the day",
        "let us go feeling powerful and unstoppable",
        "I feel like running and sprinting hard",
        "gym session beast mode activated going hard",
        "feeling hyper and buzzing with adrenaline",
        "I have so much energy cannot sit still",
        "ready to crush it today hustling hard",
        "charged up feeling alive intense powerful",
        "I feel explosive strength and determination today",
        "high energy and ready for anything today",
    ],
    "relaxed": [
        "I feel calm and very peaceful inside",
        "want to relax and completely unwind",
        "feeling serene tranquil and totally at ease",
        "I want to rest quietly and slow down",
        "gentle easy peaceful quiet mood right now",
        "feeling at peace still and very mellow",
        "breathe slowly and just relax completely",
        "I need to de-stress and find calm",
        "peaceful Sunday afternoon vibes only",
        "feeling content still and well rested",
        "I am completely relaxed and without worry",
        "soft quiet and comfortable place feeling calm",
    ],
    "romantic": [
        "I am in love and feeling so romantic",
        "thinking about someone very special today",
        "I miss you and feel deep love for you",
        "feeling affectionate tender and warm inside",
        "I want to cuddle and be close to you",
        "love is in the air tonight feeling it",
        "feeling intimate passionate and deeply attached",
        "my heart is full of love and warmth",
        "I adore my partner feeling cherished loved",
        "romantic evening candles soft music and love",
        "I feel a deep connection and affection",
        "in love and wanting to share it with you",
    ],
    "angry": [
        "I feel very angry and furious right now",
        "I am so mad and deeply frustrated",
        "feeling intense rage and irritation",
        "I want to vent I am totally livid today",
        "so annoyed and completely pissed off right now",
        "feeling explosive aggressive and intensely angry",
        "I am outraged and boiling with anger inside",
        "nobody listens I am absolutely fed up",
        "I am seething with anger feeling aggressive",
        "I hate this situation so angry right now",
        "full of rage and frustration today angry",
        "I am furious and cannot calm down at all",
    ],
    "chill": [
        "I want to chill and relax at home",
        "lazy day just vibing and hanging out",
        "feeling laid back comfortable and totally easy",
        "just chilling cozy evening at home tonight",
        "I want to vibe with some soft background music",
        "easy breezy relaxed and calm slow afternoon",
        "coffee and music low key mood today",
        "feeling mellow and totally easygoing right now",
        "no stress just hanging and chilling tonight",
        "cozy rainy day indoors just vibing all day",
        "I feel like taking it slow and easy today",
        "soft playlist and a comfortable couch feeling chill",
    ],
    "focus": [
        "I need to concentrate and study hard now",
        "time to work and focus very deeply today",
        "I am doing deep work writing coding thinking",
        "need background music to help me concentrate",
        "studying for important exams need to focus hard",
        "working on a project need full mental clarity",
        "I want to be productive sharp and focused today",
        "deep focus mode research and deep analysis",
        "deadline is coming need to concentrate now",
        "blocking all distractions and focusing hard today",
        "I need instrumental music to stay in the zone",
        "brain in work mode need calm music to focus",
    ],
    "party": [
        "let us party and dance all night long",
        "I want to celebrate and have great fun tonight",
        "going to a club it is party time tonight",
        "feeling festive and totally ready to dance",
        "it is the weekend let us get wild",
        "I want upbeat songs for a big party tonight",
        "night out with friends dancing and celebrating",
        "celebration vibes party mode totally switched on",
        "dance floor energy feeling fully alive tonight",
        "I want to groove and bounce all night long",
        "bass drops and lights I am ready to party",
        "best night out with everyone dancing hard",
    ],
    "devotional": [
        "I want to pray and meditate peacefully today",
        "feeling very spiritual and deeply peaceful inside",
        "time for worship devotion and sincere prayer",
        "I want to connect deeply with god today",
        "feeling divine blessed and very grateful today",
        "temple visit and devotional music mood today",
        "I want bhajans mantras and spiritual songs",
        "spiritual journey and searching for inner peace",
        "feeling sacred holy and completely grateful",
        "time for meditation prayer and inner reflection",
        "I feel a deep spiritual calm and serenity",
        "seeking divine guidance and peace through music",
    ],
}

# ── Build TF-IDF + Logistic Regression classifier ─────────────────────────────
def _build_classifier() -> Pipeline:
    texts, labels = [], []
    for mood, sentences in _TRAIN.items():
        for s in sentences:
            texts.append(s.lower())
            labels.append(mood)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=4000,
            sublinear_tf=True,
            min_df=1,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=2.0,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    pipe.fit(texts, labels)
    return pipe

_CLASSIFIER = _build_classifier()


def detect_mood(text: str) -> tuple[str, float]:
    """
    Classify free-form text into one of 10 moods.
    Returns (mood_str, confidence_0_to_1).
    Falls back to 'happy' with 0.5 confidence on empty input.
    """
    if not text or not text.strip():
        return "happy", 0.5
    clean = re.sub(r"[^\w\s]", " ", text.lower().strip())
    proba = _CLASSIFIER.predict_proba([clean])[0]
    idx   = int(np.argmax(proba))
    return _CLASSIFIER.classes_[idx], round(float(proba[idx]), 3)


# ── Data loading ───────────────────────────────────────────────────────────────
def load_songs(path=None) -> pd.DataFrame:
    """Load and preprocess songs.csv. Normalises tempo and popularity."""
    p  = Path(path) if path else ROOT / "songs.csv"
    df = pd.read_csv(p)
    df["mood"]     = df["mood"].str.strip().str.lower()
    df["language"] = df["language"].str.strip()
    for col in ["energy","valence","popularity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(
                0.5 if col != "popularity" else 50)
    if "tempo" in df.columns:
        df["tempo"] = pd.to_numeric(df["tempo"], errors="coerce").fillna(100)
    # Normalised tempo [0,1] for distance calculations
    t_min, t_max   = df["tempo"].min(), df["tempo"].max()
    df["tempo_norm"] = (df["tempo"] - t_min) / (t_max - t_min + 1e-9)
    # Normalised popularity [0,1]
    p_min, p_max   = df["popularity"].min(), df["popularity"].max()
    df["pop_norm"] = (df["popularity"] - p_min) / (p_max - p_min + 1e-9)
    return df


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_age_group(age: int) -> str:
    if age < 20: return "teen"
    if age < 31: return "young_adult"
    if age < 51: return "adult"
    return "senior"


def compute_language_weight(liked: list, df: pd.DataFrame,
                             base: float = 0.20) -> float:
    """
    Boost language weight by +0.05 when >60% of liked songs share a language.
    Helps personalise recommendations for regional music lovers.
    """
    if len(liked) < 3:
        return base
    liked_df = df[df["song"].isin(liked)]
    if liked_df.empty:
        return base
    ratio = liked_df["language"].value_counts(normalize=True).iloc[0]
    th    = LANG_BOOST.get("threshold", 0.60)
    boost = LANG_BOOST.get("boost", 0.05)
    return min(base + boost, 0.30) if ratio > th else base


def _profile_dist_score(df: pd.DataFrame, mood: str) -> np.ndarray:
    """
    profile_distance = |energy−ideal| + |valence−ideal| + |tempo_norm−ideal|
    Inverted & normalised to [0,1] so higher score = closer to mood ideal.
    """
    p    = MOOD_PROFILES.get(mood, {"energy":0.5,"valence":0.5,"tempo":100})
    t_mn = df["tempo"].min(); t_mx = df["tempo"].max()
    ideal_tn = (p["tempo"] - t_mn) / (t_mx - t_mn + 1e-9)
    dist = (np.abs(df["energy"].values   - p["energy"])  +
            np.abs(df["valence"].values  - p["valence"]) +
            np.abs(df["tempo_norm"].values - ideal_tn))
    return 1.0 - dist / 3.0   # invert: higher = better


# ── Hybrid Recommender ─────────────────────────────────────────────────────────
def recommend(
    df: pd.DataFrame,
    mood: str,
    preferred_language: str = "Any",
    age: int = 25,
    n: int = 5,
    liked: list = None,
    disliked: list = None,
    language_weight: float = None,
) -> pd.DataFrame:
    """
    Hybrid scoring:
      score = W_mood  * mood_match
            + W_pop   * popularity
            + W_lang  * language_match
            + W_fb    * feedback_boost
            + W_dist  * profile_distance

    Parameters
    ----------
    df                 : songs dataframe
    mood               : target mood string
    preferred_language : filter/weight by language
    age                : used to derive age-group preference
    n                  : number of results
    liked              : song names the user liked (boost)
    disliked           : song names the user disliked (exclude)
    language_weight    : override for language score weight
    """
    liked    = liked    or []
    disliked = disliked or []
    mood     = mood.lower().strip()

    # --- exclude disliked songs ---
    pool = df[~df["song"].isin(disliked)].copy()

    # --- 1. Mood match ---
    mood_scores = (pool["mood"] == mood).astype(float).values

    # --- 2. Popularity (normalised) ---
    pop_scores = pool["pop_norm"].values

    # --- 3. Language match ---
    if preferred_language and preferred_language != "Any":
        lang_scores = (pool["language"] == preferred_language).astype(float).values
    else:
        lang_scores = np.ones(len(pool))

    # --- 4. Feedback boost ---
    fb_scores = np.where(pool["song"].isin(liked), 1.0, 0.0)

    # --- 5. Profile distance ---
    dist_scores = _profile_dist_score(pool, mood)

    # --- weights ---
    w = dict(WEIGHTS)
    if language_weight is not None:
        w["language_match"] = language_weight

    # --- final score ---
    score = (w.get("mood_match",       0.30) * mood_scores  +
             w.get("popularity",       0.20) * pop_scores   +
             w.get("language_match",   0.20) * lang_scores  +
             w.get("user_feedback",    0.15) * fb_scores    +
             w.get("profile_distance", 0.15) * dist_scores)

    pool = pool.copy()
    pool["_score"] = score
    drop = [c for c in ["_score","tempo_norm","pop_norm"] if c in pool.columns]
    result = (pool.nlargest(n, "_score")
                  .drop(columns=drop, errors="ignore")
                  .reset_index(drop=True))
    return result


def multi_mood_recommend(
    df: pd.DataFrame,
    moods: list,
    preferred_language: str = "Any",
    age: int = 25,
    n: int = 10,
    liked: list = None,
    disliked: list = None,
) -> pd.DataFrame:
    """
    Round-robin interleave from multiple moods then deduplicate.
    E.g. moods=["happy","party"] → alternates happy/party songs.
    """
    liked    = liked    or []
    disliked = disliked or []
    lw       = compute_language_weight(liked, df)
    per_mood = max(2, -(-n // len(moods)))   # ceiling division

    buckets = [
        recommend(df, m, preferred_language, age,
                  per_mood, liked, disliked, lw)
        for m in moods
    ]

    mixed, seen = [], set()
    for i in range(max(len(b) for b in buckets)):
        for bucket in buckets:
            if i < len(bucket):
                sname = bucket.iloc[i]["song"]
                if sname not in seen:
                    seen.add(sname)
                    mixed.append(bucket.iloc[i])

    return pd.DataFrame(mixed).head(n).reset_index(drop=True)


# ── Analytics helpers ──────────────────────────────────────────────────────────
def mood_language_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["mood","language"]).size().unstack(fill_value=0)


def mood_popularity_table(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("mood")["popularity"]
              .mean()
              .reset_index()
              .rename(columns={"popularity":"avg_popularity"})
              .sort_values("avg_popularity", ascending=False))
