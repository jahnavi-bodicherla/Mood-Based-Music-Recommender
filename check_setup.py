"""
check_setup.py — Run this FIRST to diagnose your MoodTunes setup.

Usage:
  python check_setup.py

Open a terminal IN your project folder, then run the command above.
It will tell you exactly what is missing and how to fix it.
"""
import sys, os

print("=" * 60)
print("  MoodTunes — Setup Checker")
print("=" * 60)
print(f"\n📁 Current folder:\n   {os.getcwd()}\n")

errors   = []
warnings = []

# ── Required files ─────────────────────────────────────────────
REQUIRED = {
    "app.py":          "Main Streamlit UI",
    "recommender.py":  "ML recommendation engine",
    "security.py":     "Security utilities (hashing, tokens)",
    "songs.csv":       "Song dataset",
    "config.yaml":     "Mood profiles & weights",
    "requirements.txt":"Python dependencies",
}
OPTIONAL = {
    "spotify.py":      "Spotify API (Live Mode)",
    "youtube.py":      "YouTube API (Live Mode)",
    ".env":            "API keys (Live Mode only)",
}

print("📄 Required files:")
for f, desc in REQUIRED.items():
    ok = os.path.exists(f)
    print(f"   {'✅' if ok else '❌'}  {f:<22} {desc}")
    if not ok:
        errors.append(
            f"MISSING: {f}\n"
            f"   → Download it from the chat and save it to:\n"
            f"   → {os.getcwd()}\\{f}")

print("\n📄 Optional files (needed for Live Mode):")
for f, desc in OPTIONAL.items():
    ok = os.path.exists(f)
    print(f"   {'✅' if ok else '⚠️ '}  {f:<22} {desc}")
    if not ok and f != ".env":
        warnings.append(
            f"{f} missing — Live Mode will be disabled. "
            f"Download from chat to enable it.")

# ── Python version ─────────────────────────────────────────────
print(f"\n🐍 Python: {sys.version.split()[0]}")
if sys.version_info < (3, 8):
    errors.append("Python 3.8+ required.")

# ── Packages ───────────────────────────────────────────────────
PKGS = [
    ("streamlit",   "streamlit"),
    ("pandas",      "pandas"),
    ("sklearn",     "scikit-learn"),
    ("yaml",        "pyyaml"),
    ("numpy",       "numpy"),
    ("requests",    "requests"),
    ("dotenv",      "python-dotenv"),
]
print("\n📦 Packages:")
missing_pkgs = []
for imp, pip in PKGS:
    try:
        mod = __import__(imp)
        ver = getattr(mod, "__version__", "ok")
        print(f"   ✅  {pip:<20} {ver}")
    except ImportError:
        print(f"   ❌  {pip:<20} NOT INSTALLED")
        missing_pkgs.append(pip)

if missing_pkgs:
    errors.append(
        f"Missing packages: {' '.join(missing_pkgs)}\n"
        f"   → Run: pip install {' '.join(missing_pkgs)}")

# ── Test core imports ──────────────────────────────────────────
if not missing_pkgs and os.path.exists("recommender.py"):
    print("\n🔬 Logic tests:")
    sys.path.insert(0, os.getcwd())
    try:
        from recommender import load_songs, detect_mood, recommend
        print("   ✅  recommender.py imports OK")
        if os.path.exists("songs.csv"):
            df = load_songs("songs.csv")
            r  = recommend(df, "happy", "Any", 22, 3)
            m, c = detect_mood("I feel so happy today")
            print(f"   ✅  songs.csv: {len(df)} songs loaded")
            print(f"   ✅  detect_mood: '{m}' ({c:.0%} confidence)")
            print(f"   ✅  recommend: {len(r)} results returned")
    except Exception as e:
        errors.append(f"recommender.py error: {e}")

    if os.path.exists("security.py"):
        try:
            from security import hash_password, verify_password, generate_token
            h, salt = hash_password("test")
            assert verify_password("test", h, salt)
            print("   ✅  security.py: hashing & tokens OK")
        except Exception as e:
            errors.append(f"security.py error: {e}")

# ── Summary ────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors:
    print(f"❌  {len(errors)} issue(s) found:\n")
    for i, e in enumerate(errors, 1):
        print(f"  [{i}] {e}\n")
    print("─" * 60)
    print("QUICK FIX:\n")
    print("  Step 1 — Install all packages:")
    print("    pip install streamlit pandas scikit-learn pyyaml "
          "numpy requests python-dotenv\n")
    print("  Step 2 — Make sure ALL these files are in ONE folder:")
    for f in list(REQUIRED) + ["spotify.py", "youtube.py"]:
        print(f"    • {f}")
    print(f"\n  Your folder: {os.getcwd()}")
    print("\n  Step 3 — Run:")
    print("    streamlit run app.py")
else:
    if warnings:
        print("⚠️  Warnings (non-fatal):")
        for w in warnings:
            print(f"  • {w}")
    print("\n✅  Everything looks good! Run:\n")
    print(f"    streamlit run app.py\n")
    print(f"  Folder: {os.getcwd()}")

print("=" * 60)
