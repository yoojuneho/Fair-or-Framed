import os
import json
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────
# 1. Scoring rules
# ─────────────────────────────────────────────
BIAS_SCORE = {
    "left": -2,
    "neutral": 0,
    "right": 2
}

SUPPORTER_SCORE = {
    "left": -1,
    "right": 1,
    "left -> right": 3,
    "right -> left": -3
}

# ─────────────────────────────────────────────
# 2. Bias score calculation function
# ─────────────────────────────────────────────
def calculate_analysis_bias(analysis: dict[str, Any]) -> str:
    """
    Calculates the total bias score based on analysis (dict),
    and returns one of "left", "neutral", or "right".
    """
    total = 0

    # 2-1) Score from headline and conclusion
    headline = analysis.get("headline", "").strip()
    conclusion = analysis.get("Conclusion (article/model thoughts)", "").strip()
    total += BIAS_SCORE.get(headline, 0)
    total += BIAS_SCORE.get(conclusion, 0)

    # 2-2) Score from supporter quotes
    quotes = analysis.get("Supporter (interview respondent) quote", {})
    if isinstance(quotes, dict):
        for field, score in SUPPORTER_SCORE.items():
            supporters = quotes.get(field, "")
            if supporters:
                names = [n.strip() for n in supporters.split(",") if n.strip()]
                total += len(names) * score

    # 2-3) Return final bias category
    return "left" if total < 0 else "right" if total > 0 else "neutral"

# ─────────────────────────────────────────────
# 3. Process a single JSON file
# ─────────────────────────────────────────────
def process_json_file(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error {path} → {e}")
        return

    # Support both list and dict formats
    runs = data if isinstance(data, list) else [data]
    modified = False

    # Mapping between bias field and corresponding analysis field
    bias_targets = [
        ("Human's Bias",     "Human's analysis"),
        ("Human's Bias(1)",  "Human's analysis(1)"),
        ("GPT's Bias",       "GPT's analysis")
    ]

    for run in runs:
        for art in run.get("articles", []):
            for bias_key, ana_key in bias_targets:
                # If bias value is empty, calculate and insert
                if not art.get(bias_key):
                    new_bias = calculate_analysis_bias(art.get(ana_key, {}))
                    art[bias_key] = new_bias
                    modified = True

    if modified:
        with path.open("w", encoding="utf-8") as f:
            # Save as dict or list depending on original format
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Updated: {path}")
    else:
        print(f"➖ No update: {path}")

# ─────────────────────────────────────────────
# 4. Traverse folder and process all JSON files
# ─────────────────────────────────────────────
def process_folder(root_dir: str) -> None:
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        print(f"Path does not exist: {root}")
        return

    for file in root.rglob("*.json"):
        process_json_file(file)

# ─────────────────────────────────────────────
# 5. Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Root folder containing JSON files
    ROOT_FOLDER = ""
    process_folder(ROOT_FOLDER)
