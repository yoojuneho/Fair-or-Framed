"""
kappa.py – Fleiss' κ (3-rater) + Extract disagreement cases
Raters:
  • Human      →  "Human's Bias", "Human's analysis"
  • Human2     →  "Human's Bias(1)", "Human's analysis(1)"
  • GPT        →  "GPT's Bias", "GPT's analysis"
"""
import os, json, pandas as pd, matplotlib.pyplot as plt, matplotlib
from statsmodels.stats.inter_rater import fleiss_kappa

matplotlib.use("Agg")  # Headless rendering for environments without display

# ──────────────── Settings ────────────────
ROOT_DIR = ""  # ← Set correct path
VALID_BIASES = {"left", "neutral", "right"}
SUPPORTER_CATEGORIES = ["left", "right", "left -> right", "right -> left"]

# ──────────────── Utilities ────────────────
def parse_supporter(cat_dict: dict) -> dict:
    """Convert category dict → {name: category}"""
    res = {}
    if not cat_dict:
        return res
    for cat in SUPPORTER_CATEGORIES:
        raw = cat_dict.get(cat, "")
        for n in (x.strip() for x in raw.split(",") if x.strip()):
            res[n] = cat
    return res

def to_matrix(triples, categories):
    """Convert to Fleiss κ input matrix: n_items × n_categories"""
    mat = [[0] * len(categories) for _ in range(len(triples))]
    for i, tri in enumerate(triples):
        for lab in tri:
            try:
                j = categories.index(lab)
                mat[i][j] += 1
            except ValueError:
                pass
    return mat

# ──────────────── Main ────────────────
def process_folder(root_dir: str):
    bias_tri, head_tri, concl_tri, supp_tri = [], [], [], []
    disagreements = []

    for root, _, files in os.walk(root_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(root, fname)
            try:
                data = json.load(open(fpath, encoding="utf-8"))
            except Exception as e:
                print(f"❌ {fpath}: {e}")
                continue

            for r_idx, run in enumerate(data):
                for a_idx, art in enumerate(run.get("articles", [])):
                    # ── Labels from 3 raters ──
                    tri_bias = [art.get("Human's Bias", "").strip(),
                                art.get("Human's Bias(1)", "").strip(),
                                art.get("GPT's Bias", "").strip()]

                    h1 = art.get("Human's analysis", {})
                    h2 = art.get("Human's analysis(1)", {})
                    g = art.get("GPT's analysis", {})

                    tri_head = [h1.get("headline", "").strip(),
                                h2.get("headline", "").strip(),
                                g.get("headline", "").strip()]

                    tri_concl = [h1.get("Conclusion (article/model thoughts)", "").strip(),
                                 h2.get("Conclusion (article/model thoughts)", "").strip(),
                                 g.get("Conclusion (article/model thoughts)", "").strip()]

                    # ── Supporter label comparisons ──
                    sup_diff = {}
                    s1 = parse_supporter(h1.get("Supporter (interview respondent) quote", {}))
                    s2 = parse_supporter(h2.get("Supporter (interview respondent) quote", {}))
                    sg = parse_supporter(g.get("Supporter (interview respondent) quote", {}))

                    for name in set(s1) | set(s2) | set(sg):
                        tri_sup = [s1.get(name), s2.get(name), sg.get(name)]
                        if None in tri_sup:
                            continue  # Must have labels from all 3 raters
                        supp_tri.append(tri_sup)  # ❶ Always add

                        if len(set(tri_sup)) > 1:  # Record disagreement
                            sup_diff[name] = tri_sup

                    # ── Collect for Fleiss κ ──
                    if all(l in VALID_BIASES for l in tri_bias):
                        bias_tri.append(tri_bias)
                    if all(l in VALID_BIASES for l in tri_head):
                        head_tri.append(tri_head)
                    if all(l in VALID_BIASES for l in tri_concl):
                        concl_tri.append(tri_concl)

                    # ── Store disagreement cases ──
                    diff_block, flag = {}, False
                    if len(set(tri_bias)) > 1 and all(l in VALID_BIASES for l in tri_bias):
                        diff_block["bias"] = tri_bias
                        flag = True
                    if len(set(tri_head)) > 1 and all(l in VALID_BIASES for l in tri_head):
                        diff_block["headline"] = tri_head
                        flag = True
                    if len(set(tri_concl)) > 1 and all(l in VALID_BIASES for l in tri_concl):
                        diff_block["conclusion"] = tri_concl
                        flag = True
                    if sup_diff:
                        diff_block["supporter"] = sup_diff
                        flag = True

                    if flag:
                        disagreements.append({
                            "source_file": fpath,
                            "run_index": r_idx,
                            "article_index": a_idx,
                            "headline": art.get("headline", ""),
                            "article": art.get("article", ""),
                            "differences": diff_block
                        })

    # ── Calculate Fleiss κ ──
    κ_bias = fleiss_kappa(to_matrix(bias_tri, list(VALID_BIASES))) if bias_tri else None
    κ_head = fleiss_kappa(to_matrix(head_tri, list(VALID_BIASES))) if head_tri else None
    κ_concl = fleiss_kappa(to_matrix(concl_tri, list(VALID_BIASES))) if concl_tri else None
    κ_supp = fleiss_kappa(to_matrix(supp_tri, SUPPORTER_CATEGORIES)) if supp_tri else None

    print("\n=== Fleiss' κ (3-rater) ===")
    for name, val in [("Bias", κ_bias), ("Headline", κ_head),
                      ("Conclusion", κ_concl), ("Supporter", κ_supp)]:
        print(f"{name:10}: {val:.3f}" if val is not None else f"{name:10}: N/A")

    # ── Save disagreement JSON ──
    out_json = os.path.join(root_dir, "disagreement_cases.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(disagreements, f, indent=2, ensure_ascii=False)
    print(f"\n✅ {len(disagreements)} disagreement cases saved → {out_json}")

    # ── Save Fleiss κ table as image ──
    df = pd.DataFrame(
        [["Bias", κ_bias], ["Headline", κ_head], ["Conclusion", κ_concl], ["Supporter", κ_supp]],
        columns=["Metric", "Fleiss κ"]
    )
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.3)
    plt.savefig(os.path.join(root_dir, "fleiss_kappa_table.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("🖼️  Fleiss κ table saved.")

# ──────────────── Entry Point ────────────────
if __name__ == "__main__":
    process_folder(ROOT_DIR)
