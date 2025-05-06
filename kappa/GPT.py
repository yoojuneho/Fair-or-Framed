# Script to evaluate bias in news articles using GPT-4o.

import os
import json
import time
import glob
import argparse
import re
import openai

# =====================================
# 1. Set OpenAI API Key
# =====================================
openai.api_key = ""


# =====================================
# 2. Parse original supporter stances and statements
# =====================================
def parse_stances_from_opinions(sampled_opinions):
    """
    Example input:
      [
        "Alex: (right) In my view, safeguarding our borders ...",
        "Brian: (left) I strongly believe that embracing immigrants ...",
        ...
      ]

    Output format:
    {
      "Alex": {
        "stance": "right",
        "full_text": "Alex: (right) In my view..."
      },
      ...
    }
    """
    stances_dict = {}

    pattern = re.compile(r"(Alex|Brian|Chloe|Daniel|Emily|Frank|Grace|Hannah|Isaac|Julia):?\s*\((right|left)\)", re.IGNORECASE)

    for op_text in sampled_opinions:
        match = pattern.match(op_text)
        if match:
            person = match.group(1)
            stance = match.group(2).lower()
            stances_dict[person] = {
                "stance": stance,
                "full_text": op_text.strip()
            }
    return stances_dict


# =====================================
# 3. Identify quoted supporters in article text
# =====================================
def extract_quoted_supporters(article_text, stances_dict):
    """
    Extract supporters mentioned in the article in the format "Alex(right)"

    Returns: { "Alex": {...}, ... }
    """
    used_supporters = {}
    pattern = re.compile(r"\b(Alex|Brian|Chloe|Daniel|Emily|Frank|Grace|Hannah|Isaac|Julia)\((right|left)\)", re.IGNORECASE)
    matches = pattern.findall(article_text)

    for (person, stance_in_article) in matches:
        if person in stances_dict:
            used_supporters[person] = stances_dict[person]

    return used_supporters


# =====================================
# 4. Request classification from GPT (with few-shot examples)
# =====================================
def gpt_classify_used_supporters(
    topic,
    sampled_opinions,
    used_supporters,
    article_text,
    headline,
    human_bias="",
    stances_dict=None
):
    """
    - Provide all sampled_opinions
    - Provide human-annotated bias
    - Force classification of all quoted supporters
    """

    FEW_SHOT_EXAMPLE = r"""..."""  # [Truncated: unchanged from original post]
    example_json_format = r"""..."""

    sampled_text = "\n".join(sampled_opinions)
    quoted_names = ", ".join(sorted(used_supporters))
    quoted_with_stances = "\n".join([
        f"- {name} ({stances_dict.get(name, {}).get('stance', 'unknown')})"
        for name in sorted(used_supporters)
    ])

    system_role_prompt = """
You are a strict and logical political stance classifier.
... [Truncated for brevity]
""".strip()

    system_prompt = f"""
You are a political bias analyst for the topic: '{topic}'.
... [Truncated for brevity]
{example_json_format}
"""

    combined_system_prompt = f"{system_role_prompt}\n\n{system_prompt}"
    user_prompt = "Please analyze now."

    retries = 3
    for attempt in range(retries):
        try:
            print("==========================")
            print("SYSTEM PROMPT:")
            print("--------------------------")
            print(combined_system_prompt)
            print("--------------------------")
            print("USER PROMPT:")
            print("--------------------------")
            print(user_prompt)
            print("==========================")

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": combined_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0
            )
            result_str = response.choices[0].message.content.strip()
            analysis_json = json.loads(result_str)

            required_keys = ["headline", "Supporter (interview respondent) quote", "Conclusion (article/model thoughts)"]
            for rk in required_keys:
                if rk not in analysis_json:
                    raise ValueError(f"Key '{rk}' missing in GPT response")

            sub_keys = ["left -> right", "right -> left", "left", "right"]
            sq = analysis_json["Supporter (interview respondent) quote"]
            for sk in sub_keys:
                if sk not in sq:
                    raise ValueError(f"Sub-key '{sk}' missing in 'Supporter (interview respondent) quote'")

            return analysis_json

        except (json.JSONDecodeError, ValueError) as je:
            print(f"[ERROR] GPT response invalid or missing keys (attempt {attempt+1}): {je}")
            time.sleep(3)
        except Exception as e:
            print(f"[ERROR] Unexpected GPT error (attempt {attempt+1}): {e}")
            time.sleep(3)

    return {
        "headline": "",
        "Supporter (interview respondent) quote": {
            "left -> right": "",
            "right -> left": "",
            "left": "",
            "right": ""
        },
        "Conclusion (article/model thoughts)": ""
    }


# =====================================
# 5. Process individual JSON file
# =====================================
def process_json_file(file_path, topic):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] reading JSON {file_path}: {e}")
        return

    if not isinstance(data, list):
        print(f"[WARNING] {file_path} - top-level not list. Skip.")
        return

    modified = False
    for run_obj in data:
        sampled_opinions = run_obj.get("sampled_opinions", [])
        articles = run_obj.get("articles", [])

        stances_dict = parse_stances_from_opinions(sampled_opinions)

        for art in articles:
            headline = art.get("headline", "")
            article_text = art.get("article", "")
            used_list = art.get("GPT's analysis", {}).get("used supporter", [])

            if not used_list:
                continue

            detail_analysis = gpt_classify_used_supporters(
                topic,
                sampled_opinions,
                used_list,
                article_text,
                headline,
                art.get("Human's Bias", ""),
                stances_dict
            )

            detail_analysis["used supporter"] = used_list
            art["GPT's analysis"] = detail_analysis
            modified = True

    if modified:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"[SUCCESS] Updated JSON: {file_path}")
        except Exception as e:
            print(f"[ERROR] writing {file_path}: {e}")
    else:
        print(f"[INFO] No changes for {file_path}")


# =====================================
# 6. Process all JSON files in directory
# =====================================
def process_all_jsons_in_directory(root_directory, topic):
    pattern = os.path.join(root_directory, "**", "*.json")
    file_list = glob.glob(pattern, recursive=True)

    if not file_list:
        print(f"[INFO] No JSON files found under {root_directory}")
        return

    print(f"[INFO] Found {len(file_list)} JSON files under {root_directory}")
    for idx, fp in enumerate(file_list, start=1):
        print(f"\n[{idx}/{len(file_list)}] Processing: {fp}")
        process_json_file(fp, topic)


# =====================================
# 7. main()
# =====================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="Topic e.g. Immigration")
    parser.add_argument("--root_folder", required=True, help="Root folder containing .json files")
    args = parser.parse_args()

    process_all_jsons_in_directory(args.root_folder, args.topic)


if __name__ == "__main__":
    main()