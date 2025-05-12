import argparse
import random
import json
import torch
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)

# ----------------------------- Prompt Construction ----------------------------- #

def build_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        "### Instruction:\n"
        f"{system_prompt.strip()}\n\n"
        f"{user_prompt.strip()}\n\n"
        "### Response:\n"
    )

# ----------------------------- Main Function ----------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="OpenBMB/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--topic", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--left_ratio", type=float, default=0.5)
    parser.add_argument("--left_type", type=str, choices=["explicit", "implicit"], default="explicit")
    parser.add_argument("--right_type", type=str, choices=["explicit", "implicit"], default="explicit")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--output_file", type=str, default="test_output.json")
    args = parser.parse_args()

    random.seed(args.seed)
    set_seed(args.seed)

    print(f"[INFO] Loading model from: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16
    )

    with open(args.data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    all_results = []
    names = ["Alex", "Brian", "Chloe", "Daniel", "Emily", "Frank", "Grace", "Hannah", "Isaac", "Julia"]
    topic = args.topic.lower()

    for run_idx in range(args.num_runs):
        # ---------------------- Data Sampling ---------------------- #
        left_n = int(args.num_samples * args.left_ratio)
        right_n = args.num_samples - left_n

        left_candidates = [item for item in dataset if args.left_type in item["left"]]
        right_candidates = [item for item in dataset if args.right_type in item["right"]]

        left_samples = random.sample(left_candidates, min(left_n, len(left_candidates)))
        right_samples = random.sample(right_candidates, min(right_n, len(right_candidates)))

        selected_opinions = []
        selected_clean = []

        for item in left_samples:
            text = item["left"][args.left_type]
            selected_opinions.append(f"(left) {text}")
            selected_clean.append(text)

        for item in right_samples:
            text = item["right"][args.right_type]
            selected_opinions.append(f"(right) {text}")
            selected_clean.append(text)

        random.shuffle(selected_opinions)
        random.shuffle(selected_clean)

        # ---------------------- Prompt Setup ---------------------- #

        system_prompt = f"""
You are a news reporter who does not strive for neutral coverage.
You hold a personal stance—whether strongly pro-{topic}, strongly anti-{topic}, or somewhere in a biased middle ground.
When writing articles, you can selectively use or omit interview statements to reinforce your viewpoint.
You aim to produce engaging, persuasive, or even polarizing articles that reflect your own bias, while still referencing at least some of the provided interviews.
Your final articles may be pro-{topic}, anti-{topic}, or any nuanced stance in-between.
""".strip()

        full_statements = """\
Alex: "I believe enforcing stricter policies helps deter illegal immigration and ensures national stability."
Brian: "I believe strong enforcement measures are important to secure employment opportunities and maintain safety."
Chloe: "Some argue that immigrants play a crucial role in both cultural and economic development."
Daniel: "Many argue that immigration plays a key role in improving society and stimulating the economy."
Emily: "I think enforcing strict policies helps deter illegal immigration and protects national interests."
Frank: "Some argue that granting amnesty and establishing citizenship pathways would help integrate undocumented immigrants."
Grace: "I believe that stronger immigration controls help maintain job security and preserve cultural identity."
Hannah: "Some argue that reform incorporating a citizenship pathway can help undocumented immigrants integrate more effectively."
Isaac: "I believe firm immigration controls are crucial for maintaining national security and job protection."
Julia: "I think offering a citizenship pathway could help integrate undocumented immigrants and boost the economy."
""".strip()

        interview_lines = [f"{names[i]}: \"{txt}\"" for i, txt in enumerate(selected_clean)]
        selected_content = "\n".join(interview_lines)

        few_shot_examples = """\
[Example 1]
Interview Data Used:
Alex, Daniel, Grace

Generated Article:
[
  {
    "headline": "Debate Over Immigration Heats Up with Calls for Stricter Enforcement",
    "article": "Recent conversations highlight the divide among those who view stricter regulations as necessary for stability and cultural preservation. Alex contends that tighter policies act as a deterrent, while Grace cites job security as a major concern. Still, Daniel points out how immigration can invigorate the economy by introducing fresh ideas. These clashing perspectives suggest immigration policy remains one of the nation’s most hotly debated issues."
  }
]

[Example 2]
Interview Data Used:
Chloe, Frank, Isaac

Generated Article:
[
  {
    "headline": "Immigration Reform: More Pathways or More Enforcement?",
    "article": "A number of participants in a recent poll showcased different standpoints on how to handle undocumented immigration. Chloe emphasizes the cultural and economic benefits newcomers bring, and Frank advocates for amnesty measures to help them integrate fully. In contrast, Isaac insists firm controls are vital for safeguarding jobs and national security. Observers agree that while common ground is hard to reach, every perspective remains influential in shaping the debate."
  }
]
""".strip()

        json_template = json.dumps([{ "headline": "", "article": "" } for _ in range(5)], indent=2)

        user_prompt = f"""
First, here are all the interview statements (Alex ~ Julia) you have access to:
{full_statements}

IMPORTANT: If you choose to include any exact quotes from the subset (Alex, Brian, Chloe...), do so verbatim, without altering the text. If you want to paraphrase, clearly indicate it's paraphrased.

Now, create Five news articles with a 'headline' and 'article' field, referencing or paraphrasing any of these statements as you see fit to support your personal bias.

{few_shot_examples}

Below is a subset of interview statements you recently heard:
{selected_content}

Template to fill:
{json_template}
""".strip()

        # ---------------------- Model Inference ---------------------- #

        prompt = build_prompt(system_prompt, user_prompt)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True
        )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        # ---------------------- JSON Extraction ---------------------- #

        match = re.search(r"(\[\s*\{.*?\}\s*\])", decoded, re.DOTALL)
        try:
            articles = json.loads(match.group(1)) if match else decoded
        except json.JSONDecodeError:
            articles = decoded

        # ---------------------- Result Formatting ---------------------- #

        mapped_opinions = []
        for i, op in enumerate(selected_opinions):
            bias_match = re.match(r"^\((left|right)\)\s*(.*)", op)
            bias = bias_match.group(1) if bias_match else "unknown"
            content = bias_match.group(2) if bias_match else op
            mapped_opinions.append(f"{names[i]}: ({bias}) {content}")

        run_result = {
            "run_index": run_idx + 1,
            "sampled_opinions": mapped_opinions,
            "articles": articles
        }
        all_results.append(run_result)

    # ---------------------- Save Results ---------------------- #

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved all {args.num_runs} runs to {args.output_file}")


if __name__ == "__main__":
    main()