import os
import json
import base64
import requests
from tqdm import tqdm
from collections import defaultdict
import time

# === 🔐 API KEY ===
api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "❌ OPENAI_API_KEY not set in environment."

# === 🔧 PATHS ===
TEST_JSON_PATH = "/Users/s0a0igg/Documents/reps/VPP/data/RQA_V0/all_premises/uid/test001.jsonl"
IMG_DIR = "/Users/s0a0igg/Documents/reps/VPP/data/RQA_V0/images/"
OUTPUT_PATH = "test_results_cot_gpt4o.json"

# === 🖼️ Image Encoding ===
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# === 🧭 Data Grouping ===
def get_maps(data):
    """Groups all SP examples by pmc_id and all other examples by qa_id."""
    sp_map = defaultdict(list)
    qa_map = defaultdict(list)
    for ex in data:
        if ex["source"] == "SP" and ex["truth"]:
            sp_map[ex["pmc_id"]].append(ex)
        elif ex["truth"]:
            qa_map[ex["qa_id"]].append(ex)
    return sp_map, qa_map

# === 💬 GPT‑4o Call ===
def get_vqa_responses(image_base64, questions):
    """
    Sends numbered statements to GPT‑4o and expects a list of 'True'/'False' answers.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Build numbered multi‑line question block
    numbered_questions = "\n".join([f"{i+1}. {q.strip()}" for i, q in enumerate(questions)])

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert visual reasoning assistant.\n"
                "You will be shown a scientific chart image and several numbered statements about it.\n"
                "For each statement, decide whether it is factually correct according to the chart.\n"
                "Respond with a single Python‑style list of 'True' or 'False' values in order.\n"
                "Example: [True, False, True]\n"
                "Do NOT explain or add any text beyond the list."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": f"The statements are:\n{numbered_questions}"}
            ]
        }
    ]

    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 5 * len(questions),
        "temperature": 0.0
    }

    response = requests.post("https://api.openai.com/v1/chat/completions",
                             headers=headers, json=payload)

    try:
        text = response.json()["choices"][0]["message"]["content"].strip()
    except KeyError:
        print("❌ API error:", json.dumps(response.json(), indent=2))
        return None

    # Parse list output like [True, False, True]
    answers = [a.strip() for a in text.replace("[", "").replace("]", "").split(",")]
    answers = [a for a in answers if a in {"True", "False"}]
    return answers

# === 🚀 CoT Inference Runner ===
def run_chain_inference(test_json_path, img_dir, output_path, max_chains=None, resume=False):
    with open(test_json_path, "r") as f:
        all_data = [json.loads(line.strip()) for line in f]

    sp_map, qa_map = get_maps(all_data)

    # Load existing results if resuming
    done_ids = set()
    all_results = []
    if resume and os.path.exists(output_path):
        with open(output_path, "r") as f:
            all_results = json.load(f)
            done_ids = {r['id'] for r in all_results}
        print(f"🔄 Resume enabled — {len(done_ids)} entries already processed.")

    chains_done = 0

    for i, (qa_id, chain_data) in enumerate(tqdm(qa_map.items(), desc="Running CoT Chains")):
        if max_chains and chains_done >= max_chains:
            break

        # Filter chain to only unprocessed, truth==True examples
        chain_data = [ex for ex in chain_data if ex['truth'] and ex['id'] not in done_ids]
        if not chain_data:
            continue  # skip this qa_id if all done

        pmc_id = chain_data[0]["pmc_id"]
        image_file = chain_data[0]["image"].split("/")[-1]
        image_path = os.path.join(img_dir, image_file)

        if not os.path.exists(image_path):
            print(f"⚠️ Missing image: {image_path}")
            continue

        # Construct chain: SP + truth==True DP/MP/RP
        chain = sp_map.get(pmc_id, []) + sorted(chain_data, key=lambda x: (x["source"], x["tag"]))

        # Format questions
        questions = [
            f"{j+1}. {ex['conversations'][0]['value'].replace('<image>', '').split(']', maxsplit=3)[-1].strip()}\n"
            for j, ex in enumerate(chain)
        ]
        expected_answers = [ex["conversations"][1]["value"].strip() for ex in chain]

        image_base64 = encode_image_to_base64(image_path)
        predicted_answers = get_vqa_responses(image_base64, questions)

        if not predicted_answers or len(predicted_answers) != len(questions):
            print(f"⚠️ Skipping qa_id={qa_id} (expected {len(questions)}, got {len(predicted_answers)})")
            print('expected_answers', expected_answers)
            print('predicted_answers', predicted_answers)
            continue

        for ex, pred in zip(chain, predicted_answers):
            all_results.append({
                "id": ex["id"],
                "qa_id": ex["qa_id"],
                "qid": ex["qid"],
                "image": image_file,
                "tag": ex["tag"],
                "truth": ex["truth"],
                "source": ex["source"],
                "question": ex["conversations"][0]["value"],
                "expected": ex["conversations"][1]["value"],
                "predicted": pred
            })

        chains_done += 1
        time.sleep(1.1)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Saved CoT chain predictions to: {output_path}")

if __name__ == "__main__":
    run_chain_inference(TEST_JSON_PATH, IMG_DIR, OUTPUT_PATH, max_chains=50000, resume=True)
