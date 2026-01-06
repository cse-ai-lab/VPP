import os
import json
import base64
import requests
import time
from tqdm import tqdm
import re

# === 🔧 PATHS ===
TEST_JSON_PATH = "/Users/s0a0igg/Documents/reps/VPP/data/RQA_V0/all_premises/uid/test001.jsonl"
IMG_DIR = "/Users/s0a0igg/Documents/reps/VPP/data/RQA_V0/images/"
OUTPUT_PATH = "test_results_gpt4o.json"

# === 🔑 API KEY ===
api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "❌ OPENAI_API_KEY not set in environment."

# === 🔄 Helper ===
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_vqa_response(image_base64, questions):
    if isinstance(questions, str):
        questions = [questions]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. For each question about the image, "
                "reply with a single word: 'True' or 'False'. Do not add any explanation. "
                "Only return exactly 'True' or 'False' for each question."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                *[{"type": "text", "text": q} for q in questions]
            ]
        }
    ]

    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 1 * len(questions)
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return [choice['message']['content'].strip() for choice in response.json()['choices']]
    except Exception as e:
        print("❌ Error:", e)
        print("Response text:", response.text if 'response' in locals() else "No response")
        return None

# === 🚀 Main Inference ===
def run_inference(test_json_path, img_dir, output_path, num_samples=None, resume=True):
    # Load input data
    with open(test_json_path, "r") as f:
        lines = [json.loads(line.strip()) for line in f]

    if num_samples:
        lines = lines[:num_samples]

    # Resume support
    existing_ids = set()
    results = []

    if resume and os.path.exists(output_path):
        with open(output_path, "r") as f:
            results = json.load(f)
            existing_ids = {r["id"] if "id" in r else f"{r['qa_id']}__{r['qid']}" for r in results}

    try:
        for data in tqdm(lines, desc="Running GPT-4o inference"):
            uid = data.get("id", f"{data['qa_id']}__{data['qid']}")
            if uid in existing_ids:
                continue

            image_file = data["image"].split("/")[-1]
            image_path = os.path.join(img_dir, image_file)

            if not os.path.exists(image_path):
                print(f"⚠️ Skipping missing image: {image_path}")
                continue

            question = data["conversations"][0]["value"].replace("<image>", "").strip()
            question = re.sub(r"^(?:\[[^\]]+\])+", "", question).strip()

            expected_answer = data["conversations"][1]["value"].strip()
            image_base64 = encode_image_to_base64(image_path)

            predicted = get_vqa_response(image_base64, question)

            if not predicted:
                print(f"⚠️ Skipping due to model failure: {uid}")
                print('expected_answer', expected_answer)
                continue

            results.append({
                "id": uid,
                "qa_id": data["qa_id"],
                "qid": data["qid"],
                "image": image_file,
                "tag": data["tag"],
                "truth": data["truth"],
                "source": data["source"],
                "question": question,
                "predicted": predicted[0] if isinstance(predicted, list) else predicted
            })

            # Delay for rate limiting
            time.sleep(1.1)

            # Save intermediate results every 10
            if len(results) % 10 == 0:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)

    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt detected! Saving current results to disk...")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Checkpoint saved at: {output_path}")
        return

    # Final save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved final predictions to: {output_path}")
 

# === 🔧 Run
if __name__ == "__main__":
    run_inference(TEST_JSON_PATH, IMG_DIR, OUTPUT_PATH, num_samples=None, resume=True)
