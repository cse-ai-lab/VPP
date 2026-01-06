import os
import json
import time
import re
from tqdm import tqdm
from PIL import Image
from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted

# === 🔧 PATHS ===
TEST_JSON_PATH = "/Users/s0a0igg/Documents/reps/VPP/data/RQA_V0/all_premises/uid/test001.jsonl"
IMG_DIR = "/Users/s0a0igg/Documents/reps/VPP/data/RQA_V0/images/"
OUTPUT_PATH = "test_results_gemini_2_5.json"

# === 🔑 API KEY ===
api_key = os.getenv("GEMINI_API_KEY")
assert api_key, "❌ GEMINI_API_KEY not set in environment."

# === 🧠 Gemini Client ===
client = genai.Client(api_key=api_key)
model = "gemini-2.5-pro"

# === 📷 Image Loader with MIME detection ===
def load_image_bytes_and_mime(path):
    try:
        with open(path, "rb") as f:
            img_bytes = f.read()
        ext = os.path.splitext(path)[-1].lower()
        mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
        return img_bytes, mime
    except Exception as e:
        print(f"❌ Error loading image {path}: {e}")
        return None, None

# === 🤖 Gemini 2.5 VQA ===
def get_gemini_vqa_response(image_bytes, mime_type, question):
    """
    Works with latest google‑genai client for Gemini 2.5 (Pydantic 2.x).
    Uses types.Blob for image + Part for both text and image.
    """
    try:
        prompt = (
            "You are a helpful assistant. For each question about the image, "
            "reply with a single word: 'True' or 'False'. Do not add any explanation. "
            "Only return exactly 'True' or 'False'."
        )

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type,
                            data=image_bytes,
                        )
                    ),
                    types.Part(
                        text=f"{prompt}\n\nQ: {question}\nA:"
                    ),
                ],
            )
        ]

        tools = [types.Tool(googleSearch=types.GoogleSearch())]
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            tools=tools,
        )

        stream = client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        )

        full_text = "".join(
            [chunk.text for chunk in stream if hasattr(chunk, "text") and chunk.text]
        ).strip()

        return full_text

    except ResourceExhausted as e:
        print("⚠️ Rate limit hit:", e)
        time.sleep(5)
        return None
    except Exception as e:
        print("❌ Gemini API Error:", e)
        return None

# === 🚀 Main Inference ===
def run_inference(test_json_path, img_dir, output_path, num_samples=None, resume=True):
    with open(test_json_path, "r") as f:
        lines = [json.loads(line.strip()) for line in f]

    if num_samples:
        lines = lines[:num_samples]

    existing_ids = set()
    results = []
    if resume and os.path.exists(output_path):
        with open(output_path, "r") as f:
            results = json.load(f)
            existing_ids = {r["id"] if "id" in r else f"{r['qa_id']}__{r['qid']}" for r in results}

    try:
        for data in tqdm(lines, desc="Running Gemini 2.5 inference"):
            uid = data.get("id", f"{data['qa_id']}__{data['qid']}")
            if uid in existing_ids:
                continue

            image_file = data["image"].split("/")[-1]
            image_path = os.path.join(img_dir, image_file)
            if not os.path.exists(image_path):
                print(f"⚠️ Missing image: {image_path}")
                continue

            question = data["conversations"][0]["value"].replace("<image>", "").strip()
            question = re.sub(r"^(?:\[[^\]]+\])+", "", question).strip()
            expected_answer = data["conversations"][1]["value"].strip()

            image_bytes, mime_type = load_image_bytes_and_mime(image_path)
            if image_bytes is None:
                continue

            predicted = get_gemini_vqa_response(image_bytes, mime_type, question)

            if not predicted or predicted not in {"True", "False"}:
                print(f"⚠️ Model failed to respond clearly for {uid}: {predicted}")
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
                "predicted": predicted
            })

            if len(results) % 10 == 0:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)

            time.sleep(1.1)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted. Saving results...")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print("✅ Partial results saved.")
        return

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved full predictions to: {output_path}")

# === 🔧 Run ===
if __name__ == "__main__":
    run_inference(TEST_JSON_PATH, IMG_DIR, OUTPUT_PATH, num_samples=None, resume=True)
