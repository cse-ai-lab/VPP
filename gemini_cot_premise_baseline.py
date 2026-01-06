import os
import json
import time
import re
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted

# === 🔧 PATHS ===
TEST_JSON_PATH = "/Users/s0a0igg/Documents/reps/VPP/data/RQA_V0/all_premises/uid/test001.jsonl"
IMG_DIR = "/Users/s0a0igg/Documents/reps/VPP/data/RQA_V0/images/"
OUTPUT_PATH = "test_results_cot_gemini_2_5.json"

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

# === 📊 Grouping Logic ===
def get_maps(data):
    sp_map = defaultdict(list)
    qa_map = defaultdict(list)
    for ex in data:
        if ex["source"] == "SP" and ex["truth"]:
            sp_map[ex["pmc_id"]].append(ex)
        elif ex["truth"]:
            qa_map[ex["qa_id"]].append(ex)
    return sp_map, qa_map


def get_maps(data):
    sp_map = defaultdict(list)
    qa_map = defaultdict(list)
    for ex in data:
        if ex["source"] == "SP" and ex["truth"]:
            io = ex["image"][:-4]
            sp_map[io].append(ex)
        elif ex["truth"]:
            qa_map[ex["qa_id"]].append(ex)
    return sp_map, qa_map

# === 🧠 Gemini CoT Call (with debug logs) ===

def get_gemini_vqa_list_response(image_bytes, mime_type, questions, debug_id=""):
    """
    Robust Gemini 2.5 streaming inference for multi-question CoT VQA benchmarking.
    Returns list of 'True'/'False' strings.
    """
    try:
        numbered = "\n".join([f"{i+1}. {q.strip()}" for i, q in enumerate(questions)])
        prompt = (
            "You are a scientific reasoning assistant. Based on the image and the following statements,\n"
            "determine whether each is factually correct based only on the image.\n\n"
            "Respond ONLY with a Python list of True or False values, in order.\n"
            "Example: [True, False, True]\n"
            "No explanation or extra text.\n\n"
            f"The statements are:\n{numbered}\n\n"
            "Answer:"
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
                        text=f"{prompt}\n\nThe statements are:\n{numbered}"
                    ),
                ],
            )
        ]

        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens= 1000* len(questions),  # bump this
            safety_settings=[],
            stop_sequences=[],
        )

        stream = client.models.generate_content_stream(
            model="gemini-2.5-pro",
            contents=contents,
            config=config,
        )

        full_text = ""
        for chunk in stream:
            # print("[CHUNK]", chunk)
            if hasattr(chunk, "text") and chunk.text:
                full_text += chunk.text

        full_text = full_text.strip()
        # print("[RAW RESPONSE]", repr(chunk.text))

        # Debug log
        # print(f"📤 [Gemini Output {debug_id}]:", full_text)

        # Parse output like: [True, False, True]
        answers = [a.strip() for a in full_text.replace("[", "").replace("]", "").split(",")]
        answers = [a for a in answers if a in {"True", "False"}]

        return answers

    except ResourceExhausted as e:
        print("⚠️ Rate limit hit:", e)
        time.sleep(5)
        return None
    except Exception as e:
        print(f"❌ Gemini API Error for {debug_id}:", e)
        return None


# === 🚀 Main Inference ===
def run_inference(test_json_path, img_dir, output_path, num_samples=None, resume=True):
    with open(test_json_path, "r") as f:
        lines = [json.loads(line.strip()) for line in f]

    sp_map, qa_map = get_maps(lines)
    qa_items = list(qa_map.items())
    if num_samples:
        qa_items = qa_items[:num_samples]

    existing_ids = set()
    results = []
    if resume and os.path.exists(output_path):
        with open(output_path, "r") as f:
            results = json.load(f)
            existing_ids = {r["id"] for r in results}
        print(f"🔄 Resume enabled — {len(existing_ids)} entries already processed.")
    else:
        print('NO : resume', resume)
        print('output_path',output_path,  os.path.exists(output_path))

    try:
        for qa_id, chain_data in tqdm(qa_items, desc="Running Gemini 2.5 CoT Debug"):
            chain_data = [ex for ex in chain_data if ex["truth"] and ex["id"] not in existing_ids]
            if not chain_data:
                continue

            pmc_id = chain_data[0]["pmc_id"]
            image_file = chain_data[0]["image"].split("/")[-1]
            image_path = os.path.join(img_dir, image_file)

            if not os.path.exists(image_path):
                print(f"⚠️ Missing image: {image_path}")
                continue

            chain = sp_map.get(pmc_id, []) + sorted(chain_data, key=lambda x: (x["source"], x["tag"]))
            questions = [
                ex["conversations"][0]["value"].replace("<image>", "").split("]", maxsplit=3)[-1].strip()
                for ex in chain
            ]
            expected_answers = [ex["conversations"][1]["value"].strip() for ex in chain]

            image_bytes, mime_type = load_image_bytes_and_mime(image_path)
            if image_bytes is None:
                continue

            # === Debug logging ===
            print(f"\n============================")
            print(f"🔍 Processing qa_id={qa_id}, pmc_id={pmc_id}, #premises={len(questions)}")
            print(f"Image: {image_file} | MIME: {mime_type}")
            print(f"Expected: {expected_answers}")
            

            predicted = get_gemini_vqa_list_response(image_bytes, mime_type, questions, debug_id=qa_id)
            
            if not predicted or len(predicted) != len(questions):
                print(f"⚠️ qa_id={qa_id} (expected {len(questions)}, got {len(predicted) if predicted else 0})")
                if not predicted :
                    predicted = ['False']*len(questions)
                if len(predicted) > len(questions):
                    predicted = predicted[:len(questions)]
                if len(predicted) < len(questions):
                    ln_ = ['False'] * (len(questions) - len(predicted))
                    predicted.extend(ln_)

            
            print("predicted:", predicted)

            for ex, pred in zip(chain, predicted):
                results.append({
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

            if len(results) % 5 == 0:  # Save frequently in debug mode
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"💾 [DEBUG] Saved interim results ({len(results)} entries)")

            time.sleep(1.1)

    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt detected — saving current results...")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print("✅ Partial results saved at:", output_path)
        return

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved full CoT predictions to: {output_path}")

# === 🔧 Run ===
if __name__ == "__main__":
    run_inference(TEST_JSON_PATH, IMG_DIR, OUTPUT_PATH, num_samples=None, resume=True)
