import os
import json
import csv
import argparse
from collections import defaultdict
from evaluation.VPP_eval import calculate_all_metrics

def load_predictions(path):
    print(f"[INFO] Loading predictions from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith("["):
            print("[INFO] JSON array format detected.")
            data = json.loads(content)
        else:
            print("[INFO] JSONL format detected.")
            data = [json.loads(line) for line in content.splitlines() if line.strip()]
    print(f"[INFO] Loaded {len(data)} records.")
    return data

def group_by_key(data, key):
    groups = defaultdict(list)
    for i, entry in enumerate(data):
        groups[entry.get(key, f"missing_{key}")].append(i)
    print(f"[INFO] Grouped into {len(groups)} groups by `{key}`")
    return list(groups.values())

def group_by_qa_id(data):
    return group_by_key(data, "qa_id")

def group_by_keys(data, keys):
    groups = defaultdict(list)
    for i, entry in enumerate(data):
        k = tuple(entry.get(k, f"missing_{k}") for k in keys)
        groups[k].append(i)
    print(f"[INFO] Grouped into {len(groups)} groups by {keys}")
    return list(groups.values())

def filter_truth(data, expected, truth_value=True):
    return [d for d, e in zip(data, expected) if e == truth_value]

def get_expected_predicted(data, sanity=False):
    expected = [str(d.get("expected", d.get("truth", False))).lower() == "true" for d in data]
    predicted = expected if sanity else [str(d.get("predicted", "")).lower() == "true" for d in data]
    return expected, predicted

def deduplicate_entries(data):
    seen_ids = set()
    deduped = []
    combined = {}
    for d in data:
        ex_id = d.get("id")
        exp = d.get("expected")
        pred = d.get("predicted")
        if ex_id not in combined:
            combined[ex_id].append({'expected':exp})
        # else : 


            
    print(f"[INFO] Deduplicated entries to {len(deduped)}")
    return deduped

def deduplicate_entries(data):
    seen_ids = set()
    deduped = []
    for d in data:
        ex_id = d.get("id", d.get("qa_id", None))
        if ex_id not in seen_ids:
            deduped.append(d)
            seen_ids.add(ex_id)
    print(f"[INFO] Deduplicated entries to {len(deduped)}")
    return deduped

def evaluate_setting(name, data, expected, predicted, groups=None, return_metrics=False):
    print(f"\n=== Evaluation Report: {name} ===")
    if groups : 
        print('len(groups)', groups)

    metrics = calculate_all_metrics(expected, predicted, groups) if groups else calculate_all_metrics(expected, predicted)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    return metrics if return_metrics else None

def get_avg_chain_length(groups):
    if not groups:
        return 0.0
    return round(sum(len(g) for g in groups) / len(groups), 2)

def write_results_to_csv(results, out_file):
    if not results:
        print("[WARN] No results to write.")
        return
    keys = results[0].keys()
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] Wrote compiled results to {out_file}")

def run_all_modes(prediction_file, sanity=False, output_csv="./compiled_results.csv"):
    all_modes = ["all", "true_only", "false_only", "dedup_all", "dedup_true", "chained_true"]
    results = []

    for mode in all_modes:
        try:
            data = load_predictions(prediction_file)
            expected, predicted = get_expected_predicted(data, sanity=sanity)
            name, avg_chain_len = "", 0.0

            if mode.startswith("dedup"):
                data = deduplicate_entries(data)
                expected, predicted = get_expected_predicted(data, sanity=sanity)
                if "true" in mode:
                    data = filter_truth(data, expected, truth_value=True)
                    expected, predicted = get_expected_predicted(data, sanity=sanity)
                name = f"Deduplicated ({'True' if 'true' in mode else 'All'})"
                metrics = evaluate_setting(name, data, expected, predicted, return_metrics=True)

            elif mode == "chained_true":
                data = filter_truth(data, expected, truth_value=True)
                expected, predicted = get_expected_predicted(data, sanity=sanity)
                name = "Chained (True-only)"
                qa_groups = group_by_qa_id(data)
                avg_chain_len = get_avg_chain_length(qa_groups)
                metrics = evaluate_setting(name, data, expected, predicted, qa_groups, return_metrics=True)

            else:
                if mode == "true_only":
                    data = filter_truth(data, expected, truth_value=True)
                    expected, predicted = get_expected_predicted(data, sanity=sanity)
                    name = "Flat (True-only)"
                elif mode == "false_only":
                    data = filter_truth(data, expected, truth_value=False)
                    expected, predicted = get_expected_predicted(data, sanity=sanity)
                    name = "Flat (False-only)"
                else:
                    name = "Flat (All)"

                if len(data) == 0:
                    print(f"[WARN] Skipping mode {mode}: No premises.")
                    continue

                metrics = evaluate_setting(name, data, expected, predicted, return_metrics=True)
            results.append({
                "Mode": name,
                "Premises": metrics.get("Premise Accuracy Count", "0 / 0").split(" / ")[1],
                "Correct": metrics.get("Premise Accuracy Count", "0 / 0").split(" / ")[0],
                "Premise_Accuracy": round(metrics.get("Premise Accuracy", 0), 4),
                "Chains": metrics.get("Num Chains", 0),
                "Correct Chains": metrics.get("Acc_VPP Count", "0 / 0").split(" / ")[0],
                "Acc_VPP": round(metrics.get("Acc_VPP", 0), 4),
                "DCP": round(metrics.get("DCP", 0), 4),
                "Avg_Chain_Len": avg_chain_len
            })

        except Exception as e:
            print(f"[ERROR] Mode '{mode}' failed: {e}")

    write_results_to_csv(results, output_csv)

def get_output_path(input_path, suffix="output", new_ext=".csv"):
    base, _ = os.path.splitext(input_path)
    return f"{base}.try.{suffix}{new_ext}"

def main(prediction_file, sanity=False, eval_mode="all"):
    data = load_predictions(prediction_file)
    expected, predicted = get_expected_predicted(data, sanity=sanity)
    print('^'*80)
    print('expected', len(expected), sum(expected))
    print('predicted', len(expected), sum(expected))
    print('^'*80)

    if eval_mode.startswith("dedup"):
        data = deduplicate_entries(data)
        expected, predicted = get_expected_predicted(data, sanity=sanity)
        if "true" in eval_mode:
            data = filter_truth(data, expected, truth_value=True)
            expected, predicted = get_expected_predicted(data, sanity=sanity)
        evaluate_setting("Isolated (deduplicated)", data, expected, predicted)

    elif eval_mode == "chained_true":
        data = filter_truth(data, expected, truth_value=True)
        expected, predicted = get_expected_predicted(data, sanity=sanity)
        qa_groups = group_by_qa_id(data)
        evaluate_setting("Chained (True-only)", data, expected, predicted, qa_groups)

    else:
        if eval_mode == "true_only":
            data = filter_truth(data, expected, truth_value=True)
            expected, predicted = get_expected_predicted(data, sanity=sanity)
        elif eval_mode == "false_only":
            data = filter_truth(data, expected, truth_value=False)
            expected, predicted = get_expected_predicted(data, sanity=sanity)
        evaluate_setting("Flat (no group)", data, expected, predicted)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VPP predictions.")
    parser.add_argument("prediction_file", type=str, help="Path to JSON or JSONL prediction file.")
    parser.add_argument("--sanity", action="store_true", help="Sanity check: prediction == expected")
    parser.add_argument("--run_all", action="store_true", help="Run all evaluation modes and write CSV.")
    parser.add_argument("--eval_mode", type=str, default="all", choices=[
        "all", "true_only", "false_only", "dedup_all", "dedup_true", "chained_true"
    ])
    args = parser.parse_args()

    output_path = get_output_path(args.prediction_file)

    if args.run_all:
        run_all_modes(args.prediction_file, sanity=args.sanity, output_csv=output_path)
    else:
        main(args.prediction_file, sanity=args.sanity, eval_mode=args.eval_mode)
