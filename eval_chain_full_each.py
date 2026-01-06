import argparse
import pandas as pd
import numpy as np


def calculate_all_metrics(expected, predicted, qa_id_groups=None, verbose=True):
    assert len(expected) == len(predicted), "Mismatched lengths"
    total_samples = len(expected)

    correct_premises = [int(e == p) for e, p in zip(expected, predicted)]
    premise_acc = sum(correct_premises) / total_samples

    if qa_id_groups is None:
        qa_id_groups = [[i] for i in range(total_samples)]

    sequences = []
    fully_correct_chains = 0
    dcp_sum = 0
    dcp_count = 0
    s_len = 0
    min_len = 999
    max_len = 0 

    for indices in qa_id_groups:
        chain = [correct_premises[i] for i in indices]
        sequences.append(chain)
        s_len += len(chain)
        if len(chain) > max_len : 
            max_len = len(chain)
        if len(chain) < min_len : 
            min_len = len(chain)

        if all(chain):
            fully_correct_chains += 1
        else:
            dcp_sum += sum(chain) / len(chain)
            dcp_count += 1

    acc_vpp = fully_correct_chains / len(qa_id_groups)
    dcp = dcp_sum / dcp_count if dcp_count > 0 else 0

    if verbose:
        print(f"[DEBUG] Total Sequences: {len(sequences)}")
        print(f"[DEBUG] Avg seq len : {s_len}/{len(sequences)} = {s_len/len(sequences)}")
        print(f"[DEBUG] min seq len : {min_len} Max len {max_len}")
        print(f"[DEBUG] Total premises: {total_samples}")
        print(f"[DEBUG] Correct premises: {sum(correct_premises)} / {total_samples}")
        print(f"[DEBUG] Fully correct chains: {fully_correct_chains} / {len(qa_id_groups)}")
        print(f"[DEBUG] Incorrect chains for DCP: {dcp_count}")

    return {
        "Premise Accuracy": premise_acc,
        "Premise Accuracy Count": f"{sum(correct_premises)} / {total_samples}",
        "Acc_VPP": acc_vpp,
        "Acc_VPP Count": f"{fully_correct_chains} / {len(qa_id_groups)}",
        "DCP": dcp,
        "DCP Count": f"{dcp_count} chains",
        "Num Chains": len(qa_id_groups),
    }


def reconstruct_chained_true_with_sps(df):
    true_only = df[df["truth"] == True]
    print('len(true_only)', len(true_only))
    sp_rows = true_only[true_only["source"] == "SP"]
    print('len(sp_rows)', len(sp_rows))
    chain_rows = true_only[true_only["source"] != "SP"]
    print('true, non sp rows', len(chain_rows))
    qa_id_chain = df[(df["source"] == "DP") & (df['truth'] == True)]['qa_id'].unique()
    print('dp true unique qid set', len(qa_id_chain))
    chain_rows = chain_rows[chain_rows['qa_id'].isin(qa_id_chain)]
    chain_groups = chain_rows.groupby("qa_id")
    print('len(chain_groups)', len(chain_groups))
    full_chains = []
    group_indices = []
    index = 0
    for qa_id, group in chain_groups:
        image_path = group["image"].iloc[0]
        relevant_sps = sp_rows[sp_rows["image"] == image_path]
        combined = pd.concat([group, relevant_sps], ignore_index=True)
        full_chains.append(combined)
        group_indices.append(list(range(index, index + len(combined))))
        index += len(combined)
    all_chains_df = pd.concat(full_chains, ignore_index=True)
    return all_chains_df, group_indices


def sample_failed_chains(df, group_indices, expected, predicted, max_samples=5):
    failed = []
    for indices in group_indices:
        if not all(predicted[i] == expected[i] for i in indices):
            failed.append(df.iloc[indices])
        if len(failed) >= max_samples:
            break
    print("\n[Sample Failed Chains]\n")
    for i, f in enumerate(failed):
        print(f"--- Failed Chain {i+1} ---")
        print(f[["qa_id", "source", "tag", "truth", "predicted"]])


def main(input_path):
    df = pd.read_json(input_path)
    print(f"[INFO] Loaded {len(df)} premises from {input_path}")

    full_df, qa_id_groups = reconstruct_chained_true_with_sps(df)
    expected = full_df["truth"].astype(bool).tolist()
    predicted = full_df["predicted"].astype(str).str.lower() == "true"

    metrics = calculate_all_metrics(expected, predicted, qa_id_groups)

    for k, v in metrics.items():
        print(f"{k}: {v}")

    sample_failed_chains(full_df, qa_id_groups, expected, predicted)

    # Write to CSV
    output_path = input_path.rsplit(".", 1)[0] + "_chain_.csv"
    full_df.to_csv(output_path, index=False)
    print(f"[INFO] Wrote reconstructed chains to {output_path}")

'''
Singleton Chain Eval. 
This file is to reconstruct and evaluate chain metrics for singleton output. 
Singleton Output has 51007 rows for each test set premise. 
This consists of unique premises and partial chains. 
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct chained-true VPP chains with SP tiling.")
    parser.add_argument("input_file", type=str, help="Path to JSON or JSONL prediction file.")
    args = parser.parse_args()
    main(args.input_file)
