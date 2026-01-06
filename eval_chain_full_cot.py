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

def identify_cot_chains(df):
    print('*' * 80)
    print('In Identify Chain')

    # Step 1: Get valid qa_ids from DP rows with truth == True
    qa_id_chain = df[(df["source"] == "DP") & (df['truth'] == True)]['qa_id'].unique()
    print('dp true unique qa_id set:', len(qa_id_chain))

    if 'chain_id' in df.columns:
        # Step 2A: Group by chain_id and propagate qa_id from non-SP rows
        true_qa_ids = df[df['source'] != 'SP'].groupby('chain_id')['qa_id'].first()
        df['qa_id'] = df['chain_id'].map(true_qa_ids)

        # Filter to only rows with valid qa_id
        chain_rows = df[df['qa_id'].isin(qa_id_chain)]
        chain_groups = chain_rows.groupby('chain_id')

        print('len(chain_groups):', len(chain_groups))
        print('*' * 80)

        # Return grouped row indices
        return [list(indices) for indices in chain_groups.groups.values()]

    else:
        # Step 2B: Fallback — ordered split by SP0 tags
        chains = []
        current_chain = []

        for idx, row in df.iterrows():
            if row.get('tag') == 'SP0':
                if current_chain:
                    chains.append(current_chain)
                current_chain = [idx]
            else:
                current_chain.append(idx)
        if current_chain:
            chains.append(current_chain)

        print('Total raw chains (SP0 split):', len(chains))

        # Filter: Keep only chains where any row has qa_id in qa_id_chain
        valid_chains = [
            chain for chain in chains
            if any(df.loc[i, 'qa_id'] in qa_id_chain for i in chain)
        ]

        print('Valid chains after qa_id filtering:', len(valid_chains))
        print('*' * 80)
        return valid_chains


def sample_failed_chains(df, chain_indices, expected, predicted, max_samples=5):
    failed = []
    for indices in chain_indices:
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

    chain_indices = identify_cot_chains(df)
    expected = df["truth"].astype(bool).tolist()
    predicted = df["predicted"].astype(str).str.lower() == "true"

    metrics = calculate_all_metrics(expected, predicted, chain_indices)

    for k, v in metrics.items():
        print(f"{k}: {v}")

    sample_failed_chains(df, chain_indices, expected, predicted)

    output_path = input_path.rsplit(".", 1)[0] + ".cot.csv"
    df.to_csv(output_path, index=False)
    print(f"[INFO] Wrote CoT-evaluated data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CoT-mode chained VPP predictions.")
    parser.add_argument("input_file", type=str, help="Path to JSON or JSONL prediction file.")
    args = parser.parse_args()
    main(args.input_file)
