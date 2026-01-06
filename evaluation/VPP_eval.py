import numpy as np

def calculate_acc_vpp(sequences):
    """Compute Accuracy of Visual Premise Proving (Acc_VPP)."""
    S = len(sequences)
    total = sum(np.prod(seq) for seq in sequences)
    return total / S if S > 0 else 0


def calculate_dcp(sequences):
    """Compute Depth of Correct Premises (DCP)."""
    S = len(sequences)
    S_correct = sum(np.prod(seq) for seq in sequences)
    S_incorrect = S - S_correct

    if S_incorrect == 0:
        return 0

    total_depth = 0
    for seq in sequences:
        if np.prod(seq) == 0:  # only count incorrect sequences
            total_depth += sum(seq) / len(seq)
    return total_depth / S_incorrect


def calculate_premise_accuracy(expected, predicted):
    """Simple per-premise accuracy."""
    return np.mean(np.array(expected) == np.array(predicted))

def calculate_all_metrics(expected, predicted, qa_id_groups=None, verbose=True):
    assert len(expected) == len(predicted), "Mismatched lengths"
    total_samples = len(expected)

    correct_premises = [int(e == p) for e, p in zip(expected, predicted)]
    premise_acc = sum(correct_premises) / total_samples

    # If no grouping provided, treat all as a single chain of length 1 each
    if qa_id_groups is None:
        qa_id_groups = [[i] for i in range(total_samples)]

    sequences = []
    fully_correct_chains = 0
    dcp_sum = 0
    dcp_count = 0
    s_len = 0

    for indices in qa_id_groups:
        chain = [correct_premises[i] for i in indices]
        sequences.append(chain)
        s_len += len(chain)

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
