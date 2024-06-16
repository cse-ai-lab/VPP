import numpy as np

# Define dummy data for sequences and premises
# Each sequence is a list of binary values indicating if a premise is correct (1) or not (0)
sequences = [
    [1, 1, 1, 0],  # Sequence 1
    [1, 0, 1],     # Sequence 2
    [1, 1, 1, 1],  # Sequence 3
    [0, 0, 0],     # Sequence 4
    [1, 1, 0, 0],  # Sequence 5
]

def calculate_acc_vpp(sequences):
    """Calculate the Accuracy for Visual Premise Proving (Acc_VPP)."""
    S = len(sequences)
    total = 0
    for seq in sequences:
        total += np.prod(seq)
    acc_vpp = total / S
    return acc_vpp

def calculate_dcp(sequences):
    """Calculate the Depth of Correct Premises (DCP)."""
    S = len(sequences)
    S_correct = sum(np.prod(seq) for seq in sequences)
    S_incorrect = S - S_correct
    
    if S_incorrect == 0:  # Avoid division by zero
        return 0
    
    total_depth = 0
    for seq in sequences:
        if np.prod(seq) == 0:
            total_depth += sum(seq) / len(seq)
    
    dcp = total_depth / S_incorrect
    return dcp

# Calculate metrics
acc_vpp = calculate_acc_vpp(sequences)
dcp = calculate_dcp(sequences)

print(f"Acc_VPP: {acc_vpp:.3f}")
print(f"DCP: {dcp:.3f}")
