import os
import sys
import json
import pandas as pd

def load_json_to_df(filepath):
    """Load a JSON file into a pandas DataFrame."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

def deduplicate_by_id(df):
    """Remove duplicates based on the 'id' column."""
    return df.drop_duplicates(subset='id', keep='first')

def compute_accuracy_per_tag(df):
    """
    Compare 'truth' (bool) vs 'predicted' (string) and compute accuracy per 'tag'.
    Returns a DataFrame with accuracy statistics.
    """
    df = df.copy()
    # Normalize predicted strings to boolean
    df['predicted_bool'] = df['predicted'].str.lower().map({'true': True, 'false': False})
    df['correct'] = df['truth'] == df['predicted_bool']

    grouped = df.groupby('tag')['correct'].agg(['sum', 'count'])
    grouped['accuracy'] = grouped['sum'] / grouped['count']
    grouped = grouped.rename(columns={'sum': 'num_correct', 'count': 'total'})
    
    return grouped.sort_values(by='accuracy', ascending=False)


def compute_accuracy_per_qid(df):
    """
    Compare 'truth' (bool) vs 'predicted' (string) and compute accuracy per 'tag'.
    Returns a DataFrame with accuracy statistics.
    """
    df = df.copy()
    df['predicted_bool'] = df['predicted'].str.lower().map({'true': True, 'false': False})
    df['correct'] = df['truth'] == df['predicted_bool']

    grouped = df.groupby('qid')['correct'].agg(['sum', 'count'])
    grouped['accuracy'] = grouped['sum'] / grouped['count']
    grouped = grouped.rename(columns={'sum': 'num_correct', 'count': 'total'})
    
    return grouped.sort_values(by='accuracy', ascending=False)



def compute_accuracy_per_source(df):
    """
    Compare 'truth' (bool) vs 'predicted' (string) and compute accuracy per 'tag'.
    Returns a DataFrame with accuracy statistics.
    """
    # Normalize predicted strings to boolean
    df = df.copy()
    df['predicted_bool'] = df['predicted'].str.lower().map({'true': True, 'false': False})
    df['correct'] = df['truth'] == df['predicted_bool']

    grouped = df.groupby('source')['correct'].agg(['sum', 'count'])
    grouped['accuracy'] = grouped['sum'] / grouped['count']
    grouped = grouped.rename(columns={'sum': 'num_correct', 'count': 'total'})
    
    return grouped.sort_values(by='accuracy', ascending=False)




def compute_accuracy_per_image(df):
    """
    Compare 'truth' (bool) vs 'predicted' (string) and compute accuracy per 'tag'.
    Returns a DataFrame with accuracy statistics.
    """
    # Normalize predicted strings to boolean
    df = df.copy()
    df['predicted_bool'] = df['predicted'].str.lower().map({'true': True, 'false': False})
    df['correct'] = df['truth'] == df['predicted_bool']

    grouped = df.groupby('image')['correct'].agg(['sum', 'count'])
    grouped['accuracy'] = grouped['sum'] / grouped['count']
    grouped = grouped.rename(columns={'sum': 'num_correct', 'count': 'total'})
    
    return grouped.sort_values(by='accuracy', ascending=False)


def compute_accuracy_per_truth(df):
    """
    Compare 'truth' (bool) vs 'predicted' (string) and compute accuracy per 'tag'.
    Returns a DataFrame with accuracy statistics.
    """
    # Normalize predicted strings to boolean
    df = df.copy()
    df['predicted_bool'] = df['predicted'].str.lower().map({'true': True, 'false': False})
    df['correct'] = df['truth'] == df['predicted_bool']

    grouped = df.groupby('truth')['correct'].agg(['sum', 'count'])
    grouped['accuracy'] = grouped['sum'] / grouped['count']
    grouped = grouped.rename(columns={'sum': 'num_correct', 'count': 'total'})
    
    return grouped.sort_values(by='accuracy', ascending=False)


def main(argv):
    if len(argv) < 2:
        print("Usage: python analyze_accuracy_by_tag.py <results_file.json>")
        sys.exit(1)

    filepath = argv[1]
    filename_prefix = os.path.splitext(os.path.basename(filepath))[0]  # e.g. "test_results_cot_gemini_2_5"

    df = load_json_to_df(filepath)
    print('df', df)

    df_ = deduplicate_by_id(df)
    print('df_', df_)

    tag_accuracy = compute_accuracy_per_tag(df_)
    print("\nAccuracy by tag:")
    print(tag_accuracy)
    output_filename = f"{filename_prefix}_tag_accuracy.csv"
    tag_accuracy.to_csv(output_filename)
    print(f"\nSaved accuracy stats to {output_filename}")

    source_accuracy = compute_accuracy_per_source(df_)
    print("\nAccuracy by source:")
    print(source_accuracy)
    output_filename = f"{filename_prefix}_source_accuracy.csv"
    source_accuracy.to_csv(output_filename)
    print(f"\nSaved accuracy stats to {output_filename}")

    qid_accuracy = compute_accuracy_per_qid(df_)
    print("\nAccuracy by qid:")
    print(qid_accuracy)
    output_filename = f"{filename_prefix}_qid_accuracy.csv"
    qid_accuracy.to_csv(output_filename)
    print(f"\nSaved accuracy stats to {output_filename}")



    image_accuracy = compute_accuracy_per_image(df_)
    print("\nAccuracy by image:")
    print(image_accuracy)
    output_filename = f"{filename_prefix}_image_accuracy.csv"
    image_accuracy.to_csv(output_filename)
    print(f"\nSaved accuracy stats to {output_filename}")

    truth_accuracy = compute_accuracy_per_truth(df_)
    print("\nAccuracy by Truth:")
    print(truth_accuracy)
    output_filename = f"{filename_prefix}_truth_accuracy.csv"
    truth_accuracy.to_csv(output_filename)
    print(f"\nSaved accuracy stats to {output_filename}")




if __name__ == "__main__":
    main(sys.argv)