import pandas as pd
import numpy as np
import argparse

# Argument parser
parser = argparse.ArgumentParser(
    prog='query_params',
    description='Query model parameter combinations from CSV'
)

parser.add_argument('-f', '--file', type=str, required=True, help='Path to the CSV file with all combinations')
parser.add_argument('-o', '--output', type=str, default="filtered_combinations.csv", help='Output CSV file for query results')
parser.add_argument('--step_limit', type=int, default=2, help='Maximum step distance between emb and ffw in index positions')
args = parser.parse_args()

# Load the CSV file
df = pd.read_csv(args.file)

# Convert relevant columns to numeric if necessary
df[['Enc_Layers', 'Dec_Layers', 'Emb', 'FFW', 'Heads', 'Total_Params']] = df[['Enc_Layers', 'Dec_Layers', 'Emb', 'FFW', 'Heads', 'Total_Params']].apply(pd.to_numeric)

# Define the possible values for Emb and FFW (used for indexing)
embed_ffw_values = [256, 512, 1024, 2048, 4096]

# Define a helper function to filter combinations based on "diagonal proximity" for emb and ffw
def filter_near_diagonal(df, embed_ffw_values, step_limit=2):
    """
    Filters the DataFrame to find rows where `emb` and `ffw` are within `step_limit` steps of each other
    in the ordered list `embed_ffw_values`.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the model combinations.
        embed_ffw_values (list of int): The ordered list of possible `emb` and `ffw` values.
        step_limit (int): The maximum index step distance allowed between `emb` and `ffw`.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only the rows that satisfy the condition.
    """
    # Map each emb and ffw value to its index in the embed_ffw_values list
    emb_indices = df['Emb'].map(lambda x: embed_ffw_values.index(x))
    ffw_indices = df['FFW'].map(lambda x: embed_ffw_values.index(x))
    
    # Filter where the index difference between emb and ffw is within the step limit
    filtered_df = df[np.abs(emb_indices - ffw_indices) <= step_limit]
    return filtered_df

# Apply the filter
filtered_df = filter_near_diagonal(df, embed_ffw_values, step_limit=args.step_limit)

# Save the filtered results to a CSV file
filtered_df.to_csv(args.output, index=False)
print(f"Filtered combinations saved to {args.output}")

# Display results in console (optional, can be commented out if not needed)
print("Filtered combinations:")
print(filtered_df)
