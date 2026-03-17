import pandas as pd

# Read the CSV file (this is labeled csv)
df = pd.read_csv('path_to_your_experiment_folder/cluster_metadata_labeled.csv ')  # Replace with your actual file name

# Identify the columns to group by: all except 'Cluster' and 'Count'
group_cols = [col for col in df.columns if col not in ['Cluster', 'Count']]

# Group by those columns and sum the 'Count'
grouped_df = df.groupby(group_cols, as_index=False)['Count'].sum()

grouped_df.to_csv('grouped_by_behavior.csv', index=False)

print("Saved grouped data to 'grouped_by_behavior.csv")
