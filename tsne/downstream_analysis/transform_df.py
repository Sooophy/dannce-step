import pandas as pd


# Load the data
root = "path_to_your_experment_folder"
df = pd.read_csv(root + '/cluster_metadata.csv')

# Create a pivot table
pivot_df = df.pivot_table(index=['AnimalID', 'Timepoint', 'Cylinder', 'Condition'], 
                          columns='Cluster', 
                          values='Count', 
                          fill_value=0).reset_index()

# Rename the cluster columns
pivot_df.columns = [f'Cluster_{col}' if isinstance(col, int) else col for col in pivot_df.columns]

print(pivot_df)
pivot_df.to_csv(root + '/cluster_metadata_transform_groupcondition.csv', index=False)
