import pandas as pd
import numpy as np
import pickle

root = "path_to_your_experiment_folder"
data_obj = pickle.load(open(root + '/datastruct.p','rb'))
# print(data_obj.data.columns.values)

# Load the pickle file
# Replace 'your_file.pkl' with the path to your pickle file
df = data_obj.data[['AnimalID', 'Timepoint', 'Cluster', 'frame', 'Condition', 'Cylinder']]
# Define a function to count unique frames
def count_unique_frames(group):
    # Group by all columns except 'frame' , then count unique frames
    unique_count = group.groupby(['Cluster', 'AnimalID', 'Timepoint', 'Condition', 'Cylinder'])['frame'].nunique()
    return unique_count.reset_index(name='Count')

# Group by 'Cluster' and apply the counting function
result = count_unique_frames(df)
print(result[:10])
result.to_csv(root + '/cluster_metadata.csv', index=False)
print("CSV file has been created successfully.")
