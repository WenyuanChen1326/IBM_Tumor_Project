# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision.models.video import r3d_18

# Load the pretrained 3D ResNet model
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

def most_common_diagnosis(diagnoses):
    if not diagnoses:
        return None
    return Counter(diagnoses).most_common(1)[0][0]

def create_stratified_patient_splits(metadata_csv_path, output_dir, test_size=0.2, val_size=0.1):
    # Load metadata
    df = pd.read_csv(metadata_csv_path)
    
    # Extract new patient_id and study_id from the 'study_location' field
    df['patient_id'] = df['study_location'].apply(lambda x: x.split('/')[2])
    df['study_id'] = df['study_location'].apply(lambda x: x.split('/')[3])
    
    # Use the new patient_id for further grouping
    patient_diagnoses = df.groupby('patient_id')['diagnosis'].agg(list).reset_index()
    
    # Determine the most common diagnosis for stratification
    patient_diagnoses['most_common_diagnosis'] = patient_diagnoses['diagnosis'].apply(most_common_diagnosis)
    
    # Stratify based on the most common diagnosis
    patient_ids = patient_diagnoses['patient_id']
    stratify_on = patient_diagnoses['most_common_diagnosis']
    
    # Split the data into train and test sets while maintaining the distribution of the most common diagnosis
    train_ids, test_ids = train_test_split(patient_ids, test_size=test_size, 
                                           stratify=stratify_on, random_state=42)
    
    # Further split the train set into training and validation sets
    train_ids, val_ids = train_test_split(train_ids, test_size=val_size / (1 - test_size), 
                                          stratify=stratify_on[train_ids.index], random_state=42)
    
    # Assign split labels
    patient_diagnoses['split'] = 'test'
    patient_diagnoses.loc[patient_diagnoses['patient_id'].isin(train_ids), 'split'] = 'train'
    patient_diagnoses.loc[patient_diagnoses['patient_id'].isin(val_ids), 'split'] = 'val'
    
    # Merge the split information back into the original DataFrame
    df_with_splits = df.merge(patient_diagnoses[['patient_id', 'split']], on='patient_id', how='left')
    
    # Save the updated DataFrame with split information to a CSV file
    os.makedirs(output_dir, exist_ok=True)
    df_with_splits.to_csv(os.path.join(output_dir, 'data_with_splits.csv'), index=False)
    
    # Return the DataFrame with splits for further processing if needed
    return df_with_splits

# Run the function with your data
metadata_csv_path = '/home/ubuntu/jupyter-sandy/3D_ResNet/Data/autoPETmeta.csv' # Replace with your actual path
output_dir = '/home/ubuntu/jupyter-sandy/3D_ResNet/Data/Data_Split'  # Replace with your actual path
df_with_splits = create_stratified_patient_splits(metadata_csv_path, output_dir)

# Run the function with your data
metadata_csv_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/autoPETmeta.csv'
output_dir = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data_Split'
df_with_splits = create_stratified_patient_splits(metadata_csv_path, output_dir)


# Create the splits
# # train_df, val_df, test_df = create_stratified_patient_splits(metadata_csv_path, output_dir)
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator

# # Load the data
# train_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data_Split/train_split.csv'
# val_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data_Split/val_split.csv'
# test_path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data_Split/test_split.csv'
# train_df = pd.read_csv(train_path)
# val_df = pd.read_csv(val_path)
# test_df = pd.read_csv(test_path)

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np


'''
plot patient level disease distribution
# # Aggregate the counts for each disease in each split
# train_counts = train_df['diagnosis'].value_counts().sort_index()
# val_counts = val_df['diagnosis'].value_counts().sort_index()
# test_counts = test_df['diagnosis'].value_counts().sort_index()

# # Ensure all diseases are present in each split for consistent plotting
# all_diseases = set(train_counts.index) | set(val_counts.index) | set(test_counts.index)
# train_counts = train_counts.reindex(all_diseases, fill_value=0)
# val_counts = val_counts.reindex(all_diseases, fill_value=0)
# test_counts = test_counts.reindex(all_diseases, fill_value=0)

# # Get a sorted list of all diseases
# sorted_diseases = sorted(all_diseases)

# # Plot settings
# bar_width = 0.2
# index = np.arange(3)
# fig, ax = plt.subplots()

# # Plotting each disease count as a separate bar
# for i, disease in enumerate(sorted_diseases):
#     counts = [train_counts[disease], val_counts[disease], test_counts[disease]]
#     ax.bar(index + i * bar_width, counts, bar_width, label=disease)

# # Set the position and labels for the X ticks
# ax.set_xticks(index + bar_width / 2 * (len(sorted_diseases) - 1))
# ax.set_xticklabels(['Train', 'Validation', 'Test'])

# # Adding labels and title
# ax.set_xlabel('Dataset Split')
# ax.set_ylabel('Count')
# ax.set_title('Disease Distribution Across Dataset Splits')

# # Adding a legend and adjusting the layout to fit the plot
# ax.legend(title='Disease', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()

# # Show the plot
# plt.show()
'''
