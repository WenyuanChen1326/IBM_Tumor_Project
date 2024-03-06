import pandas as pd
import numpy as np
import warnings
from utils import *
# Ignore all warnings
# warnings.filterwarnings("ignore")
import os
import ast
from tqdm import tqdm
import pandas as pd
import gc  # Import garbage collector interface
from sklearn.model_selection import train_test_split


def get_diagnosis(raw_meta_df):
    raw_df = raw_meta_df.copy()
    diagnosis_dict = {(row['patient_id'], row['study_id']): row['diagnosis'] for index, row in raw_df.iterrows()}
    return diagnosis_dict

def read_diagnosis_sample_patient_id(raw_meta_df,diagnosis = "NEGATIVE"):
    raw_df = raw_meta_df.copy()

    diagnosis_id_lst = raw_df[raw_df['diagnosis'] == diagnosis]['patient_id']
    diagnosis_study_lst = raw_df[raw_df['diagnosis'] == diagnosis]['study_id']
    return diagnosis_id_lst, diagnosis_study_lst

# create inputs for model:
def create_inputs(df, split, raw_all_data_directory = "/Volumes/T7 Shield/IBM/FDG-PET-CT-Lesions"):
    # raw_input_data = []  # List to collect data for raw_input_df
    # raw_input_label = []
    # scale_up_input_data = []  # List to collect data for scale_up_input_df
    # scale_up_input_label = []
     # Preallocate numpy arrays for efficiency
    num_samples = df.shape[0]
    raw_input_data = np.empty((num_samples,), dtype=object)  # Use dtype=object for arrays of arrays
    raw_input_label = np.empty((num_samples,), dtype=int)
    scale_up_input_data = np.empty((num_samples,), dtype=object)
    
    for index, (idx, row) in enumerate(tqdm(df.iterrows(), total=num_samples)):
    # count = 0
    # for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        patient_path = os.path.join(raw_all_data_directory, row['Patient ID'], row['Study ID'])
        block_coordinate = ast.literal_eval(row['Coordinate'])
        block_coordinate = tuple(map(int, block_coordinate))
        block_size = ast.literal_eval(row['Block Size'])
        block_size = tuple(map(int, block_size))
        
        # print(block_size[0])
        # break
        suv_path = os.path.join(patient_path, 'SUV.nii.gz')
        # seg_path = os.path.join(patient_path, 'SEG.nii.gz')
        # seg_data = nib.load(seg_path).get_fdata()
        # suv_data = nib.load(suv_path).get_fdata()
        suv_data = nib.load(suv_path).dataobj
        label = row['Positive Tumor']
        
        # seg_block = seg_data[int(block_coorindate[0]):block_coorindate[0]+block_size[0], block_coorindate[1]:block_coorindate[1]+block_size[1], block_coorindate[2]:block_coorindate[2]+block_size[2]]
        # assert np.sum(seg_block) > 0, f"Positive block has no tumor in {patient_path}"
        # assert np.sum(seg_block) == 0, f"Negative block has tumor in {patient_path}"

        # Prepare label
        label = 0 if int(label) == 0 else 1
        # suv_block = suv_data[block_coorindate[0]:block_coorindate[0]+block_size[0], block_coorindate[1]:block_coorindate[1]+block_size[1], block_coorindate[2]:block_coorindate[2]+block_size[2]]
       # Extract the relevant block without loading the entire dataset into memory
        suv_block = np.array(suv_data[slice(block_coordinate[0], block_coordinate[0] + block_size[0]),
                                       slice(block_coordinate[1], block_coordinate[1] + block_size[1]),
                                       slice(block_coordinate[2], block_coordinate[2] + block_size[2])])
        # raw_input_data.append(suv_block)
        # raw_input_label.append(label)

        scale_up_suv_block = scale_up_block(suv_block, new_resol= [224,224,224])
        # Store the data in preallocated arrays
        raw_input_data[index] = suv_block
        scale_up_input_data[index] = scale_up_suv_block
        raw_input_label[index] = int(row['Positive Tumor']) > 0
        # scale_up_input_data.append(scale_up_suv_block)

    # Save the data
    print(f'Saving {split} data')
    np.save(f'{split}_original_reso_dataset.npy', raw_input_data)
    np.save(f'{split}_original_reso_labels.npy', raw_input_label)


    np.save(f'{split}_scale_up_reso_dataset.npy', scale_up_input_data)
    np.save(f'{split}_scale_up_reso_labels.npy', raw_input_label)
    print(f'{split} data saved')

    # Explicitly call the garbage collector
    gc.collect()
    return {'raw_input_df': raw_input_data, 'raw_input_label': raw_input_label, 'scale_up_input_df': scale_up_input_data, 'scale_up_input_label': raw_input_label}
    # return {'raw_input_df': raw_input_data, 'scale_up_input_df': scale_up_input_data}

    # print(row['Block Size'])
    # break
    



if __name__ =="__main__":
    '''# Read the raw metadata
    path = '/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/Data/autoPETmeta.csv'
    raw_meta_df = pd.read_csv(path)
    raw_meta_df.fillna('unknown', inplace=True)
    raw_meta_df[['patient_id','study_id']] = raw_meta_df['study_location'].str.rsplit('/', n=2, expand=True)[[0,1]]
    raw_meta_df['patient_id'] = raw_meta_df['patient_id'].str.split('/').str[-1]

    patient_study_counts = raw_meta_df.groupby('patient_id')['study_id'].nunique()
    patients_with_multiple_studies = patient_study_counts[patient_study_counts > 1]

    # The length of this filtered series gives us the number of Patient IDs with more than 1 unique Study ID
    number_of_patients_more_than_1_study = len(patients_with_multiple_studies)
    number_of_patients_more_than_1_study
    print(f'There are {number_of_patients_more_than_1_study} patients with more than 1 study.')
    # Find patients with negative diagnosis
    negative_patients = set(raw_meta_df[raw_meta_df['diagnosis'] == 'NEGATIVE']['patient_id'])

    # Find patients with positive diagnosis
    positive_patients = set(raw_meta_df[raw_meta_df['diagnosis'] != 'NEGATIVE']['patient_id'])

    # Find patients that are in both negative diagnosis and other positive diagnosis
    cross_diagnosis_patients = negative_patients.intersection(positive_patients)

    cross_diagnosis_patients
    print(f"Number of patients that have more than 1 diagnosis: {len(cross_diagnosis_patients)}")
    diagnosis_dict = get_diagnosis(raw_meta_df = raw_meta_df)
    negative_id_lst, negative_study_lst = read_diagnosis_sample_patient_id(raw_meta_df)'''


    # path = '../corrected_all_patients_tumor_pos_neg_block.csv'
    # path = '../all_patients_sample_size_100_for_both_pos_neg.csv'
    path ='/Users/wenyuanchen/Desktop/IBM/IBM_Tumor_Project/all_patients_sample_size_100_for_both_pos_neg.csv'
    # path = '../all_patients_results.csv'
    raw_df = pd.read_csv(path)
    print(f'raw_df shape: {raw_df.shape}')
    raw_df.groupby(['Patient ID', 'Study ID']).size()
    num_unique_patients = raw_df.groupby(['Patient ID']).size()
    assert num_unique_patients.shape[0] == 900-1

    num_unique_studies = raw_df.groupby(['Patient ID', 'Study ID']).size()
    assert num_unique_studies.shape[0] == 1014-1

    # Create a new column called 'Diagnosis' in the raw_df
    tuples = list(zip(raw_df['Patient ID'], raw_df['Study ID']))
    # Now map these tuples to their corresponding diagnosis using the diagnosis_dict
    # raw_df['Diagnosis'] = pd.Series(tuples).map(diagnosis_dict)

    remove_all_0s = raw_df[~(raw_df['Block Size'] == '0')]


    # Load the dataset
    df = remove_all_0s
    # First split: Separate out the test set
    train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Positive Tumor'], random_state=42)
    # Second split: Split the remaining data into training and validation sets
    train_df, val_df = train_test_split(train_val_df, test_size=0.25, stratify=train_val_df['Positive Tumor'], random_state=42)
    # Note: 0.25 in the second split results in 20% of the original data because 0.25 * 0.8 = 0.2
    

    train = create_inputs(train_df, "train")
    val = create_inputs(val_df, "val")
    test = create_inputs(test_df, "test")


