import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from layers.snippets import truncate_sequences
from sklearn.model_selection import train_test_split


class EHRDataset(Dataset):
    def __init__(self, data, max_length=256):
        self.max_length = max_length
        self.data = self._preprocess_data(data)

    def _preprocess_data(self, data):
        # Sort data by 'id' and 'date'
        df_sorted = data.sort_values(by=['id', 'date'])
        # Group data by 'id' and aggregate 'PheCode' and 'age_at_diag' into lists
        grouped = df_sorted.groupby('id').agg({
            'PheCode': list,
            'age_at_diag': list,
            'month_of_birth': 'first'  # optional, 'month_of_birth' is the same for each patient in a group
        }).reset_index()
        return grouped

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        :param idx: patient id
        :return: icd: a sequence of token ids, date: a sequence of [year, month, date], age: a sequence of ages (year-level)
        '''
        patient = self.data.iloc[idx]
        # tokenize, truncate, and pad tokens
        seq_phecodes = [phecode for phecode in patient.PheCode]
        if len(seq_phecodes) > self.max_length: # keep the last few max_length number of tokens
            seq_phecodes = truncate_sequences(self.max_length, seq_phecodes)[0]
        seq_phecodes.extend([0] * (self.max_length - len(seq_phecodes))) # pad

        # tokenize, truncate, and pad ages
        seq_ages = [0] + [age for age in patient.age_at_diag] # Initial age for [CLS] token is 0
        if len(seq_ages) > self.max_length:
            seq_ages = truncate_sequences(self.max_length, seq_ages)[0] # keep the last few max_length number of tokens
        # pad with the age year
        # birth_year = int(patient.month_of_birth.split('-')[0])
        # birth_month = int(patient.month_of_birth.split('-')[1])
        # current_year = datetime.now().year
        # current_month = datetime.now().month
        # age_at_end_months = (current_year - birth_year) * 12 + (current_month - birth_month)
        # age_at_end_months = min(age_at_end_months, 120 * 12)
        # pad with the age at the last token
        age_at_end_months = seq_ages[-1]
        seq_ages.extend([age_at_end_months] * (self.max_length - len(seq_ages)))  # pad with end age

        seq_tokens_tensor = torch.tensor(seq_phecodes) # [sos] token will add to first position by adding a fixed embeding
        seq_ages_tensor = torch.tensor(seq_ages)
        return seq_tokens_tensor, seq_ages_tensor


def data_provider(batch_size,
                  max_length,
                  train_ratio=0.8,
                  valid_ratio=0.1):
    total_data = pd.read_csv('data/processed_pophr_data.csv', nrows=10000)
    # total_data = pd.read_csv('data/processed_pophr_data.csv')

    # Getting unique patient IDs and splitting them
    patient_ids = total_data['id'].unique()
    train_ids, temp_ids = train_test_split(patient_ids, test_size=1-train_ratio, random_state=42)
    valid_ids, test_ids = train_test_split(temp_ids, test_size=1-(valid_ratio/(1-train_ratio)), random_state=42)
    print(len(train_ids), len(valid_ids), len(test_ids))

    # Using the split IDs to filter the original DataFrame
    train_data = total_data[total_data['id'].isin(train_ids)]
    valid_data = total_data[total_data['id'].isin(valid_ids)]
    test_data = total_data[total_data['id'].isin(test_ids)]

    train_dataset = EHRDataset(train_data, max_length=max_length)
    valid_dataset = EHRDataset(valid_data, max_length=max_length)
    test_dataset = EHRDataset(test_data, max_length=max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, train_dataloader, valid_dataset, valid_dataloader, test_dataset, test_dataloader

# if __name__ == "__main__":
#     max_length = 10
#     batch_size = 10
#     train_dataset, train_dataloader, valid_dataset, valid_dataloader, test_dataset, test_dataloader = data_provider(batch_size, max_length)
#     # Iterating over training minibatches
#     for batch_idx, (phecode, age) in enumerate(train_dataloader):
#         # Your operations on the minibatch go here
#         print(f"Batch {batch_idx + 1} - Phecode: {phecode}, Age: {age}")
#         print(phecode.shape)
