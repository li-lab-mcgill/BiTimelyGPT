import pandas as pd

# Load the data
data_path = '../data/filtered_services_df_50.csv'
data = pd.read_csv(data_path)

path = "./Code/David_data/"
patient_info = pd.read_csv(path + 'patients.csv', index_col=0)
patient_info = patient_info[['month_of_birth']]
patient_info['month_of_birth'] = pd.to_datetime(patient_info['month_of_birth'])

# Merge the dataframes on the 'id' column
merged_data = pd.merge(data, patient_info, on='id', how='left')

# Convert date columns to datetime
merged_data['date'] = pd.to_datetime(merged_data['date'])
merged_data['month_of_birth'] = pd.to_datetime(merged_data['month_of_birth'], format='%Y-%m')

# Calculate the age in months at diagnosis
merged_data['age_at_diag'] = ((merged_data['date'] - merged_data['month_of_birth']) / pd.Timedelta(days=30)).round(2)
merged_data['age_at_diag'] = merged_data['age_at_diag'].round(0)

# Convert 'age_at_diag' column to integer
merged_data['age_at_diag'] = merged_data['age_at_diag'].astype(int)
merged_data.to_csv('../data/concurrent_pophr_data.csv', index=False)

# Drop duplicates for each patient based on 'age_at_diag'
merged_data = merged_data.drop_duplicates(subset=['id', 'age_at_diag'], keep='first')
merged_data.to_csv('../data/processed_pophr_data.csv', index=False)


