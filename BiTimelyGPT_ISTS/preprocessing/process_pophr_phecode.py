import time
import pandas as pd

# Load the CSV files into dataframes
path = "./Code/David_data/"
services_df = pd.read_csv(path + 'services.csv')
services_df.dropna(inplace=True)
services_df = services_df[['id', 'date', 'icd']]

# read the PheWAS
mapping = pd.read_csv('phecode_icd9_rolled.csv')
mapping['ICD9'] = mapping['ICD9'].str.replace('.', '')
mapping['ICD9'] = mapping['ICD9'].astype(str)
# make the ICD9 column if only 3 characters, add a 0 at the end
# if there are ICDs with more than 4 numbers, only keep the first 4 characters
mapping['ICD9'] = mapping['ICD9'].apply(lambda x: x+'0' if len(x)==3 else x[:4])
# remove the duplicates ICDs
mapping = mapping.drop_duplicates(subset=['ICD9'])
# only keep the first 4 number
mapping = mapping.iloc[:, :4]
# change the name of the column ICD9 to icd
mapping=mapping.rename(columns={'ICD9':'icd'})
# for all PheCode, only keep the integer part, remove the decimal part, optional
mapping['PheCode']=mapping['PheCode'].astype(str).apply(lambda x: x.split('.')[0])
mapping = mapping[['icd', 'PheCode', 'Phenotype']]

# associate patient records with PheCodes
all_services = pd.merge(services_df, mapping, on='icd', how='left')
all_services.dropna(inplace=True)
all_services.to_csv('all_services.csv', index=False)

# Count the frequency of each PheCode
phecode_counts = all_services['PheCode'].value_counts()
# Calculate the frequency as the count over all records
total_records = len(all_services)
phecode_frequency = phecode_counts / total_records

# Combine counts and frequency into a DataFrame and sort by counts
phecode_summary = pd.DataFrame({'PheCode': phecode_counts.index, 'Count': phecode_counts.values, 'Frequency': phecode_frequency.values})
phecode_summary_sorted = phecode_summary.sort_values(by='Count', ascending=False)
phecode_summary_sorted.to_csv('phecode_summary_sorted.csv', index=False)
phecode_filtered = phecode_summary_sorted[phecode_summary_sorted['Frequency'] > 0.001]

# read all_services.csv
all_services = pd.read_csv('all_services.csv')
# filter the patient records with respect to only the phecode_filtered (frequent phecodes)
filtered_services_df = all_services[all_services['PheCode'].isin(phecode_filtered.PheCode)]
filtered_services_df.to_csv('filtered_services_df.csv', index=False)
print(filtered_services_df)

# filter patients with at least 50 phecodes, and count the number of patients
# Count the number of records for each patient
patient_counts = filtered_services_df['id'].value_counts()
# Get the list of patient IDs that have 50 or more records
patients_with_50_or_more_records = patient_counts[patient_counts >= 50].index.tolist()
# Filter the dataframe to only include these patients
filtered_services_df = filtered_services_df[filtered_services_df['id'].isin(patients_with_50_or_more_records)]
print(filtered_services_df)
# output the records from the patients with at least 50 phecodes
filtered_services_df.to_csv('filtered_services_df_50.csv', index=False)
print(filtered_services_df.columns)
print(len(filtered_services_df.PheCode.unique()))

