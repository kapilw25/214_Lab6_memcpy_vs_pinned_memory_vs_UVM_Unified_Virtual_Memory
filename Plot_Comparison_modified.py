import pandas as pd

# Function to format data size
def format_data_size(size_in_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} TB"

# Read the original CSV file
df = pd.read_csv('performance_data.csv')

# Convert 'Data Size' in the original DataFrame
df['Data Size (Bytes)'] = df['DataSize'].apply(format_data_size)

# Drop redundant columns from the original DataFrame
df.drop(['DataSize'], axis=1, inplace=True)

# Split the DataFrame based on the method
memcpy_df = df[df['Method'] == 'memcpy']
pinned_df = df[df['Method'] == 'pinned']
uvm_df = df[df['Method'] == 'uvm']

# Convert 'Execution Time' and round off to 2 decimal places
for dataframe in [memcpy_df, pinned_df, uvm_df]:
    dataframe['Execution Time (10^-4 s)'] = (dataframe['ExecutionTime'] * 10000).round(2)
    dataframe.drop(['Method', 'ExecutionTime'], axis=1, inplace=True)

# Remove duplicate 'Data Size (Bytes)' columns from pinned and uvm dataframes
pinned_df.drop(['Data Size (Bytes)'], axis=1, inplace=True)
uvm_df.drop(['Data Size (Bytes)'], axis=1, inplace=True)

# Rename columns for clarity
memcpy_df.rename(columns={'Execution Time (10^-4 s)': 'Execution Time (10^-4 s) - Memcpy'}, inplace=True)
pinned_df.rename(columns={'Execution Time (10^-4 s)': 'Execution Time (10^-4 s) - Pinned'}, inplace=True)
uvm_df.rename(columns={'Execution Time (10^-4 s)': 'Execution Time (10^-4 s) - UVM'}, inplace=True)

# Reset index for joining
memcpy_df.reset_index(drop=True, inplace=True)
pinned_df.reset_index(drop=True, inplace=True)
uvm_df.reset_index(drop=True, inplace=True)

# Join the DataFrames horizontally
final_df = pd.concat([memcpy_df, pinned_df, uvm_df], axis=1)

# Save the final DataFrame as a new CSV file
final_df.to_csv('modified_performance_data.csv', index=False)
