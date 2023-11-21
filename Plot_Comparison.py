import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV data into a DataFrame
df = pd.read_csv('performance_data.csv')

# Convert 'Data Size' to a more readable format (e.g., KB, MB, GB)
def format_data_size(size_in_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} TB"

df['Data Size Formatted'] = df['Data Size'].apply(format_data_size)

# convert execution time to 10^-4 seconds scale
df['Execution Time (10^-4 s)'] = df['Execution Time'] * 10000

# Plot the data
plt.figure(figsize=(20, 8))  # 15 inches wide by 8 inches tall

# Plot execution time for each method
for method in df['Method'].unique():
    method_df = df[df['Method'] == method]
    plt.plot(method_df['Data Size Formatted'], method_df['Execution Time (10^-4 s)'], label=method, marker='o')

# Set the x-axis labels with a rotation for better readability
plt.xticks(rotation=45)

# Adding labels and title
plt.xlabel('Data Size')
plt.ylabel('Execution Time (10^-4 seconds)')
plt.title('Lab 6 : Comparison of the performance of Vector_Addition using memcpy, pinned memory, and UVM ')

# Adding a legend to the plot
plt.legend()

# Ensure layout fits the labels
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('performance_plot.png', dpi=300) # higher dpi higher resolution
