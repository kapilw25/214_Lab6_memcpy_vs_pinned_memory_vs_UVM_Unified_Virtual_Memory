import subprocess
import csv
import re
import os

# Function to parse the execution time from the program output
def parse_output(output):
    try:
        # Assuming the execution time is printed as "Elapsed time: X.XXX seconds"
        match = re.search(r"Elapsed time: ([\d.]+) seconds", output.decode('utf-8'))
        if match:
            return float(match.group(1))
        else:
            raise ValueError("Execution time not found in the output")
    except Exception as e:
        print(f"Error parsing output: {e}")
        return None

# Determine the directory where the script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

# Paths to the CUDA executables
executables = {
    "memcpy": os.path.join(script_dir, "vector_add_Memcpy"),
    "pinned": os.path.join(script_dir, "vector_add_Pinned"),
    "uvm": os.path.join(script_dir, "vector_add_UVM")
}

# Data sizes (32B to 2GB)
data_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648]

results = []

for method, executable in executables.items():
    for size in data_sizes:
        # Run the CUDA program
        command = [executable, str(size)]  # Assuming the executable takes data size as an argument
        try:
            output = subprocess.run(command, capture_output=True, check=True)
            execution_time = parse_output(output.stdout)

            if execution_time is not None:
                # Record the result
                results.append([method, size, execution_time])
            else:
                print(f"Failed to get execution time for {method} with size {size}")
        except subprocess.CalledProcessError as e:
            print(f"Error running {command}: {e}")

# Write results to CSV
with open('performance_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Method', 'DataSize', 'ExecutionTime'])
    for row in results:
        writer.writerow(row)
