import os
import json
import re
import statistics

# Define the base log directory
log_dir = "../log"

# Dictionary to store experiment data
experiment_data = {}

# Regular expressions for parsing log information
param_re = re.compile(r"Model parameter number ([\d\.]+)M")
input_re = re.compile(r"Input size ([\d\.]+)MB")
output_re = re.compile(r"Output size ([\d\.]+)MB")
duration_re = re.compile(r"duration ([\d\.]+)s")
power_re = re.compile(r"VDD_IN (\d+)mW")  # Extracts VDD_IN values in milliwatts

# Iterate through each subdirectory
for sub_dir in sorted(os.listdir(log_dir)):
    sub_dir_path = os.path.join(log_dir, sub_dir)

    # Ensure it's a directory
    if os.path.isdir(sub_dir_path):
        # Extract method, environment, and model name
        parts = sub_dir.split("_", 2)
        if len(parts) == 3:
            method, environment, model_name = parts
        else:
            print(f"Skipping directory with unexpected structure: {sub_dir}")
            continue

        # Initialize data structure for this experiment
        experiment_data.setdefault(environment, {}).setdefault(model_name, {})[method] = {}

        # Iterate through log files
        for log_file in os.listdir(sub_dir_path):
            log_file_path = os.path.join(sub_dir_path, log_file)

            if log_file in ["inference_node.log", "kapao_test.log", "run_torchvision.log"]:
                # Read main log file
                with open(log_file_path, "r") as f:
                    content = f.readlines()

                # Extract data
                durations = []
                model_params = None
                input_size = None
                output_size = None

                for line in content:
                    if param_match := param_re.search(line):
                        model_params = float(param_match.group(1))  # Convert to float
                    
                    if input_match := input_re.search(line):
                        input_size = float(input_match.group(1))  # Convert to float
                    
                    if output_match := output_re.search(line):
                        output_size = float(output_match.group(1))  # Convert to float
                    
                    if duration_match := duration_re.search(line):
                        durations.append(float(duration_match.group(1)) * 1000)  # Convert seconds to milliseconds

                # Compute statistics
                avg_duration = statistics.mean(durations) if durations else None
                std_duration = statistics.stdev(durations) if len(durations) > 1 else None

                # Store in dictionary
                experiment_data[environment][model_name][method].update({
                    "model_params": model_params,  # in millions
                    "input_size": input_size,  # in MB
                    "output_size": output_size,  # in MB
                    "avg_forward_duration_ms": avg_duration,  # average forward pass duration in ms
                    "std_forward_duration_ms": std_duration  # standard deviation in ms
                })

            elif log_file == "power_consumption.log":
                # Read power log file
                with open(log_file_path, "r") as f:
                    power_data = f.readlines()

                power_values = []
                for line in power_data:
                    if power_match := power_re.search(line):
                        power_values.append(int(power_match.group(1)))  # Convert to int (mW)

                # Compute power consumption statistics
                avg_power = statistics.mean(power_values) if power_values else None
                std_power = statistics.stdev(power_values) if len(power_values) > 1 else None

                # Store in dictionary
                experiment_data[environment][model_name][method]["avg_power_mW"] = avg_power  # in milliwatts
                experiment_data[environment][model_name][method]["std_power_mW"] = std_power  # in milliwatts

        # Print extracted data for this experiment
        print("\n" + "=" * 50)
        print(f"🏷  Environment: {environment} | Method: {method} | Model: {model_name}")
        print("-" * 50)
        for key, value in experiment_data[environment][model_name][method].items():
            print(f"🔹 {key}: {value}")
        print("=" * 50 + "\n")

print("Processing complete. Saving results...")

# Save as JSON
output_file = "experiment_results.json"
with open(output_file, "w") as json_file:
    json.dump(experiment_data, json_file, indent=4)

print(f"✅ Results saved to {output_file}")